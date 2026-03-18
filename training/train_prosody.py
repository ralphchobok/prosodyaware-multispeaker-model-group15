"""
Training Script for Prosody-Aware Multi-Speaker LightSpeech

This script trains the prosody-aware LightSpeech model with emotion conditioning.
It can either:
1. Fine-tune an existing pretrained model (recommended)
2. Train from scratch

Usage:
    # Fine-tune from existing model
    python train_prosody.py --pretrained model.pt
    
    # Train from scratch
    python train_prosody.py --from-scratch

Emotion Mapping:
    0: Angry
    1: Happy
    2: Neutral
    3: Sad
    4: Surprise
"""

from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import sys
import os
import time
import logging
import random
import numpy as np
import glob
import pandas as pd
import whisper
import jiwer
from torch import optim
from typing import List, Tuple
from scipy.io import wavfile
from collections import defaultdict
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add preprocess_datasets directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocess_datasets'))

from process_ljspeech import VOCODER, VOCODER_NAME, TRIM_SILENCE

# Update OUTPUT_PATH to point to the correct location
OUTPUT_PATH = str(Path(__file__).parent.parent / 'preprocessed_datasets' / 'processed')

from torch.nn import MSELoss, L1Loss

from lightspeech_prosody import ProsodyAwareModel
from lightspeech import Model as BaseLightSpeech

DEVICE = "cuda:1"  # Using GPU 1 which has more free memory
SEED = 3
EPOCHS = 200  # More epochs for emotion learning
WARMUP = 10
LR_RATE = 5e-4  # Lower LR for fine-tuning
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAINING_SPLIT = 0.15  # More validation data for better evaluation

WHISPER_SIZE = "tiny"

# Emotion mapping
EMOTION_TO_ID = {
    'Angry': 0,
    'Happy': 1,
    'Neutral': 2,
    'Sad': 3,
    'Surprise': 4
}
ID_TO_EMOTION = {v: k for k, v in EMOTION_TO_ID.items()}
NUM_EMOTIONS = len(EMOTION_TO_ID)


def setup_logger(log_file="training_prosody.log"):
    logger = logging.getLogger("prosody_training_logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Set up the logger
logger = setup_logger()


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Deterministic algorithms disabled for CuBLAS compatibility
    # Set CUBLAS_WORKSPACE_CONFIG=:4096:8 if you need full determinism
    torch.use_deterministic_algorithms(False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WarmupLinearSchedule(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then linear decay."""

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


class EmotionalCustomDataset(Dataset):
    """
    Dataset with emotion labels from preprocessed .pt files.
    
    Expected structure:
    processed/SPEAKER_ID/filename.pt (emotion is stored in the .pt file)
    """
    def __init__(self, files: List[str], periodicity_range=[], pitch_mean_std=[]):
        self.files = files

        if not periodicity_range or not pitch_mean_std:
            self.periodicity_range = [float("inf"), float("-inf")]
            self.pitch_mean_std = [0.0, 0.0]
            self._compute_statistics()
            logger.info(
                f"Pitch mean/std: {self.pitch_mean_std[0]:.4f}, {self.pitch_mean_std[1]:.4f}"
            )
            logger.info(
                f"Periodicity range min/max: {self.periodicity_range[0]:.4f}, {self.periodicity_range[1]:.4f}"
            )
        else:
            self.periodicity_range = periodicity_range
            self.pitch_mean_std = pitch_mean_std

    def _compute_statistics(self):
        count = 0
        mean = 0.0
        M2 = 0.0

        for filename in tqdm(self.files, desc="Computing statistics"):
            try:
                data = torch.load(filename, weights_only=False)
                pitch = data["pitch"]
                periodicity = data["pitch_periodicity"]

                valid_pitch = pitch[periodicity > 0.5]
                if len(valid_pitch) == 0:
                    continue

                for value in valid_pitch:
                    count += 1
                    delta = value - mean
                    mean += delta / count
                    delta2 = value - mean
                    M2 += delta * delta2

                self.periodicity_range[0] = min(
                    self.periodicity_range[0], periodicity.min().item()
                )
                self.periodicity_range[1] = max(
                    self.periodicity_range[1], periodicity.max().item()
                )

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

        if count > 1:
            variance = M2 / (count - 1)
            std_dev = np.sqrt(variance.item())
            self.pitch_mean_std = [mean.item(), std_dev]
        else:
            logger.warning("Warning: Insufficient data to compute statistics.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.files[idx], weights_only=False)
        
        # Emotion is already stored in the file, no need to extract from path
        # Just validate it exists and is a valid emotion ID
        if 'emotion' not in data:
            logger.warning(f"No emotion found in {self.files[idx]}, defaulting to Neutral")
            data['emotion'] = EMOTION_TO_ID['Neutral']
        elif data['emotion'] not in range(NUM_EMOTIONS):
            logger.warning(f"Invalid emotion ID {data['emotion']} in {self.files[idx]}, defaulting to Neutral")
            data['emotion'] = EMOTION_TO_ID['Neutral']
        
        return data

    @staticmethod
    def pad_tensors(data: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        if not data:
            raise ValueError("Cannot pad empty list")

        max_len = max(d.shape[0] for d in data)
        if data[0].dim() == 1:
            padded_data = torch.full(
                (len(data), max_len), pad_value, dtype=data[0].dtype
            )
            for i, d in enumerate(data):
                padded_data[i, : d.shape[0]] = d
        elif data[0].dim() == 2:
            padded_data = torch.full(
                (len(data), max_len, data[0].shape[1]), pad_value, dtype=data[0].dtype
            )
            for i, d in enumerate(data):
                padded_data[i, : d.shape[0], :] = d
        else:
            raise ValueError(f"Unsupported tensor dimension: {data[0].dim()}")

        return padded_data

    def collate_fn(self, batch: List[dict]) -> Tuple[torch.Tensor, ...]:
        speakers = torch.tensor([b["speaker"] for b in batch])
        emotions = torch.tensor([b["emotion"] for b in batch])  # NEW: Emotion IDs
        
        # Convert lists to tensors if needed (for encoded_text and encoded_tone)
        texts = self.pad_tensors([
            torch.tensor(b["encoded_text"]) if isinstance(b["encoded_text"], list) else b["encoded_text"]
            for b in batch
        ])
        tones = self.pad_tensors([
            torch.tensor(b["encoded_tone"]) if isinstance(b["encoded_tone"], list) else b["encoded_tone"]
            for b in batch
        ])

        pitches = self.pad_tensors(
            [
                (b["pitch"] - self.pitch_mean_std[0]) / self.pitch_mean_std[1]
                for b in batch
            ]
        )
        periodicity = self.pad_tensors(
            [
                (b["pitch_periodicity"] - self.periodicity_range[0])
                / (self.periodicity_range[1] - self.periodicity_range[0])
                for b in batch
            ]
        ).float()
        durations_rounded = self.pad_tensors([b["duration"] for b in batch])
        mels = self.pad_tensors([b["mel"] for b in batch])

        padding_mask_pitch = self.pad_tensors(
            [torch.ones_like(b["pitch"]) for b in batch]
        ).bool()
        padding_mask_mel = self.pad_tensors(
            [torch.ones_like(b["mel"]) for b in batch]
        ).bool()
        padding_mask_dur = self.pad_tensors(
            [torch.ones_like(b["duration"]) for b in batch]
        ).bool()

        return (
            speakers,
            emotions,  # NEW: Added emotions
            texts,
            tones,
            pitches,
            periodicity,
            durations_rounded,
            mels,
            padding_mask_pitch,
            padding_mask_mel,
            padding_mask_dur,
        )


def train_one_epoch(model, train_loader, optimizer, scaler, scheduler, gradient_accumulation_steps=1):
    model.train()
    mse_loss = MSELoss(reduction="none")
    l1_loss = L1Loss(reduction="none")
    total_losses = defaultdict(float)

    optimizer.zero_grad()
    
    for batch_idx, audio in enumerate(tqdm(train_loader, desc="Training")):
        audio = [k.to(DEVICE) for k in audio]
        (
            speakers,
            emotions,  # NEW
            texts,
            tones,
            pitches,
            periodicity,
            durations,
            mels,
            padding_mask_pitch,
            padding_mask_mel,
            padding_mask_dur,
        ) = audio

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            mel_pred, dur_pred, pitch_pred, periodicity_pred = model(
                speakers, texts, tones, emotions, pitches, periodicity, durations, mels
            )

            mel_loss = l1_loss(mel_pred, mels)[padding_mask_mel].mean()
            dur_loss = mse_loss(dur_pred, torch.log1p(durations.float()))[
                padding_mask_dur
            ].mean()
            pitch_loss = (periodicity * mse_loss(pitch_pred, pitches))[
                padding_mask_pitch
            ].mean()
            periodicity_loss = mse_loss(periodicity_pred, periodicity)[
                padding_mask_pitch
            ].mean()

            loss_all = (mel_loss + dur_loss + pitch_loss + periodicity_loss) / gradient_accumulation_steps

        scaler.scale(loss_all).backward()
        
        # Update weights only after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        batch_size = speakers.size(0)
        for loss_name, loss_value in [
            ("train_mel_loss", mel_loss),
            ("train_dur_loss", dur_loss),
            ("train_pitch_loss", pitch_loss),
            ("train_periodicity_loss", periodicity_loss),
        ]:
            total_losses[loss_name] += loss_value.item() * batch_size

    total_samples = len(train_loader.dataset)
    return {k: v / total_samples for k, v in total_losses.items()}


def val_one_epoch(model, val_loader):
    model.eval()
    mse_loss = MSELoss(reduction="none")
    l1_loss = L1Loss(reduction="none")
    total_losses = defaultdict(float)
    emotion_losses = defaultdict(lambda: defaultdict(float))
    emotion_counts = defaultdict(int)

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=True, dtype=torch.float16
    ):
        for audio in tqdm(val_loader, desc="Validation"):
            audio = [k.to(DEVICE) for k in audio]
            (
                speakers,
                emotions,  # NEW
                texts,
                tones,
                pitches,
                periodicity,
                durations,
                mels,
                padding_mask_pitch,
                padding_mask_mel,
                padding_mask_dur,
            ) = audio

            mel_pred, dur_pred, pitch_pred, periodicity_pred = model(
                speakers, texts, tones, emotions, pitches, periodicity, durations, mels
            )

            mel_loss = l1_loss(mel_pred, mels)[padding_mask_mel].mean()
            dur_loss = mse_loss(dur_pred, torch.log1p(durations.float()))[
                padding_mask_dur
            ].mean()
            pitch_loss = (periodicity * mse_loss(pitch_pred, pitches))[
                padding_mask_pitch
            ].mean()
            periodicity_loss = mse_loss(periodicity_pred, periodicity)[
                padding_mask_pitch
            ].mean()

            batch_size = speakers.size(0)
            for loss_name, loss_value in [
                ("val_mel_loss", mel_loss),
                ("val_dur_loss", dur_loss),
                ("val_pitch_loss", pitch_loss),
                ("val_periodicity_loss", periodicity_loss),
            ]:
                total_losses[loss_name] += loss_value.item() * batch_size
            
            # Track per-emotion losses
            for i, emotion_id in enumerate(emotions.cpu().numpy()):
                emotion_name = ID_TO_EMOTION[emotion_id]
                emotion_losses[emotion_name]["mel"] += mel_loss.item()
                emotion_losses[emotion_name]["pitch"] += pitch_loss.item()
                emotion_counts[emotion_name] += 1

    total_samples = len(val_loader.dataset)
    losses = {k: v / total_samples for k, v in total_losses.items()}
    losses["val_total_loss"] = sum(losses.values())
    
    # Average per-emotion losses
    for emotion_name in emotion_losses:
        count = emotion_counts[emotion_name]
        for loss_type in emotion_losses[emotion_name]:
            losses[f"val_{emotion_name.lower()}_{loss_type}_loss"] = (
                emotion_losses[emotion_name][loss_type] / count
            )

    return losses


def load_pretrained_model(pretrained_path, prosody_model):
    """
    Load weights from a pretrained base LightSpeech model.
    
    Only compatible weights are loaded. New emotion-related parameters
    are randomly initialized.
    """
    logger.info(f"Loading pretrained model from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, weights_only=False)
    pretrained_state = checkpoint.get('state_dict', checkpoint)
    
    # Get model state dict
    model_state = prosody_model.state_dict()
    
    # Filter and load compatible weights
    compatible_weights = {}
    incompatible_keys = []
    
    for key, value in pretrained_state.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                compatible_weights[key] = value
            else:
                incompatible_keys.append(f"{key} (shape mismatch)")
        else:
            incompatible_keys.append(f"{key} (not in new model)")
    
    # Load compatible weights
    prosody_model.load_state_dict(compatible_weights, strict=False)
    
    logger.info(f"Loaded {len(compatible_weights)} compatible weight tensors")
    if incompatible_keys:
        logger.info(f"Skipped {len(incompatible_keys)} incompatible keys:")
        for key in incompatible_keys[:10]:  # Show first 10
            logger.info(f"  - {key}")
        if len(incompatible_keys) > 10:
            logger.info(f"  ... and {len(incompatible_keys) - 10} more")
    
    return prosody_model


def parse_speakers(filename):
    speaker_df = pd.read_csv(filename, sep="\t", dtype={'name': str})  # Keep name as string
    dicts_list = speaker_df.to_dict(orient="records")
    return dicts_list, speaker_df["name"]


def get_train_val_files(
    file_list, speaker_ids, unique_speakers, test_size, random_state=42
):
    """Generate train and validation files using a stratified split."""
    train_files = []
    val_files = []

    def text_length_score(text):
        """Calculate the text length score, excluding specific characters."""
        return len(text.replace("<sil>", ""))

    # Convert speaker_ids to regular strings for comparison
    speaker_ids_str = np.array([str(sid) for sid in speaker_ids])
    
    for speaker_id in tqdm(unique_speakers, desc="Splitting files"):
        speaker_id_str = str(speaker_id)
        speaker_files = np.array(file_list)[speaker_ids_str == speaker_id_str]
        logger.debug(f"Speaker {speaker_id_str}: found {len(speaker_files)} files")
        speaker_data = []

        for file_path in speaker_files:
            try:
                data = torch.load(file_path, weights_only=False)
                text = data.get("original_text", "")
                text_score = text_length_score(text)
                speaker_data.append((file_path, text_score))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        if speaker_data:
            files, scores = zip(*speaker_data)
            train_f, val_f = train_test_split(
                list(files), test_size=test_size, random_state=random_state, 
                stratify=[s // 10 for s in scores]  # Stratify by text length bins
            )
            train_files.extend(train_f)
            val_files.extend(val_f)

    return train_files, val_files


def main():
    global DEVICE
    
    import argparse
    parser = argparse.ArgumentParser(description="Train prosody-aware LightSpeech")
    parser.add_argument("--pretrained", type=str, default="../models/model.pt",
                      help="Path to pretrained model checkpoint")
    parser.add_argument("--from-scratch", action="store_true",
                      help="Train from scratch instead of fine-tuning")
    parser.add_argument("--output", type=str, default="../models/model2.pt",
                      help="Output model path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                      help=f"Batch size for training (default: {BATCH_SIZE})")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                      help="Number of gradient accumulation steps (default: 1)")
    parser.add_argument("--device", type=str, default=DEVICE,
                      help=f"Device to use for training (default: {DEVICE})")
    args = parser.parse_args()
    
    # Override global settings with command-line arguments
    DEVICE = args.device
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    start_time = time.time()

    # Load file list - filter to only include EDS files (in speaker folders like 0011-0020)
    all_files = sorted(glob.glob(os.path.join(OUTPUT_PATH, "**", "*pt"), recursive=True))
    
    # Filter out LJ dataset files - only keep files in 4-digit speaker folders
    # File structure: processed/SPEAKER_ID/filename.pt (emotion is stored in the file)
    file_list = []
    for f in all_files:
        # Get speaker folder name (immediate parent directory)
        speaker_folder = os.path.basename(os.path.dirname(f))
        # Only include files in 4-digit speaker folders (0011-0020), exclude LJ
        if speaker_folder.isdigit() and len(speaker_folder) == 4 and speaker_folder != 'LJ':
            file_list.append(f)
    
    file_list = np.asarray(file_list)
    logger.info(f"Found {len(file_list)} EDS files in {OUTPUT_PATH} (filtered from {len(all_files)} total)")
    
    # Extract speaker IDs from file paths: processed/SPEAKER_ID/filename.pt
    # Get immediate parent directory basename
    speaker_ids = np.asarray([os.path.basename(os.path.dirname(f)) for f in file_list])
    
    speaker_dict, unique_speakers = parse_speakers(
        os.path.join(OUTPUT_PATH, "speakers_esd.tsv")
    )
    num_speakers = len(unique_speakers)
    logger.info(f"Number of speakers: {num_speakers}")
    logger.info(f"Unique speaker IDs in dataset: {sorted(set(speaker_ids))}")
    logger.info(f"Speaker names from TSV: {list(unique_speakers)}")

    train_files, val_files = get_train_val_files(
        file_list, speaker_ids, unique_speakers, TRAINING_SPLIT
    )
    logger.info(
        f"Training files: {len(train_files)}, validation files: {len(val_files)}"
    )
    logger.info(f"Batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"Using device: {DEVICE}")

    seed_all(SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = EmotionalCustomDataset(train_files)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=train_dataset.collate_fn,
        generator=g,
    )

    val_dataset = EmotionalCustomDataset(
        val_files, train_dataset.periodicity_range, train_dataset.pitch_mean_std
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=val_dataset.collate_fn,
        generator=g,
    )

    train_epoch_steps = len(train_loader)

    # Load phone vocabulary
    phone_df = pd.read_csv(f"{OUTPUT_PATH}/phones_esd.tsv", sep="\t", keep_default_na=False)
    phone_dict = phone_df.set_index("text")["phone_id"].to_dict()
    num_phones = phone_df["phone_id"].max() + 1
    logger.info(f"Number of phones: {num_phones}")
    
    # Load words dictionary
    words_df = pd.read_csv(f"{OUTPUT_PATH}/words_esd.tsv", sep="\t", keep_default_na=False)
    words_dict = words_df.set_index("text")["phones"].to_dict()

    # Create prosody-aware model
    model = ProsodyAwareModel(
        num_phones=num_phones,
        num_speakers=num_speakers,
        num_mel_bins=VOCODER.num_mels,
        num_emotions=NUM_EMOTIONS,
        emotion_embedding_dim=64,
    ).to(DEVICE)
    
    # Load pretrained weights if available
    if not args.from_scratch and os.path.exists(args.pretrained):
        model = load_pretrained_model(args.pretrained, model)
        logger.info("Fine-tuning from pretrained model")
    else:
        logger.info("Training from scratch")
    
    logger.info(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")

    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=LR_RATE)

    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=WARMUP * train_epoch_steps,
        t_total=EPOCHS * train_epoch_steps,
    )

    best_loss = float("inf")
    log_file = []
    
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch: {epoch}/{EPOCHS}")
        epoch_start_time = time.time()

        epoch_info = {"epoch": epoch}

        epoch_info |= train_one_epoch(model, train_loader, optimizer, scaler, scheduler, gradient_accumulation_steps)
        epoch_info |= val_one_epoch(model, val_loader)

        if epoch_info["val_total_loss"] < best_loss:
            best_loss = epoch_info["val_total_loss"]
            checkpoint = {
                "state_dict": model.state_dict(),
                "phone_dict": phone_dict,
                "words_dict": words_dict,
                "speaker_dict": speaker_dict,
                "vocoder_name": VOCODER_NAME,
                "num_phones": num_phones,
                "num_speakers": num_speakers,
                "num_mel_bins": VOCODER.num_mels,
                "d_model": model.d_model,
                "num_emotions": NUM_EMOTIONS,
                "emotion_mapping": EMOTION_TO_ID,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "pitch_mean_std": train_dataset.pitch_mean_std,
                "periodicity_range": train_dataset.periodicity_range,
            }
            torch.save(checkpoint, args.output)
            logger.info(f"✓ Best model saved (loss: {best_loss:.6f})")

        log_file.append(
            epoch_info
            | {
                "elapsed": (time.time() - epoch_start_time) / 60,
                "elapsed_total": (time.time() - start_time) / 60,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        logger.info(log_file[-1])

    # Save training history
    pd.DataFrame(log_file).to_csv("model_prosody_history.csv", index=False)

    logger.info(f"Best loss: {best_loss}")
    logger.info(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")
    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
