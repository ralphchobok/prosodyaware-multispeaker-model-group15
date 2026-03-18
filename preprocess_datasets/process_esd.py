"""
Process Emotional Speech Dataset (ESD) - Extract Features

This script extracts acoustic features from ESD dataset that has been
prepared with MFA alignment (TextGrid files).

Handles the complex ESD structure where:
- TextGrids are in: preprocessed_datasets/esd/textgrids/SPEAKER/EMOTION/
- WAV files are in: datasets/Emotion Speech Dataset/SPEAKER/EMOTION/

Extracts:
- Mel-spectrograms
- Pitch (F0) using PENN
- Pitch periodicity
- Phone durations
- Stress patterns
- Emotion labels

Usage:
    python process_esd.py
    python process_esd.py --textgrids ../preprocessed_datasets/esd/textgrids --wavs ../datasets/Emotion Speech Dataset --output ../preprocessed_datasets/processed
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import librosa
import numpy as np
import pandas as pd
import penn
import tgt
import torch
from tqdm import tqdm

# Vocoder configuration
VOCODER_NAME = "hifigan_universal_v1"
VOCODER = torch.hub.load(
    "lars76/bigvgan-mirror",
    VOCODER_NAME,
    source="github",
    trust_repo=True,
    pretrained=True,
)

# Pitch extraction
PITCH_FMIN = 75
PITCH_FMAX = 500
DEVICE = "cuda:1"
BATCH_SIZE = 512

# Silence handling
SILENCE_TOKEN = "<sil>"
MAX_SILENCE_LENGTH = 40 * VOCODER.hop_size
TRIM_SILENCE = False
SILENT_TO_ZERO = True

# Emotion categories
EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


def approximate_integer_sum(
    rational_numbers: np.ndarray, target_sum: int, valid_indices: np.ndarray
) -> np.ndarray:
    """Adjust phoneme durations to match STFT length."""
    rounded_integers = np.maximum(np.rint(rational_numbers), 1)
    float_differences = rational_numbers - rounded_integers
    current_sum = np.sum(rounded_integers)
    error = int(current_sum - target_sum)

    if error > 0:
        to_round_down = np.argsort(float_differences)[valid_indices][:error]
        rounded_integers[to_round_down] -= 1
    elif error < 0:
        to_round_up = np.argsort(-float_differences)[valid_indices][:abs(error)]
        rounded_integers[to_round_up] += 1

    # Final adjustment
    current_sum = np.sum(rounded_integers)
    final_error = int(current_sum - target_sum)
    if final_error < 0:
        rounded_integers[np.argmin(rounded_integers)] += abs(final_error)
    elif final_error > 0:
        rounded_integers[np.argmax(rounded_integers)] -= final_error

    return rounded_integers


def words_to_phones_stress(
    words_tier: List[Dict], phones_tier: List[Dict]
) -> Tuple[List[Dict], List[int]]:
    """Extract stress patterns from MFA output."""
    mapper = []
    stress = []
    
    for word in words_tier:
        phone_text = ""
        for phone in phones_tier:
            if (phone["start_frames"] >= word["start_frames"] and 
                phone["end_frames"] <= word["end_frames"]):
                phone_str = phone["text"]
                
                # Extract stress level
                if phone_str == SILENCE_TOKEN:
                    stress.append(1)
                elif phone_str and phone_str[-1].isdigit():
                    stress_level = int(phone_str[-1])
                    stress.append({0: 1, 1: 2, 2: 3}.get(stress_level, 1))
                    phone_str = phone_str[:-1]
                else:
                    stress.append(1)
                    
                phone_text += f"{phone_str} "
        mapper.append({"text": word["text"], "phones": phone_text.strip()})
    
    return mapper, stress


def tier_to_dict(textgrid: tgt.TextGrid, tier_name: str) -> Tuple[List[Dict], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Convert TextGrid tier to frame-based dictionary."""
    tier = textgrid.get_tier_by_name(tier_name)
    result = []
    textgrid_offset = 0
    phone_regions = []
    silent_regions = []
    
    for i, interval in enumerate(tier):
        start_frames = textgrid_offset + int(np.ceil(interval.start_time * VOCODER.sampling_rate))
        end_frames = textgrid_offset + int(np.ceil(interval.end_time * VOCODER.sampling_rate))
        wav_start_frames = int(np.ceil(interval.start_time * VOCODER.sampling_rate))
        wav_end_frames = int(np.ceil(interval.end_time * VOCODER.sampling_rate))
        duration_frames = end_frames - start_frames
        
        if not interval.text:
            text = SILENCE_TOKEN
            silent_regions.append((wav_start_frames, wav_end_frames))
            if (i == 0 or i == len(tier) - 1) and TRIM_SILENCE:
                textgrid_offset -= duration_frames
                continue
            if MAX_SILENCE_LENGTH < duration_frames:
                textgrid_offset -= duration_frames - MAX_SILENCE_LENGTH
                duration_frames = MAX_SILENCE_LENGTH
                end_frames = start_frames + duration_frames
                wav_start_frames = wav_end_frames - MAX_SILENCE_LENGTH if i == 0 else wav_start_frames
                wav_end_frames = wav_start_frames + MAX_SILENCE_LENGTH if i != 0 else wav_end_frames
        else:
            text = interval.text.strip()
            
        result.append({
            "text": text,
            "start_frames": start_frames,
            "end_frames": end_frames,
            "duration_stft_frames": duration_frames / VOCODER.hop_size,
        })
        phone_regions.append((wav_start_frames, wav_end_frames))
        
    return result, phone_regions, silent_regions


def compute_mel(y: np.ndarray) -> np.ndarray:
    """Compute mel-spectrogram."""
    mel_basis = librosa.filters.mel(
        sr=VOCODER.sampling_rate, n_fft=VOCODER.n_fft,
        n_mels=VOCODER.num_mels, fmin=VOCODER.fmin, fmax=VOCODER.fmax
    )
    pad_length = int((VOCODER.n_fft - VOCODER.hop_size) / 2)
    y = np.pad(y, (pad_length, pad_length), mode="reflect")
    D = librosa.stft(y, n_fft=VOCODER.n_fft, hop_length=VOCODER.hop_size,
                     win_length=VOCODER.win_size, window="hann", center=False, pad_mode="reflect")
    S = np.sqrt(np.abs(D) ** 2 + 1e-9)
    S = np.dot(mel_basis, S)
    return np.log(np.maximum(S, 1e-5))


def compute_pitch(audio: np.ndarray, resample: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pitch using PENN."""
    audio = torch.from_numpy(audio)[None].to(DEVICE)
    pitch, periodicity = penn.from_audio(
        audio=audio, sample_rate=VOCODER.sampling_rate,
        fmin=PITCH_FMIN, fmax=PITCH_FMAX,
        gpu=None if DEVICE == "cpu" else DEVICE.replace("cuda:", ""),
        batch_size=BATCH_SIZE
    )
    pitch = pitch.cpu().squeeze().numpy()
    periodicity = periodicity.cpu().squeeze().numpy()
    
    if DEVICE != "cpu":
        torch.cuda.empty_cache()
    
    if resample > 0:
        pitch = np.interp(np.linspace(0, pitch.shape[0], resample), np.arange(pitch.shape[0]), pitch)
        periodicity = np.interp(np.linspace(0, periodicity.shape[0], resample), np.arange(periodicity.shape[0]), periodicity)
    
    return pitch, periodicity


def read_audio(file_path: str, subtract_dc: bool = False) -> np.ndarray:
    """Load and normalize audio."""
    y, _ = librosa.load(file_path, sr=VOCODER.sampling_rate)
    if subtract_dc:
        y = y - np.mean(y)
    
    # RMS normalization to -20 dBFS
    rms = np.sqrt(np.mean(y**2))
    desired_rms = 10 ** (-20 / 20)
    gain = np.clip(desired_rms / rms, 10**(-3/20), 10**(3/20))
    y = y * gain
    y = y / np.max(np.abs(y))
    
    # 16-bit depth simulation
    y = (y * 32767).astype(np.int16).astype(np.float32) / 32767
    return y


def process_file(
    textgrid_file: Path, 
    wav_base_path: Path,
    speaker_info: Dict, 
    phone_to_id: Dict, 
    cur_phone_id: int, 
    output_path: Path,
    emotion_to_id: Dict
) -> Tuple[Dict, List, List, int]:
    """Process single audio file."""
    # Parse path: textgrids/SPEAKER/EMOTION/FILE.TextGrid
    speaker_id = textgrid_file.parent.parent.name
    emotion = textgrid_file.parent.name
    file_name = textgrid_file.stem
    
    # Construct WAV path: wavs/SPEAKER/EMOTION/FILE.wav
    wav_filename = wav_base_path / speaker_id / emotion / f"{file_name}.wav"
    
    if not wav_filename.exists():
        raise FileNotFoundError(f"{wav_filename} does not exist")
    
    # Track speaker
    if speaker_id not in speaker_info:
        speaker_info[speaker_id] = {"num_files": 0, "speaker_index": len(speaker_info)}
    
    speaker_output_path = output_path / speaker_id
    speaker_output_path.mkdir(parents=True, exist_ok=True)
    
    num_files = speaker_info[speaker_id]["num_files"]
    output_filepath = speaker_output_path / f"{num_files:03}.pt"
    speaker_info[speaker_id]["num_files"] += 1
    
    # Parse TextGrid
    textgrid = tgt.io.read_textgrid(textgrid_file, include_empty_intervals=True)
    phones, phone_regions, silent_regions = tier_to_dict(textgrid, "phones")
    words, _, _ = tier_to_dict(textgrid, "words")
    
    # Load and process audio
    wav = read_audio(str(wav_filename), subtract_dc=True)
    if SILENT_TO_ZERO:
        for start, stop in silent_regions:
            wav[start:stop] = 0
    
    mask = np.zeros(len(wav), dtype=bool)
    for start, stop in phone_regions:
        mask[start:stop] = True
    wav = wav[mask]
    
    # Extract features
    words_to_phones, stress = words_to_phones_stress(words, phones)
    mel = compute_mel(wav)
    pitch, periodicity = compute_pitch(wav, resample=mel.shape[1])
    
    # Compute durations
    durations = np.array([p["duration_stft_frames"] for p in phones])
    valid_indices = np.array([p["text"] != SILENCE_TOKEN for p in phones])
    rounded_durations = approximate_integer_sum(durations, mel.shape[1], valid_indices)
    
    # Encode phones
    phone_text = []
    for p in phones:
        text = p["text"]
        if text not in phone_to_id:
            phone_to_id[text] = cur_phone_id
            cur_phone_id += 1
        phone_text.append(phone_to_id[text])
    
    # Save processed data
    torch.save({
        "speaker": torch.tensor(speaker_info[speaker_id]["speaker_index"], dtype=torch.long),
        "emotion": torch.tensor(emotion_to_id[emotion], dtype=torch.long),
        "encoded_text": torch.tensor(phone_text, dtype=torch.long),
        "encoded_tone": torch.tensor(stress, dtype=torch.long),
        "pitch": torch.from_numpy(pitch).float(),
        "pitch_periodicity": torch.from_numpy(periodicity).float(),
        "duration": torch.from_numpy(rounded_durations).long(),
        "mel": torch.from_numpy(mel.T),
        "original_text": " ".join([k["text"] for k in words if k["text"]]),
    }, output_filepath)
    
    return speaker_info, phones, words_to_phones, cur_phone_id


def main():
    parser = argparse.ArgumentParser(description="Process ESD Dataset - Extract Features")
    parser.add_argument("--textgrids", type=str, default="../preprocessed_datasets/esd/textgrids",
                       help="Path to ESD TextGrid files")
    parser.add_argument("--wavs", type=str, default="../datasets/Emotion Speech Dataset",
                       help="Path to ESD WAV files")
    parser.add_argument("--output", type=str, default="../preprocessed_datasets/processed",
                       help="Output directory for processed files")
    args = parser.parse_args()
    
    textgrid_path = Path(args.textgrids)
    wav_path = Path(args.wavs)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ESD FEATURE EXTRACTION")
    print("="*60)
    
    emotion_to_id = {emotion: i for i, emotion in enumerate(EMOTIONS)}
    phone_to_id = {}
    cur_phone_id = 1
    phone_stats = []
    words_stats = []
    speaker_info = {}
    
    textgrid_files = sorted(textgrid_path.glob("**/*.TextGrid"))
    print(f"\nFound {len(textgrid_files)} TextGrid files")
    print(f"Emotions: {EMOTIONS}\n")
    
    for filename in tqdm(textgrid_files, desc="Processing files"):
        try:
            speaker_info, file_phone_stats, file_words_stats, cur_phone_id = process_file(
                filename, wav_path, speaker_info, phone_to_id, cur_phone_id, output_path, emotion_to_id
            )
            phone_stats.extend(file_phone_stats)
            words_stats.extend(file_words_stats)
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
    
    # Generate statistics
    phone_stats_df = pd.DataFrame(phone_stats)
    phone_stats_df.insert(0, "phone_id", phone_stats_df["text"].map(phone_to_id))
    phone_stats_df = phone_stats_df.groupby(["phone_id", "text"]).agg(
        occurrences=("text", "count")
    ).sort_values("occurrences", ascending=False).reset_index()
    phone_stats_df.to_csv(output_path / "phones_esd.tsv", index=False, sep="\t")
    
    words_stats_df = pd.DataFrame(words_stats)
    words_stats_df = words_stats_df.groupby(["text", "phones"]).agg(
        occurrences=("text", "count")
    ).sort_values("occurrences", ascending=False).reset_index()
    words_stats_df.to_csv(output_path / "words_esd.tsv", index=False, sep="\t")
    
    speaker_df = pd.DataFrame([{
        "speaker_id": info["speaker_index"],
        "name": speaker_id,
        "num_files": info["num_files"]
    } for speaker_id, info in speaker_info.items()])
    speaker_df.to_csv(output_path / "speakers_esd.tsv", index=False, sep="\t")
    
    emotion_df = pd.DataFrame([{
        "emotion_id": emo_id,
        "name": emo_name
    } for emo_name, emo_id in emotion_to_id.items()])
    emotion_df.to_csv(output_path / "emotions.tsv", index=False, sep="\t")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"Total files processed: {speaker_df['num_files'].sum()}")
    print(f"Speakers: {len(speaker_info)}")
    print(f"Unique phones: {len(phone_to_id)}")
    print("="*60)


if __name__ == "__main__":
    main()
