"""
Preprocessing Script for Emotional Speech Dataset (EDS)

This script prepares the Emotional Speech Dataset for training
the prosody-aware LightSpeech model.

Dataset Structure Expected:
    EmotionalSpeechDataset/1/Emotion Speech Dataset/
    ├── 0011/
    │   ├── 0011.txt (transcriptions)
    │   ├── Angry/*.wav
    │   ├── Happy/*.wav
    │   ├── Neutral/*.wav
    │   ├── Sad/*.wav
    │   └── Surprise/*.wav
    ├── 0012/
    │   └── ...
    └── ...

Output Structure:
    processed/
    ├── 0011/
    │   ├── Angry/*.pt
    │   ├── Happy/*.pt
    │   ├── Neutral/*.pt
    │   ├── Sad/*.pt
    │   └── Surprise/*.pt
    └── ...

Usage:
    python preprocess_eds.py --input "../EmotionalSpeechDataset/1/Emotion Speech Dataset/"
"""

from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import librosa
import numpy as np
import pandas as pd
import penn
import torch
from tqdm import tqdm
import subprocess
import tempfile
import os

# Import from existing preprocess.py
from preprocess import (
    VOCODER,
    PITCH_FMIN,
    PITCH_FMAX,
    DEVICE,
    BATCH_SIZE,
    SILENCE_TOKEN,
    MAX_SILENCE_LENGTH,
    TRIM_SILENCE,
    SILENT_TO_ZERO,
    approximate_integer_sum,
    compute_mel,
    read_audio  # CRITICAL FIX: Import audio normalization function
)

OUTPUT_PATH = "processed/"

# Emotion categories
EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# MFA paths (Montreal Forced Aligner)
MFA_ACOUSTIC_MODEL = "english_mfa"  # Download with: mfa model download acoustic english_mfa
MFA_DICTIONARY = "english_mfa"      # Download with: mfa model download dictionary english_mfa


def parse_transcription_file(txt_file: Path) -> Dict[str, Tuple[str, str]]:
    """
    Parse speaker transcription file.
    
    Format: FILE_ID\tTRANSCRIPTION\tEMOTION
    
    Returns:
        Dict[file_id] = (transcription, emotion)
    """
    transcriptions = {}
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                file_id = parts[0].strip()
                transcription = parts[1].strip()
                emotion = parts[2].strip()
                transcriptions[file_id] = (transcription, emotion)
    
    return transcriptions


def run_mfa_alignment(wav_dir: Path, transcription_file: Path, output_dir: Path):
    """
    Run Montreal Forced Aligner on audio files.
    
    This creates TextGrid files with word and phone alignments.
    """
    print(f"Running MFA alignment for {wav_dir}")
    
    # Create temporary directory for MFA input
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create input structure for MFA
        # MFA expects: audio_file.wav and audio_file.txt pairs
        transcriptions = parse_transcription_file(transcription_file)
        
        for wav_file in wav_dir.glob("*.wav"):
            file_id = wav_file.stem
            if file_id in transcriptions:
                transcription, _ = transcriptions[file_id]
                
                # Copy wav file
                import shutil
                shutil.copy(wav_file, temp_path / wav_file.name)
                
                # Create text file
                with open(temp_path / f"{file_id}.txt", 'w') as f:
                    f.write(transcription)
        
        # Run MFA
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "mfa", "align",
            str(temp_path),
            MFA_DICTIONARY,
            MFA_ACOUSTIC_MODEL,
            str(output_dir),
            "--clean"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ MFA alignment completed for {wav_dir}")
        except subprocess.CalledProcessError as e:
            print(f"✗ MFA alignment failed: {e}")
            print("Make sure MFA is installed and models are downloaded:")
            print("  conda install -c conda-forge montreal-forced-aligner")
            print(f"  mfa model download acoustic {MFA_ACOUSTIC_MODEL}")
            print(f"  mfa model download dictionary {MFA_DICTIONARY}")
            raise


def words_to_phones_stress(words_tier: List[Dict], phones_tier: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """
    Map English words to their phones and extract stress patterns.
    
    Returns:
        phones_with_stress: List of phone dicts with stress info
        stress_ids: List of stress IDs (1=unstressed, 2=primary, 3=secondary)
    """
    phones_with_stress = []
    stress_ids = []
    
    word_idx = 0
    
    for phone in phones_tier:
        phone_text = phone['text']
        
        # Check if this phone belongs to current word
        while word_idx < len(words_tier) and phone['start'] >= words_tier[word_idx]['end']:
            word_idx += 1
        
        # Extract stress from phone (MFA marks with 0, 1, 2)
        if phone_text:
            # Check for stress markers
            if phone_text.endswith('0'):
                stress = 1  # Unstressed
                phone_text = phone_text[:-1]
            elif phone_text.endswith('1'):
                stress = 2  # Primary stress
                phone_text = phone_text[:-1]
            elif phone_text.endswith('2'):
                stress = 3  # Secondary stress
                phone_text = phone_text[:-1]
            else:
                stress = 1  # Default unstressed
            
            phone['text'] = phone_text
            stress_ids.append(stress)
        else:
            stress_ids.append(1)  # Silence is unstressed
        
        phones_with_stress.append(phone)
    
    return phones_with_stress, stress_ids


def process_audio_file(
    wav_file: Path,
    textgrid_file: Path,
    speaker_id: int,
    emotion_id: int,
    vocab_phones: set,
    vocab_words: set
) -> Dict:
    """Process a single audio file and extract features."""
    
    # ==========================================
    # CRITICAL FIX: Use read_audio() for proper normalization
    # This applies RMS normalization (-20 dBFS) and peak normalization
    # to match LJ preprocessing and ensure correct mel distribution
    # ==========================================
    y = read_audio(str(wav_file), subtract_dc=True)
    
    # Extract mel spectrogram (audio is already normalized)
    mel = compute_mel(y).T
    
    # Extract pitch with PENN
    audio_tensor = torch.from_numpy(y).unsqueeze(0).to(DEVICE)
    pitch, periodicity = penn.from_audio(
        audio=audio_tensor,
        sample_rate=VOCODER.sampling_rate,
        fmin=PITCH_FMIN,
        fmax=PITCH_FMAX,
        gpu=None if DEVICE == "cpu" else DEVICE.replace("cuda:", ""),
        batch_size=BATCH_SIZE,
    )
    pitch = pitch.cpu().squeeze().numpy()
    periodicity = periodicity.cpu().squeeze().numpy()
    
    # Clear GPU cache to prevent memory accumulation
    if DEVICE != "cpu":
        torch.cuda.empty_cache()
    
    # Parse TextGrid
    import tgt
    tg = tgt.read_textgrid(str(textgrid_file))
    
    words_tier = [{'text': interval.text, 'start': interval.start_time, 'end': interval.end_time}
                  for interval in tg.get_tier_by_name('words')]
    phones_tier = [{'text': interval.text, 'start': interval.start_time, 'end': interval.end_time}
                   for interval in tg.get_tier_by_name('phones')]
    
    phones_with_stress, stress_ids = words_to_phones_stress(words_tier, phones_tier)
    
    # Convert phone timings to frame indices
    durations = []
    phone_texts = []
    
    for phone, stress in zip(phones_with_stress, stress_ids):
        phone_text = phone['text'] if phone['text'] else SILENCE_TOKEN
        duration_frames = (phone['end'] - phone['start']) * VOCODER.sampling_rate / VOCODER.hop_size
        
        phone_texts.append(phone_text)
        durations.append(duration_frames)
        vocab_phones.add(phone_text)
    
    # Adjust durations to match mel length
    durations = approximate_integer_sum(
        np.array(durations),
        mel.shape[0],
        np.arange(len(durations))
    )
    
    # Get original text
    original_text = ' '.join([w['text'] for w in words_tier if w['text']])
    
    return {
        'speaker': speaker_id,
        'emotion': emotion_id,
        'encoded_text': phone_texts,  # Will be encoded later with vocab
        'encoded_tone': stress_ids,
        'pitch': torch.from_numpy(pitch[:mel.shape[0]]),
        'pitch_periodicity': torch.from_numpy(periodicity[:mel.shape[0]]),
        'duration': torch.from_numpy(durations),
        'mel': torch.from_numpy(mel),
        'original_text': original_text
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Emotional Speech Dataset")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to Emotional Speech Dataset")
    parser.add_argument("--output", type=str, default="processed/",
                       help="Output directory")
    parser.add_argument("--skip-mfa", action="store_true",
                       help="Skip MFA alignment (use existing TextGrids)")
    args = parser.parse_args()
    
    dataset_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EMOTIONAL SPEECH DATASET PREPROCESSING")
    print("="*60)
    
    # Emotion mapping
    emotion_to_id = {emotion: i for i, emotion in enumerate(EMOTIONS)}
    
    # Speaker mapping
    speaker_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    speaker_to_id = {speaker_dir.name: i for i, speaker_dir in enumerate(speaker_dirs)}
    
    print(f"\nFound {len(speaker_dirs)} speakers: {list(speaker_to_id.keys())}")
    print(f"Emotions: {EMOTIONS}")
    
    # Vocabularies
    vocab_phones = set()
    vocab_words = set()
    
    # Statistics
    speaker_stats = []
    
    # Process each speaker
    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        speaker_id = speaker_to_id[speaker_dir.name]
        transcription_file = speaker_dir / f"{speaker_dir.name}.txt"
        
        if not transcription_file.exists():
            print(f"Warning: Transcription file not found for {speaker_dir.name}")
            continue
        
        transcriptions = parse_transcription_file(transcription_file)
        
        # Process each emotion
        for emotion in EMOTIONS:
            emotion_dir = speaker_dir / emotion
            
            if not emotion_dir.exists():
                print(f"Warning: Emotion folder {emotion} not found for {speaker_dir.name}")
                continue
            
            emotion_id = emotion_to_id[emotion]
            wav_files = list(emotion_dir.glob("*.wav"))
            
            if not wav_files:
                continue
            
            # Run MFA alignment if needed
            textgrid_output = output_path / "textgrids" / speaker_dir.name / emotion
            if not args.skip_mfa:
                # Check if TextGrids already exist
                if textgrid_output.exists() and list(textgrid_output.glob("*.TextGrid")):
                    print(f"  TextGrids already exist for {speaker_dir.name}/{emotion}, skipping MFA")
                else:
                    try:
                        run_mfa_alignment(emotion_dir, transcription_file, textgrid_output)
                    except Exception as e:
                        print(f"✗ MFA alignment failed for {speaker_dir.name}/{emotion}: {e}")
                        print(f"  Skipping this emotion folder...")
                        continue
            else:
                # If skip_mfa is set, look in the default textgrid location
                if not textgrid_output.exists():
                    textgrid_output = emotion_dir  # Fallback to emotion folder
            
            # Create output directory
            output_emotion_dir = output_path / speaker_dir.name / emotion
            output_emotion_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each audio file
            processed_count = 0
            skipped_count = 0
            error_count = 0
            
            for wav_file in tqdm(wav_files, desc=f"  {emotion}", leave=False):
                output_file = output_emotion_dir / f"{wav_file.stem}.pt"
                
                # Skip if already processed
                if output_file.exists():
                    skipped_count += 1
                    continue
                
                textgrid_file = (textgrid_output / wav_file.stem).with_suffix('.TextGrid')
                
                if not textgrid_file.exists():
                    print(f"Warning: TextGrid not found for {wav_file.name}")
                    error_count += 1
                    continue
                
                try:
                    data = process_audio_file(
                        wav_file,
                        textgrid_file,
                        speaker_id,
                        emotion_id,
                        vocab_phones,
                        vocab_words
                    )
                    
                    # Save processed data
                    torch.save(data, output_file)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {wav_file.name}: {e}")
                    error_count += 1
            
            if skipped_count > 0 or error_count > 0:
                print(f"  {speaker_dir.name}/{emotion}: processed={processed_count}, skipped={skipped_count}, errors={error_count}")
            
            speaker_stats.append({
                'speaker': speaker_dir.name,
                'emotion': emotion,
                'files': processed_count + skipped_count  # Total successfully processed files
            })
    
    # Create phone vocabulary
    phone_vocab = sorted(list(vocab_phones))
    phone_to_id = {phone: i for i, phone in enumerate(phone_vocab)}
    
    phone_df = pd.DataFrame([
        {'text': phone, 'phone_id': i}
        for i, phone in enumerate(phone_vocab)
    ])
    phone_df.to_csv(output_path / "phones.tsv", sep="\t", index=False)
    
    # CRITICAL FIX: Update all .pt files to use integer IDs instead of phone strings
    print("\nEncoding phone strings to IDs in all processed files...")
    pt_files = list(output_path.glob("**/*.pt"))
    pt_files = [f for f in pt_files if 'textgrids' not in str(f)]  # Exclude textgrid folder
    
    for pt_file in tqdm(pt_files, desc="Encoding phones"):
        try:
            data = torch.load(pt_file, weights_only=True)
            
            # Convert phone text strings to integer IDs
            if isinstance(data['encoded_text'], list) and len(data['encoded_text']) > 0:
                if isinstance(data['encoded_text'][0], str):
                    # Convert strings to IDs
                    phone_ids = [phone_to_id.get(phone, 0) for phone in data['encoded_text']]
                    data['encoded_text'] = torch.tensor(phone_ids, dtype=torch.long)
                    
            # Convert tone IDs to tensor if needed
            if isinstance(data['encoded_tone'], list):
                data['encoded_tone'] = torch.tensor(data['encoded_tone'], dtype=torch.long)
            
            # Save updated data
            torch.save(data, pt_file)
        except Exception as e:
            print(f"Warning: Could not update {pt_file}: {e}")
    
    print(f"✓ Encoded {len(pt_files)} files with phone IDs")
    
    # Create speaker metadata
    speaker_df = pd.DataFrame([
        {'speaker_id': speaker_id, 'name': speaker_name}
        for speaker_name, speaker_id in speaker_to_id.items()
    ])
    speaker_df.to_csv(output_path / "speakers.tsv", sep="\t", index=False)
    
    # Create emotion metadata
    emotion_df = pd.DataFrame([
        {'emotion_id': emotion_id, 'name': emotion_name}
        for emotion_name, emotion_id in emotion_to_id.items()
    ])
    emotion_df.to_csv(output_path / "emotions.tsv", sep="\t", index=False)
    
    # Print statistics
    stats_df = pd.DataFrame(speaker_stats)
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"\nTotal phones in vocabulary: {len(phone_vocab)}")
    print(f"Total speakers: {len(speaker_to_id)}")
    print(f"Total emotions: {len(emotion_to_id)}")
    print(f"\nFiles processed by speaker and emotion:")
    print(stats_df.pivot(index='speaker', columns='emotion', values='files'))
    print(f"\nTotal files: {stats_df['files'].sum()}")
    print(f"\nOutput saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
