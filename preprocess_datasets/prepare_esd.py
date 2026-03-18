"""
Prepare Emotional Speech Dataset (ESD) for FastSpeech2/LightSpeech Training

This script:
1. Reads transcription files from ESD dataset
2. Creates paired WAV and TXT files for MFA
3. Runs Montreal Forced Aligner (MFA) to generate TextGrid alignments
4. Organizes TextGrid files by speaker and emotion

Usage:
    python prepare_esd.py
    python prepare_esd.py --skip-mfa  # Skip MFA alignment step
    python prepare_esd.py --input ../datasets/Emotion Speech Dataset --output ../preprocessed_datasets/esd
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, Tuple
from tqdm import tqdm

# MFA paths (Montreal Forced Aligner)
MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_DICTIONARY = "english_mfa"

# Emotion categories
EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


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
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                file_id, text, emotion = parts[0], parts[1], parts[2]
                transcriptions[file_id] = (text, emotion)
    
    return transcriptions


def run_mfa_alignment(wav_dir: Path, transcription_file: Path, output_dir: Path, emotion: str):
    """
    Run Montreal Forced Aligner on audio files for a specific emotion.
    
    Args:
        wav_dir: Directory containing WAV files
        transcription_file: Speaker's transcription file
        output_dir: Output directory for TextGrid files
        emotion: Emotion category being processed
    """
    print(f"  Running MFA alignment for {emotion}...")
    
    # Create temporary directory for MFA input
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Parse transcriptions
        transcriptions = parse_transcription_file(transcription_file)
        
        # Create paired WAV and TXT files for MFA
        files_prepared = 0
        for wav_file in wav_dir.glob("*.wav"):
            file_id = wav_file.stem
            if file_id in transcriptions:
                text, file_emotion = transcriptions[file_id]
                
                # Copy audio file
                dest_wav = temp_path / wav_file.name
                shutil.copy(wav_file, dest_wav)
                
                # Create text file
                dest_txt = temp_path / f"{file_id}.txt"
                dest_txt.write_text(text)
                files_prepared += 1
        
        if files_prepared == 0:
            print(f"    ⚠ No files found for {emotion}, skipping MFA")
            return
        
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
            subprocess.run(cmd, check=True, capture_output=True)
            textgrid_count = len(list(output_dir.glob("*.TextGrid")))
            print(f"    ✓ {textgrid_count} TextGrid files created")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ MFA alignment failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Prepare ESD Dataset with MFA Alignment")
    parser.add_argument("--input", type=str, default="../datasets/Emotion Speech Dataset",
                       help="Path to Emotion Speech Dataset directory")
    parser.add_argument("--output", type=str, default="../preprocessed_datasets/esd",
                       help="Output directory for TextGrid files")
    parser.add_argument("--skip-mfa", action="store_true",
                       help="Skip MFA alignment (use if TextGrids already exist)")
    args = parser.parse_args()
    
    dataset_path = Path(args.input)
    output_base = Path(args.output)
    
    print("="*60)
    print("ESD DATASET MFA ALIGNMENT")
    print("="*60)
    
    # Validate input
    if not dataset_path.exists():
        print(f"\n✗ Error: ESD directory not found: {dataset_path}")
        return
    
    # Get all speaker directories
    speaker_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if len(speaker_dirs) == 0:
        print(f"\n✗ Error: No speaker directories found in {dataset_path}")
        return
    
    print(f"\n📖 Found {len(speaker_dirs)} speakers: {[d.name for d in speaker_dirs]}")
    print(f"Emotions: {EMOTIONS}\n")
    
    # Process each speaker
    total_textgrids = 0
    
    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        speaker_id = speaker_dir.name
        transcription_file = speaker_dir / f"{speaker_id}.txt"
        
        if not transcription_file.exists():
            print(f"⚠ Warning: No transcription file for {speaker_id}, skipping")
            continue
        
        # Process each emotion
        for emotion in EMOTIONS:
            emotion_dir = speaker_dir / emotion
            
            if not emotion_dir.exists():
                continue
            
            # Output directory for TextGrids
            textgrid_output_dir = output_base / "textgrids" / speaker_id / emotion
            
            if not args.skip_mfa:
                run_mfa_alignment(emotion_dir, transcription_file, textgrid_output_dir, emotion)
                total_textgrids += len(list(textgrid_output_dir.glob("*.TextGrid")))
            else:
                if textgrid_output_dir.exists():
                    total_textgrids += len(list(textgrid_output_dir.glob("*.TextGrid")))
    
    # Summary
    print("\n" + "="*60)
    print("MFA ALIGNMENT COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_base}/textgrids/")
    print(f"Total speakers: {len(speaker_dirs)}")
    
    if not args.skip_mfa:
        print(f"Total TextGrid files created: {total_textgrids}")
    else:
        print(f"Existing TextGrid files found: {total_textgrids}")
    
    print("\n📌 Next steps:")
    print("   1. Verify TextGrid files in output directory")
    print("   2. Run feature extraction: python process_datasets.py")
    print("="*60)


if __name__ == "__main__":
    main()
