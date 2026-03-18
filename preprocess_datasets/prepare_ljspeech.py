"""
Prepare LJSpeech Dataset for FastSpeech2/LightSpeech Training

This script:
1. Copies WAV files from LJSpeech-1.1/wavs to preprocessed directory
2. Creates text files with transcriptions
3. Runs Montreal Forced Aligner (MFA) to generate TextGrid alignments
4. Moves TextGrid files to the preprocessed directory

Usage:
    python prepare_ljspeech.py
    python prepare_ljspeech.py --skip-mfa  # Skip MFA alignment step
    python prepare_ljspeech.py --input ../datasets/LJSpeech-1.1 --output ../preprocessed_datasets/LJ
"""

import argparse
import pandas as pd
from pathlib import Path
import shutil
import subprocess
from tqdm import tqdm

# MFA paths (Montreal Forced Aligner)
MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_DICTIONARY = "english_mfa"


def run_mfa_alignment(prepared_dir: Path):
    """
    Run Montreal Forced Aligner on prepared audio and text files.
    
    This creates TextGrid files with word and phone alignments.
    
    Args:
        prepared_dir: Directory containing paired .wav and .txt files
    """
    print("\n" + "="*60)
    print("RUNNING MFA ALIGNMENT")
    print("="*60)
    print(f"\nInput directory: {prepared_dir}")
    print(f"Dictionary: {MFA_DICTIONARY}")
    print(f"Acoustic model: {MFA_ACOUSTIC_MODEL}\n")
    
    # MFA will create TextGrid files in the output directory
    aligned_dir = prepared_dir.parent / f"{prepared_dir.name}_aligned"
    
    cmd = [
        "mfa", "align",
        str(prepared_dir),
        MFA_DICTIONARY,
        MFA_ACOUSTIC_MODEL,
        str(aligned_dir),
        "--clean"
    ]
    
    try:
        print(f"Running: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)
        print(f"\n✓ MFA alignment completed!")
        
        # Move TextGrid files back to prepared directory
        print(f"\nMoving TextGrid files to {prepared_dir}...")
        textgrid_count = 0
        for textgrid_file in aligned_dir.glob("*.TextGrid"):
            dest = prepared_dir / textgrid_file.name
            shutil.move(str(textgrid_file), str(dest))
            textgrid_count += 1
        
        print(f"✓ Moved {textgrid_count} TextGrid files")
        
        # Clean up aligned directory
        if aligned_dir.exists():
            shutil.rmtree(aligned_dir)
            print(f"✓ Cleaned up temporary directory: {aligned_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ MFA alignment failed: {e}")
        print("\nMake sure MFA is installed and models are downloaded:")
        print("  conda install -c conda-forge montreal-forced-aligner")
        print("  mfa model download acoustic english_mfa")
        print("  mfa model download dictionary english_mfa")
        raise
    except FileNotFoundError:
        print(f"\n✗ MFA command not found!")
        print("\nPlease install Montreal Forced Aligner:")
        print("  conda install -c conda-forge montreal-forced-aligner")
        print("  mfa model download acoustic english_mfa")
        print("  mfa model download dictionary english_mfa")
        raise


def main():
    parser = argparse.ArgumentParser(description="Preprocess LJSpeech Dataset with MFA Alignment")
    parser.add_argument("--input", type=str, default="../datasets/LJSpeech-1.1",
                       help="Path to LJSpeech-1.1 directory")
    parser.add_argument("--output", type=str, default="../preprocessed_datasets/LJ",
                       help="Output directory for preprocessed files")
    parser.add_argument("--skip-mfa", action="store_true",
                       help="Skip MFA alignment (use if TextGrids already exist)")
    args = parser.parse_args()
    
    ljspeech_dir = Path(args.input)
    metadata_path = ljspeech_dir / "metadata.csv"
    wavs_dir = ljspeech_dir / "wavs"
    target_dir = Path(args.output)
    
    print("="*60)
    print("LJSPEECH PREPROCESSING WITH MFA ALIGNMENT")
    print("="*60)
    
    # Validate input
    if not ljspeech_dir.exists():
        print(f"\n✗ Error: LJSpeech directory not found: {ljspeech_dir}")
        return
    if not metadata_path.exists():
        print(f"\n✗ Error: metadata.csv not found: {metadata_path}")
        return
    if not wavs_dir.exists():
        print(f"\n✗ Error: wavs directory not found: {wavs_dir}")
        return
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read metadata
    print(f"\n📖 Reading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path, sep='|', header=None, 
                     names=['filename', 'transcript', 'normalized'])
    
    print(f"✓ Found {len(df)} audio files")
    
    # Copy WAV files and create text files
    print(f"\n📁 Preparing files in {target_dir}")
    files_copied = 0
    files_skipped = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        filename = row['filename']
        
        # Copy WAV file
        src_wav = wavs_dir / f"{filename}.wav"
        dst_wav = target_dir / f"{filename}.wav"
        if src_wav.exists():
            if not dst_wav.exists():
                shutil.copy2(src_wav, dst_wav)
                files_copied += 1
            else:
                files_skipped += 1
        
        # Create text file
        txt_file = target_dir / f"{filename}.txt"
        if not txt_file.exists():
            with open(txt_file, 'w', encoding='utf-8') as f:
                # Use normalized transcript, fallback to original if NaN
                text = row['normalized'] if pd.notna(row['normalized']) else row['transcript']
                f.write(str(text))
    
    print(f"\n✓ Files copied: {files_copied}")
    if files_skipped > 0:
        print(f"  Files already existed: {files_skipped}")
    
    # Run MFA alignment
    if not args.skip_mfa:
        run_mfa_alignment(target_dir)
    else:
        print("\n⏭  Skipping MFA alignment (--skip-mfa flag set)")
        print(f"   Expected TextGrid files in: {target_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {target_dir}")
    print(f"Total files processed: {len(df)}")
    
    if not args.skip_mfa:
        textgrid_count = len(list(target_dir.glob("*.TextGrid")))
        print(f"TextGrid files created: {textgrid_count}")
    
    print("\n📌 Next steps:")
    print("   1. Verify TextGrid files exist in output directory")
    print("   2. Run feature extraction: python process_datasets.py")
    print("="*60)


if __name__ == "__main__":
    main()