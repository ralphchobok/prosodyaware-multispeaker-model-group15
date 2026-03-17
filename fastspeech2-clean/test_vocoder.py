#!/usr/bin/env python3
"""
Test if vocoder works correctly with training data mel-spectrograms.
This helps isolate whether the issue is with the model or vocoder.
"""

import torch
import numpy as np
from scipy.io import wavfile
from preprocess import VOCODER
import glob

def test_vocoder_with_training_data():
    """Generate audio from real training mel-spectrograms."""
    
    print("\n" + "="*70)
    print("VOCODER TEST WITH TRAINING DATA")
    print("="*70 + "\n")
    
    # Load a sample from training data
    sample_files = glob.glob('processed/LJ/*.pt')[:5]
    
    if not sample_files:
        print("❌ No training samples found!")
        return
    
    print(f"Testing vocoder with {len(sample_files)} training samples...")
    print()
    
    device = VOCODER.device if hasattr(VOCODER, 'device') else 'cpu'
    VOCODER.eval()
    
    for i, sample_file in enumerate(sample_files):
        data = torch.load(sample_file, weights_only=False)
        mel = data['mel']  # Shape: [T, 80]
        
        print(f"{i+1}. Processing: {sample_file}")
        print(f"   Mel shape: {mel.shape}")
        print(f"   Mel range: [{mel.min():.2f}, {mel.max():.2f}]")
        
        # Prepare mel for vocoder: [1, 80, T]
        mel_tensor = mel.transpose(0, 1).unsqueeze(0).to(device)
        
        # Generate audio
        with torch.inference_mode():
            audio = VOCODER(mel_tensor).squeeze().cpu().numpy()
        
        # Save audio
        output_path = f"test_training_mel_{i+1}.wav"
        audio_normalized = np.clip(audio, -1, 1)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        wavfile.write(output_path, VOCODER.sampling_rate, audio_int16)
        
        duration = len(audio) / VOCODER.sampling_rate
        print(f"   ✓ Generated {duration:.2f}s of audio")
        print(f"   ✓ Saved to: {output_path}")
        print()
    
    print("="*70)
    print("RESULTS:")
    print("="*70)
    print()
    print("✓ Audio files generated from TRAINING DATA mel-spectrograms")
    print()
    print("Next steps:")
    print("  1. Listen to test_training_mel_*.wav files")
    print("  2. If they sound GOOD → Model is the problem (undertrained)")
    print("  3. If they sound BAD → Vocoder config mismatch")
    print()
    print("Expected: Training data should produce clear speech")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_vocoder_with_training_data()
