#!/usr/bin/env python3
"""
Diagnostic script to check if there's a mel mismatch between training and inference.
"""

import torch
import numpy as np
from inference_prosody import load_model, load_phone_vocab, text_to_phones
from preprocess import VOCODER

def check_mel_mismatch(model_path='models/model_prosody_2.pt', device='cuda:1'):
    """Check for mel normalization and range mismatches."""
    
    print("\n" + "="*70)
    print("MEL MISMATCH DIAGNOSTIC")
    print("="*70 + "\n")
    
    # 1. Check preprocessed mel statistics
    print("1. PREPROCESSED MEL STATISTICS (from training data)")
    print("-" * 70)
    
    import glob
    sample_files = glob.glob('processed/LJ/*.pt')[:10]  # Check 10 samples
    
    if sample_files:
        mel_mins, mel_maxs, mel_means, mel_stds = [], [], [], []
        for f in sample_files:
            data = torch.load(f, weights_only=False)
            mel = data['mel']
            mel_mins.append(mel.min().item())
            mel_maxs.append(mel.max().item())
            mel_means.append(mel.mean().item())
            mel_stds.append(mel.std().item())
        
        print(f"  Analyzed {len(sample_files)} training samples:")
        print(f"    Min range: {min(mel_mins):.4f} to {max(mel_mins):.4f}")
        print(f"    Max range: {min(mel_maxs):.4f} to {max(mel_maxs):.4f}")
        print(f"    Mean: {np.mean(mel_means):.4f} ± {np.std(mel_means):.4f}")
        print(f"    Std: {np.mean(mel_stds):.4f} ± {np.std(mel_stds):.4f}")
        print()
        
        expected_min = min(mel_mins)
        expected_max = max(mel_maxs)
        expected_mean = np.mean(mel_means)
    else:
        print("  ❌ No training samples found!")
        return
    
    # 2. Check model output mel statistics
    print("2. MODEL OUTPUT MEL STATISTICS (from inference)")
    print("-" * 70)
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    phone_to_id = load_phone_vocab()
    
    # Generate sample
    text = "Hello world, this is a test"
    phones, tones = text_to_phones(text, phone_to_id)
    
    speaker_tensor = torch.tensor([0], dtype=torch.long).to(device)
    emotion_tensor = torch.tensor([2], dtype=torch.long).to(device)  # neutral
    phones_tensor = torch.tensor([phones], dtype=torch.long).to(device)
    tones_tensor = torch.tensor([tones], dtype=torch.long).to(device)
    
    with torch.inference_mode():
        mel_pred, dur_pred, pitch_pred, periodicity_pred = model(
            speaker_tensor,
            phones_tensor,
            tones_tensor,
            emotion_tensor,
        )
    
    mel_pred_np = mel_pred.squeeze().cpu().numpy()
    
    print(f"  Generated mel shape: {mel_pred_np.shape}")
    print(f"    Min: {mel_pred_np.min():.4f}")
    print(f"    Max: {mel_pred_np.max():.4f}")
    print(f"    Mean: {mel_pred_np.mean():.4f}")
    print(f"    Std: {mel_pred_np.std():.4f}")
    print()
    
    # 3. Check for mismatches
    print("3. MISMATCH ANALYSIS")
    print("-" * 70)
    
    issues = []
    
    # Check range mismatch
    if mel_pred_np.min() < expected_min - 2.0:
        issues.append(f"⚠️  Model output min ({mel_pred_np.min():.2f}) << training min ({expected_min:.2f})")
    if mel_pred_np.max() > expected_max + 2.0:
        issues.append(f"⚠️  Model output max ({mel_pred_np.max():.2f}) >> training max ({expected_max:.2f})")
    
    # Check if model outputs are normalized (mean ~0, std ~1)
    if abs(mel_pred_np.mean()) < 0.5 and abs(mel_pred_np.std() - 1.0) < 0.5:
        issues.append("⚠️  Model outputs appear NORMALIZED (mean~0, std~1)")
        issues.append("   But training data is NOT normalized!")
        issues.append("   → Need to DENORMALIZE model outputs before vocoder!")
    
    # Check if model outputs are in raw scale instead of log scale
    if mel_pred_np.min() > 0 and mel_pred_np.max() > 10:
        issues.append("⚠️  Model outputs appear in LINEAR scale")
        issues.append("   But vocoder expects LOG scale!")
        issues.append("   → Need to apply log transform!")
    
    if not issues:
        print("  ✓ Mel statistics look reasonable")
        print(f"    Model output range [{mel_pred_np.min():.2f}, {mel_pred_np.max():.2f}]")
        print(f"    Training data range [{expected_min:.2f}, {expected_max:.2f}]")
    else:
        print("  ❌ ISSUES FOUND:")
        for issue in issues:
            print(f"    {issue}")
    
    print()
    
    # 4. Vocoder config check
    print("4. VOCODER CONFIGURATION")
    print("-" * 70)
    print(f"  Model: {VOCODER.__class__.__name__}")
    print(f"  Expected input: Log mel-spectrograms")
    print(f"  Expected range: ~[-11.5, 2.5]")
    print(f"  Mel bins: {VOCODER.num_mels}")
    print(f"  Sample rate: {VOCODER.sampling_rate} Hz")
    print(f"  Hop size: {VOCODER.hop_size}")
    print()
    
    print("="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    if issues:
        print("\n❌ Mel mismatch detected! This causes unclear audio.")
        print("\nTo fix:")
        print("  1. Check if model outputs need denormalization")
        print("  2. Ensure log scale is used (not linear)")
        print("  3. Verify mel config matches between training and vocoder")
    else:
        print("\n✓ Mel statistics appear compatible.")
        print("\nIf audio is still unclear, check:")
        print("  1. Training quality (more epochs, lower loss)")
        print("  2. Phoneme mapping accuracy")
        print("  3. Speaker/emotion embedding quality")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    check_mel_mismatch()
