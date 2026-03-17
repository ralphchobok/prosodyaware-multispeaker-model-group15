#!/usr/bin/env python3
"""
Deep diagnostic: Compare model mel predictions with training data mel-spectrograms.
This will reveal if the model is learning the correct mel distribution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from inference_prosody import load_model, load_phone_vocab, text_to_phones
from preprocess import VOCODER

def compare_model_vs_training(model_path='models/model_prosody_2.pt', device='cuda:1'):
    """Compare model output with training data."""
    
    print("\n" + "="*70)
    print("MODEL OUTPUT vs TRAINING DATA COMPARISON")
    print("="*70 + "\n")
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    phone_to_id = load_phone_vocab()
    
    # 1. Generate mel from model
    print("1. GENERATING MEL FROM MODEL")
    print("-" * 70)
    
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
    
    mel_model = mel_pred.squeeze().cpu().numpy()  # [T, 80]
    
    print(f"  Text: \"{text}\"")
    print(f"  Model output shape: {mel_model.shape}")
    print(f"  Model mel stats:")
    print(f"    Min: {mel_model.min():.4f}")
    print(f"    Max: {mel_model.max():.4f}")
    print(f"    Mean: {mel_model.mean():.4f}")
    print(f"    Std: {mel_model.std():.4f}")
    print()
    
    # 2. Load training data mel
    print("2. LOADING TRAINING DATA MEL")
    print("-" * 70)
    
    import glob
    sample_file = glob.glob('processed/LJ/*.pt')[0]
    data = torch.load(sample_file, weights_only=False)
    mel_train = data['mel'].numpy()  # [T, 80]
    
    print(f"  Sample file: {sample_file}")
    print(f"  Training mel shape: {mel_train.shape}")
    print(f"  Training mel stats:")
    print(f"    Min: {mel_train.min():.4f}")
    print(f"    Max: {mel_train.max():.4f}")
    print(f"    Mean: {mel_train.mean():.4f}")
    print(f"    Std: {mel_train.std():.4f}")
    print()
    
    # 3. Statistical comparison
    print("3. STATISTICAL COMPARISON")
    print("-" * 70)
    
    issues = []
    
    # Check if ranges are similar
    if abs(mel_model.mean() - mel_train.mean()) > 2.0:
        issues.append(f"⚠️  Mean mismatch: model={mel_model.mean():.2f}, train={mel_train.mean():.2f}")
    
    if abs(mel_model.std() - mel_train.std()) > 1.0:
        issues.append(f"⚠️  Std mismatch: model={mel_model.std():.2f}, train={mel_train.std():.2f}")
    
    if mel_model.min() > mel_train.min() + 2:
        issues.append(f"⚠️  Model min ({mel_model.min():.2f}) >> train min ({mel_train.min():.2f})")
    
    if mel_model.max() < mel_train.max() - 2:
        issues.append(f"⚠️  Model max ({mel_model.max():.2f}) << train max ({mel_train.max():.2f})")
    
    # Check for clipping/saturation
    clipped_low = (mel_model <= mel_model.min() + 0.01).sum() / mel_model.size
    clipped_high = (mel_model >= mel_model.max() - 0.01).sum() / mel_model.size
    
    if clipped_low > 0.05:
        issues.append(f"⚠️  {clipped_low*100:.1f}% of values at minimum (clipping?)")
    if clipped_high > 0.05:
        issues.append(f"⚠️  {clipped_high*100:.1f}% of values at maximum (saturation?)")
    
    if issues:
        print("  ❌ ISSUES DETECTED:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  ✓ Statistics look similar")
    print()
    
    # 4. Generate audio from both
    print("4. GENERATING AUDIO FROM BOTH SOURCES")
    print("-" * 70)
    
    VOCODER.to(device)
    
    # Model output
    mel_model_tensor = torch.from_numpy(mel_model).transpose(0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        audio_model = VOCODER(mel_model_tensor).squeeze().cpu().numpy()
    
    # Training data
    mel_train_tensor = torch.from_numpy(mel_train).transpose(0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        audio_train = VOCODER(mel_train_tensor).squeeze().cpu().numpy()
    
    # Save both
    audio_model_norm = np.clip(audio_model, -1, 1)
    audio_train_norm = np.clip(audio_train, -1, 1)
    
    wavfile.write('debug_model_output.wav', VOCODER.sampling_rate, 
                  (audio_model_norm * 32767).astype(np.int16))
    wavfile.write('debug_training_reference.wav', VOCODER.sampling_rate,
                  (audio_train_norm * 32767).astype(np.int16))
    
    print(f"  ✓ Model output audio: debug_model_output.wav ({len(audio_model)/VOCODER.sampling_rate:.2f}s)")
    print(f"  ✓ Training reference audio: debug_training_reference.wav ({len(audio_train)/VOCODER.sampling_rate:.2f}s)")
    print()
    
    # 5. Visualize mel spectrograms
    print("5. VISUALIZING MEL SPECTROGRAMS")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Model output
    im1 = axes[0].imshow(mel_model.T, aspect='auto', origin='lower', 
                         interpolation='nearest', cmap='viridis')
    axes[0].set_title(f'Model Output Mel (mean={mel_model.mean():.2f}, std={mel_model.std():.2f})')
    axes[0].set_ylabel('Mel bin')
    plt.colorbar(im1, ax=axes[0])
    
    # Training data
    im2 = axes[1].imshow(mel_train.T, aspect='auto', origin='lower',
                         interpolation='nearest', cmap='viridis')
    axes[1].set_title(f'Training Data Mel (mean={mel_train.mean():.2f}, std={mel_train.std():.2f})')
    axes[1].set_xlabel('Time frame')
    axes[1].set_ylabel('Mel bin')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('debug_mel_comparison.png', dpi=150)
    print(f"  ✓ Visualization saved: debug_mel_comparison.png")
    print()
    
    # 6. Final diagnosis
    print("="*70)
    print("DIAGNOSIS")
    print("="*70)
    print()
    
    if issues:
        print("❌ MODEL IS NOT LEARNING THE CORRECT MEL DISTRIBUTION!")
        print()
        print("Possible causes:")
        print("  1. Loss function issue - model optimizing wrong objective")
        print("  2. Model architecture problem - insufficient capacity")
        print("  3. Training data mismatch - preprocessing inconsistency")
        print("  4. Normalization issue - model expects different scale")
        print()
        print("Recommendations:")
        print("  • Check if model's mel output layer has activation function")
        print("  • Verify loss computation in train_prosody.py")
        print("  • Check if there's any normalization in the model")
    else:
        print("✓ Model mel statistics are reasonable")
        print()
        print("If audio is still unclear, the issue may be:")
        print("  • Model hasn't learned fine details (needs more training)")
        print("  • Prosody features (pitch/duration) are incorrect")
        print("  • Speaker/emotion embeddings not well-learned")
    
    print()
    print("Next steps:")
    print("  1. Listen to debug_model_output.wav")
    print("  2. Listen to debug_training_reference.wav (should be clear)")
    print("  3. Compare debug_mel_comparison.png visually")
    print("="*70 + "\n")

if __name__ == "__main__":
    compare_model_vs_training()
