#!/usr/bin/env python3
"""
Comprehensive audio quality diagnostic for prosody-aware TTS.
Checks phoneme mapping, model quality indicators, and vocoder compatibility.
"""

import torch
import os

def check_model_quality(model_path):
    """Check model checkpoint for quality indicators."""
    print(f"\n{'='*70}")
    print("MODEL QUALITY CHECK")
    print(f"{'='*70}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"Model: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB\n")
    
    # Check training metadata
    keys_to_check = [
        'epoch', 'global_step', 'best_loss', 
        'train_loss', 'val_loss',
        'num_phones', 'num_speakers', 'num_emotions'
    ]
    
    print("Training Metadata:")
    for key in keys_to_check:
        if key in checkpoint:
            value = checkpoint[key]
            if key in ['train_loss', 'val_loss', 'best_loss']:
                print(f"  • {key}: {value:.4f}")
            else:
                print(f"  • {key}: {value}")
    print()
    
    # Quality assessment
    issues = []
    
    if 'val_loss' in checkpoint:
        val_loss = checkpoint['val_loss']
        if val_loss > 2.0:
            issues.append(f"⚠️  High validation loss ({val_loss:.2f}) - model may not be well-trained")
        elif val_loss > 1.5:
            issues.append(f"⚠️  Moderate validation loss ({val_loss:.2f}) - audio quality may be suboptimal")
        elif val_loss < 1.0:
            print(f"✓ Good validation loss ({val_loss:.2f})")
        else:
            print(f"  Validation loss: {val_loss:.2f} (borderline)")
    
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        if epoch < 50:
            issues.append(f"⚠️  Low epoch count ({epoch}) - model may be undertrained")
        elif epoch >= 100:
            print(f"✓ Model trained for {epoch} epochs")
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print()
        return False
    
    return True


def check_phoneme_coverage():
    """Check if all common English phonemes are in the vocabulary."""
    print(f"\n{'='*70}")
    print("PHONEME COVERAGE CHECK")
    print(f"{'='*70}\n")
    
    # Load phone vocabulary
    phone_to_id = {}
    with open('processed/phones.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                phone, phone_id = parts
                phone_to_id[phone] = int(phone_id)
    
    print(f"Total phonemes in vocabulary: {len(phone_to_id)}\n")
    
    # Check for essential phonemes
    essential_phonemes = [
        'ə', 'ɪ', 'i', 'iː', 'ʊ', 'u', 'uː',  # Vowels
        'ɛ', 'æ', 'ɑ', 'ɔ', 'ʌ',
        'ej', 'aj', 'ow', 'aw', 'ɔj',  # Diphthongs
        'p', 't', 'k', 'b', 'd', 'ɡ',  # Stops
        'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h',  # Fricatives
        'm', 'n', 'ŋ', 'l', 'ɹ', 'w', 'j',  # Sonorants
        'tʃ', 'dʒ',  # Affricates
        'spn',  # Silence
    ]
    
    missing = []
    present = []
    
    for phone in essential_phonemes:
        if phone in phone_to_id:
            present.append(phone)
        else:
            missing.append(phone)
    
    print(f"Essential phonemes found: {len(present)}/{len(essential_phonemes)}")
    
    if missing:
        print(f"\n⚠️  Missing essential phonemes: {missing}")
        print("   This may cause poor audio quality for certain words!")
        return False
    else:
        print("✓ All essential phonemes present!")
        return True


def check_vocoder_compatibility():
    """Check if vocoder is compatible with model output."""
    print(f"\n{'='*70}")
    print("VOCODER COMPATIBILITY CHECK")
    print(f"{'='*70}\n")
    
    try:
        from preprocess import VOCODER
        print(f"Vocoder: {VOCODER.__class__.__name__}")
        print(f"  • Sampling rate: {VOCODER.sampling_rate} Hz")
        print(f"  • Mel bins: {VOCODER.num_mels}")
        print(f"  • Hop size: {VOCODER.hop_size}")
        print(f"  • FFT size: {VOCODER.n_fft}")
        print()
        
        # Check if mel bins match
        model_path = 'models/model_prosody.pt'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model_mel_bins = checkpoint.get('num_mel_bins', 80)
            
            if model_mel_bins != VOCODER.num_mels:
                print(f"❌ MEL BIN MISMATCH!")
                print(f"  Model expects: {model_mel_bins} mel bins")
                print(f"  Vocoder uses: {VOCODER.num_mels} mel bins")
                print("  This will cause poor audio quality!")
                return False
            else:
                print(f"✓ Mel bins match: {model_mel_bins}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading vocoder: {e}")
        return False


def main():
    """Run all diagnostics."""
    print("\n" + "="*70)
    print("PROSODY-AWARE TTS AUDIO QUALITY DIAGNOSTIC")
    print("="*70)
    
    results = {
        'phoneme_coverage': check_phoneme_coverage(),
        'vocoder_compat': check_vocoder_compatibility(),
        'model_quality': check_model_quality('models/model_prosody.pt'),
    }
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print()
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status:8s} - {check.replace('_', ' ').title()}")
    
    print()
    
    if all_passed:
        print("✓ All checks passed! Audio quality issues may be due to:")
        print("  1. Insufficient training data")
        print("  2. Model architecture limitations")
        print("  3. Prosody feature extraction quality")
    else:
        print("❌ Issues detected that may cause poor audio quality!")
        print("  Review the detailed output above for specific problems.")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
