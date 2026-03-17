"""
Quick Verification Script - Check Mel Distribution Mismatch

This script compares the mel distributions between EDS and LJ datasets
to confirm the normalization issue.

Usage:
    python verify_mel_mismatch.py
"""

import torch
import glob
import numpy as np
from pathlib import Path

def main():
    print("="*60)
    print("MEL DISTRIBUTION VERIFICATION")
    print("="*60)
    
    # Load sample files
    eds_new_files = sorted(glob.glob('processed/00*/**/*.pt', recursive=True))[:30]
    eds_old_files = sorted(glob.glob('processed_old_unnormalized/00*/**/*.pt', recursive=True))[:30]
    lj_files = sorted(glob.glob('processed/LJ/*.pt'))[:30]
    
    if not eds_new_files and not eds_old_files:
        print("❌ No EDS files found")
        print("   Expected: processed/00*/**/*.pt or processed_old_unnormalized/00*/**/*.pt")
        return
    
    # Determine comparison mode
    if eds_new_files and eds_old_files:
        print(f"\n✓ Found {len(eds_new_files)} NEW EDS files (with normalization fix)")
        print(f"✓ Found {len(eds_old_files)} OLD EDS files (unnormalized)")
        print("\n📊 Comparing: NEW vs OLD EDS preprocessing")
        mode = "eds_comparison"
    elif lj_files:
        print(f"\n✓ Found {len(eds_new_files or eds_old_files)} EDS files")
        print(f"✓ Found {len(lj_files)} LJ files")
        print("\n📊 Comparing: EDS vs LJ dataset")
        mode = "lj_comparison"
    else:
        print(f"\n✓ Found {len(eds_new_files or eds_old_files)} EDS files")
        print("\n📊 Checking against expected values (no comparison dataset)")
        mode = "standalone"
    
    # Compute statistics
    def get_stats(files, name):
        mel_maxs = []
        mel_means = []
        mel_stds = []
        
        for f in files:
            try:
                data = torch.load(f, weights_only=True)
                mel = data['mel']
                mel_maxs.append(mel.max().item())
                mel_means.append(mel.mean().item())
                mel_stds.append(mel.std().item())
            except Exception as e:
                continue
        
        if not mel_maxs:
            return None
        
        print(f"\n{name} Statistics:")
        print(f"  Mel max (overall): {np.max(mel_maxs):.2f}")
        print(f"  Mel mean (avg): {np.mean(mel_means):.2f}")
        print(f"  Mel std (avg): {np.mean(mel_stds):.2f}")
        
        return {
            'max': np.max(mel_maxs),
            'mean': np.mean(mel_means),
            'std': np.mean(mel_stds)
        }
    
    # Get statistics based on mode
    if mode == "eds_comparison":
        eds_new_stats = get_stats(eds_new_files, "NEW EDS (with normalization)")
        eds_old_stats = get_stats(eds_old_files, "OLD EDS (unnormalized)")
        
        if not eds_new_stats or not eds_old_stats:
            print("\n❌ Failed to load file statistics")
            return
            
        print("\n" + "="*60)
        print("COMPARISON: NEW vs OLD")
        print("="*60)
        
        mean_diff = eds_new_stats['mean'] - eds_old_stats['mean']
        max_diff = eds_new_stats['max'] - eds_old_stats['max']
        
        print(f"\nMean difference: {mean_diff:+.2f}")
        print(f"Max difference: {max_diff:+.2f}")
        
        diagnose_normalization_fix(mean_diff, eds_new_stats, expected_mean=-5.3)
        
    elif mode == "lj_comparison":
        eds_stats = get_stats(eds_new_files or eds_old_files, "EDS Dataset")
        lj_stats = get_stats(lj_files, "LJ Dataset")
        
        if not eds_stats or not lj_stats:
            print("\n❌ Failed to load file statistics")
            return
        
        print("\n" + "="*60)
        print("COMPARISON: EDS vs LJ")
        print("="*60)
        
        mean_diff = lj_stats['mean'] - eds_stats['mean']
        max_diff = lj_stats['max'] - eds_stats['max']
        
        print(f"\nMean difference: {mean_diff:+.2f}")
        print(f"Max difference: {max_diff:+.2f}")
        
        diagnose_normalization_fix(mean_diff, eds_stats, expected_mean=-5.3)
        
    else:  # standalone
        eds_stats = get_stats(eds_new_files or eds_old_files, "EDS Dataset")
        
        if not eds_stats:
            print("\n❌ Failed to load file statistics")
            return
        
        print("\n" + "="*60)
        print("COMPARISON: Against Expected Values")
        print("="*60)
        
        expected_mean = -5.3
        mean_diff = expected_mean - eds_stats['mean']
        
        print(f"\nExpected mean: {expected_mean:.2f}")
        print(f"Actual mean: {eds_stats['mean']:.2f}")
        print(f"Difference: {mean_diff:+.2f}")
        
        diagnose_normalization_fix(mean_diff, eds_stats, expected_mean=expected_mean)
    
    print("\n" + "="*60)
    print("\nEXPECTED VALUES (for properly normalized data):")
    print("  Mel max: ~1.5 to 2.0")
    print("  Mel mean: ~-5.0 to -5.5")
    print("  Mel std: ~2.5 to 3.0")
    print("\n" + "="*60)


def diagnose_normalization_fix(mean_diff, eds_stats, expected_mean=-5.3):
    """Diagnose if normalization fix worked."""
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    # Check if EDS is close to expected values
    is_normalized = abs(eds_stats['mean'] - expected_mean) < 0.5
    
    if is_normalized:
        print("\n✅ NORMALIZATION LOOKS GOOD!")
        print(f"   EDS mel mean ({eds_stats['mean']:.2f}) is close to expected ({expected_mean:.2f})")
        print("   Audio should sound clear after retraining!")
        print("\n📋 NEXT STEPS:")
        print("   1. Retrain model with this properly normalized data")
        print("   2. Test inference - should work WITHOUT mel correction")
    elif abs(mean_diff) > 2.0:
        print("\n❌ NORMALIZATION ISSUE DETECTED!")
        print(f"   EDS mels are {abs(mean_diff):.2f} units off from expected")
        if eds_stats['mean'] < expected_mean:
            print("   → Mels are TOO LOW (missing normalization)")
        else:
            print("   → Mels are TOO HIGH (over-normalization)")
        print("\n📋 RECOMMENDED ACTIONS:")
        print("   1. Check if read_audio() is being used in preprocess_eds.py")
        print("   2. Re-run: python preprocess_eds.py --input ... --skip-mfa")
        print("   3. Verify this script shows improvement")
    else:
        print("\n⚠️  MINOR DIFFERENCE DETECTED")
        print(f"   Difference: {abs(mean_diff):.2f} (acceptable: < 0.5, concerning: > 2.0)")
        print("   This might be acceptable, but optimal would be < 0.5")


if __name__ == "__main__":
    main()
