# Audio Normalization Fix Guide

## Problem Summary

The EDS dataset preprocessing is **missing critical audio normalization** that exists in the LJ preprocessing, causing a ~2.3 unit shift in mel-spectrogram distributions.

### Mel Statistics Comparison

| Dataset | Mel Max | Mel Mean | Mel Std |
|---------|---------|----------|---------|
| **EDS (Current)** | 0.61 | -7.61 | 3.11 |
| **LJ (Correct)** | 1.81 | -5.28 | 2.57 |
| **Difference** | **+1.20** | **+2.33** | - |

The +2.33 mean difference matches our +2.26 inference correction, confirming the root cause.

---

## Root Cause

### Missing Code in `preprocess_eds.py`

**Current code (WRONG):**
```python
# Load audio
y, sr = librosa.load(wav_file, sr=VOCODER.sampling_rate)

# Extract mel spectrogram DIRECTLY without normalization
mel = compute_mel(y).T  # ← BUG: Uses raw, unnormalized audio!
```

**Should be (CORRECT):**
```python
# Load and normalize audio (like LJ preprocessing)
y = read_audio(str(wav_file), subtract_dc=True)

# Extract mel spectrogram from NORMALIZED audio
mel = compute_mel(y).T  # ✓ Correct: Uses normalized audio
```

### What `read_audio()` Does

From `preprocess.py`:
```python
def read_audio(file_path: str, subtract_dc: bool = False) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=VOCODER.sampling_rate)
    
    # 1. Remove DC offset
    if subtract_dc:
        y = y - np.mean(y)
    
    # 2. RMS normalization to -20 dBFS
    rms = np.sqrt(np.mean(y**2))
    desired_rms = 10 ** (-20 / 20)
    gain = desired_rms / rms
    gain = np.clip(gain, 10 ** (-3 / 20), 10 ** (3 / 20))  # Limit to ±3dB
    y = y * gain
    
    # 3. Peak normalization to [-1, 1]  ← CRITICAL!
    y = y / np.max(np.abs(y))
    
    # 4. Simulate 16-bit depth
    y = (y * 32767).astype(np.int16).astype(np.float32) / 32767
    
    return y
```

---

## Solutions

### ✅ Option 1: Re-preprocess EDS Dataset (RECOMMENDED)

**Best for:** Production use, clean solution

**Steps:**
1. Use the fixed preprocessing script:
   ```bash
   python preprocess_eds_fixed.py \
       --input "../EmotionalSpeechDataset/1/Emotion Speech Dataset/" \
       --output processed_fixed/ \
       --skip-mfa  # If you already have TextGrids
   ```

2. Verify mel statistics match LJ distribution:
   ```bash
   # Should see: max~1.8, mean~-5.3
   cat processed_fixed/mel_statistics.csv
   ```

3. Retrain model with fixed data:
   ```bash
   # Update train_prosody.py to use processed_fixed/ instead of processed/
   # Change: OUTPUT_PATH = "processed_fixed/"
   
   python train_prosody.py \
       --pretrained model.pt \
       --output models/model_prosody_fixed.pt \
       --batch-size 16 \
       --gradient-accumulation-steps 2
   ```

**Pros:**
- Clean, proper fix
- No inference corrections needed
- Model learns correct distribution from start

**Cons:**
- Need to re-preprocess entire dataset (~1-2 hours)
- Need to retrain model (200 epochs ~8-10 hours)

---

### ⚡ Option 2: Quick Fix - Normalize in Dataset Loader

**Best for:** Quick testing, temporary solution

**Modify `train_prosody.py`:**

```python
# In EmotionalCustomDataset.__getitem__() method (around line 203)

def __getitem__(self, idx: int) -> dict:
    data = torch.load(self.files[idx], weights_only=True)
    
    # ==========================================
    # QUICK FIX: Normalize mel to match LJ distribution
    # ==========================================
    mel = data["mel"]
    
    # Apply shift to match LJ distribution
    # EDS mean: -7.61, LJ mean: -5.28, diff: +2.33
    MEL_SHIFT = 2.33
    mel = mel + MEL_SHIFT
    
    # ==========================================
    
    # Rest of the code remains the same
    return {
        "speaker": torch.tensor(data["speaker"], dtype=torch.long),
        "phones": data["encoded_text"],
        "tones": data["encoded_tone"], 
        "pitch": data["pitch"],
        "periodicity": data["pitch_periodicity"],
        "duration": data["duration"],
        "mel": mel,  # ← Uses corrected mel
        "emotion": torch.tensor(data["emotion"], dtype=torch.long),
    }
```

**Then retrain:**
```bash
python train_prosody.py \
    --pretrained model.pt \
    --output models/model_prosody_normalized.pt \
    --batch-size 16 \
    --gradient-accumulation-steps 2
```

**Pros:**
- No re-preprocessing needed
- Quick to implement (5 minutes)
- Can test immediately

**Cons:**
- Band-aid fix, not addressing root cause
- Still need correction in inference
- Less clean solution

---

### 🔬 Option 3: Verify First (If Unsure)

**Compare a few files manually:**

```python
import torch
import numpy as np
from preprocess import read_audio, compute_mel

# Load a raw EDS audio file
eds_audio_path = "EmotionalSpeechDataset/1/Emotion Speech Dataset/0011/Angry/0011_000869.wav"

# Method 1: Current (wrong) - direct load
import librosa
y_wrong, _ = librosa.load(eds_audio_path, sr=22050)
mel_wrong = compute_mel(y_wrong).T

# Method 2: Fixed - with normalization
y_correct = read_audio(eds_audio_path, subtract_dc=True)
mel_correct = compute_mel(y_correct).T

print(f"Wrong method: max={mel_wrong.max():.2f}, mean={mel_wrong.mean():.2f}")
print(f"Correct method: max={mel_correct.max():.2f}, mean={mel_correct.mean():.2f}")
print(f"Difference: {mel_correct.mean() - mel_wrong.mean():.2f}")
# Should see ~2.3 difference
```

---

## What About Learning Rate?

### Current Status
- Learning rate: `5e-4` (default in train_prosody.py)
- The learning rate is **NOT** the primary issue
- The mel distribution mismatch is the root cause

### Should you lower LR?
**Maybe, but only AFTER fixing normalization:**

1. **First:** Fix preprocessing (Option 1) OR apply dataset normalization (Option 2)
2. **Then:** If model still struggles, try lower LR for fine-tuning:
   ```bash
   # In train_prosody.py, change:
   LR_RATE = 1e-4  # Instead of 5e-4
   ```

**Why this order?**
- Current model learned **wrong** mel distribution (EDS-style)
- Lower LR won't fix the distribution mismatch
- Lower LR is useful for fine-tuning from LJ→EDS, but only if EDS data is correctly normalized

---

## Recommended Action Plan

### For Best Results:

1. **Verify the problem** (5 minutes):
   ```bash
   python -c "
   import torch
   eds = torch.load('processed/0011/Angry/0011_000869.pt', weights_only=True)
   lj = torch.load('processed/LJ/005.pt', weights_only=True)
   print(f'EDS mel: max={eds[\"mel\"].max():.2f}, mean={eds[\"mel\"].mean():.2f}')
   print(f'LJ mel: max={lj[\"mel\"].max():.2f}, mean={lj[\"mel\"].mean():.2f}')
   "
   ```
   
2. **Choose your fix**:
   - Have time? → **Option 1** (re-preprocess)
   - Need quick test? → **Option 2** (dataset loader fix)

3. **Retrain model** with fixed data:
   ```bash
   # Keep current LR for first attempt
   python train_prosody.py \
       --pretrained model.pt \
       --output models/model_prosody_fixed.pt \
       --batch-size 16 \
       --gradient-accumulation-steps 2
   ```

4. **Test inference** (should work WITHOUT the +2.26 correction):
   ```bash
   # Remove the mel correction from inference_prosody.py first!
   python inference_prosody.py \
       --model models/model_prosody_fixed.pt \
       --text "Hello world" \
       --speaker 0 \
       --emotion neutral \
       --output test_fixed.wav
   ```

5. **If still unclear**, THEN try lower LR:
   ```bash
   # Edit train_prosody.py: LR_RATE = 1e-4
   python train_prosody.py --pretrained model.pt ...
   ```

---

## Other Potential Issues to Check

### 1. Sampling Rate
- ✅ **Already checked**: Both use 22050 Hz (correct)

### 2. Mel Parameters
- ✅ **Already checked**: Both use same `compute_mel()` function

### 3. Training Configuration
- Batch size: 16 with grad accumulation 2 = effective 32 ✅
- Epochs: 200 ✅
- Mixed precision: FP16 ✅
- Current settings are fine

### 4. Data Quality
- EDS might have recording quality issues
- Check a few audio files manually for noise/clipping
- If needed, add preprocessing: `librosa.effects.preemphasis()`

---

## Expected Results After Fix

### Mel Statistics (should match LJ):
```
Max overall: ~1.5-1.8
Average mean: ~-5.0 to -5.5  
Average std: ~2.5-3.0
```

### Audio Quality:
- Clear, natural speech
- No garbled/robot sounds
- Proper emotion expression
- NO inference correction needed

### Training Metrics:
- val_mel_loss: Should improve to ~0.5-0.7 (from 0.81)
- Model should converge faster
- Less overfitting

---

## Summary

**Root Cause:** Missing `read_audio()` normalization in EDS preprocessing

**Primary Fix:** Re-preprocess with `preprocess_eds_fixed.py`

**Quick Fix:** Add +2.33 shift in dataset loader

**Learning Rate:** NOT the main issue, but consider lowering to 1e-4 for fine-tuning AFTER fixing normalization

**Timeline:**
- Quick fix + retrain: ~8-10 hours
- Full re-preprocess + retrain: ~10-12 hours
- But results will be much better!
