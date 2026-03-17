## Complete Fix - Action Plan

### What Was Fixed:

✅ **preprocess_eds.py**
- Added `read_audio` import from preprocess.py  
- Replaced manual `librosa.load()` with `read_audio()` function
- This applies proper RMS normalization (-20 dBFS) and peak normalization
- EDS mel distribution will now match LJ: mean ~-5.3, max ~1.8

✅ **inference_prosody.py**
- Removed temporary +2.26 mel correction
- Model trained on properly normalized data won't need this

✅ **train_prosody.py**
- Already excludes LJ dataset (filters for 4-digit speaker IDs only)
- Training uses only EDS: 10 speakers × 5 emotions

---

## Next Steps:

### Step 1: Re-preprocess EDS Dataset (Clean Start)

```bash
cd /home/ralph_c/ralph/lightspeech/modern_fastspeech_lightspeech/fast_speech_english_customized/fastspeech2-clean

# Backup old processed data
mv processed processed_old_unnormalized

# Re-run EDS preprocessing with FIXED script
python preprocess_eds.py \
    --input "../EmotionalSpeechDataset/1/Emotion Speech Dataset/" \
    --output processed/ \
    --skip-mfa  # Use existing TextGrids
```

Expected output:
```
MEL STATISTICS (Should match LJ: max~1.8, mean~-5.3)
Mel max range: [~1.2, ~2.0]
Mel mean avg: ~-5.0 to -5.5
Mel std avg: ~2.5 to 3.0
```

### Step 2: Verify Fix Worked

```bash
# Check mel distribution after re-preprocessing
conda activate lightspeech_modern
python verify_mel_mismatch.py
```

Expected: "✅ DATASET NORMALIZATION LOOKS GOOD!" (difference < 0.5)

### Step 3: Retrain Model from Scratch

```bash
# Train on properly normalized EDS data only (LJ excluded)
python train_prosody.py \
    --pretrained model.pt \
    --output models/model_prosody_fixed.pt \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    2>&1 | tee training_prosody_fixed.log
```

Expected improvements:
- val_mel_loss: **0.5-0.7** (was 0.81)
- Faster convergence
- Less overfitting

### Step 4: Test Inference (NO correction needed!)

```bash
python inference_prosody.py \
    --model models/model_prosody_fixed.pt \
    --text "Hello world, this is a test of emotional speech synthesis" \
    --speaker 0 \
    --emotion happy \
    --output test_fixed_happy.wav \
    --device cuda:1

# Test different emotions
for emotion in neutral happy sad angry surprise; do
    python inference_prosody.py \
        --model models/model_prosody_fixed.pt \
        --text "The weather is beautiful today" \
        --speaker 0 \
        --emotion $emotion \
        --output test_${emotion}.wav \
        --device cuda:1
done
```

Expected: Clear, natural audio with proper emotion expression

---

## Timeline Estimate:

| Task | Time | Notes |
|------|------|-------|
| Re-preprocessing | 1-2 hours | Depends on dataset size |
| Verification | 2 minutes | Quick check |
| Retraining (200 epochs) | 8-10 hours | GPU dependent |
| Testing | 5-10 minutes | Multiple emotions |
| **Total** | **~10-12 hours** | Mostly unattended |

---

## FAQ

**Q: Should I keep the old model_prosody_2.pt?**
A: Yes, keep it as backup. The new model will be model_prosody_fixed.pt

**Q: What if audio is still unclear after retraining?**
A: Check:
1. Mel distribution with `verify_mel_mismatch.py` (should show < 0.5 difference)
2. Training logs - look for val_mel_loss < 0.7
3. Raw EDS audio quality - might need denoising

**Q: Should I change learning rate?**
A: Not initially. Current LR (5e-4) is fine. If model struggles after 100 epochs:
- Lower to 1e-4 for fine-tuning
- Or train longer (300 epochs)

**Q: How do I know if it worked?**
A: You'll know it worked when:
- ✅ `verify_mel_mismatch.py` shows < 0.5 difference
- ✅ val_mel_loss drops to 0.5-0.7
- ✅ Audio sounds clear and natural
- ✅ Emotions are distinguishable
- ✅ NO inference corrections needed

---

## Troubleshooting

### If preprocessing fails:
```bash
# Check if MFA TextGrids exist
ls processed_old_unnormalized/textgrids/0011/Angry/*.TextGrid

# If not, re-run MFA alignment:
python preprocess_eds.py \
    --input "../EmotionalSpeechDataset/1/Emotion Speech Dataset/" \
    --output processed/
    # (remove --skip-mfa flag)
```

### If mel distribution still wrong after preprocessing:
```python
# Manually check a file
import torch
from preprocess import read_audio, compute_mel

# Load raw audio
y = read_audio("../EmotionalSpeechDataset/.../some_file.wav", subtract_dc=True)
mel = compute_mel(y).T

print(f"Mel max: {mel.max():.2f}")  # Should be ~1.5-2.0
print(f"Mel mean: {mel.mean():.2f}")  # Should be ~-5.0 to -5.5
```

---

## Summary

**Root Cause:** Missing audio normalization in EDS preprocessing

**Fix Applied:** 
- ✅ Modified `preprocess_eds.py` to use `read_audio()` 
- ✅ Removed inference mel correction
- ✅ Training already uses EDS only

**Next Action:** Re-preprocess → Verify → Retrain → Test

**Expected Result:** Clear, natural, emotionally expressive speech without any corrections!
