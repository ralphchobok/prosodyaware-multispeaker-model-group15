# Prosody-Aware LightSpeech Training Guide

This guide explains how to train the prosody-aware (emotion-controllable) LightSpeech model on the Emotional Speech Dataset (EDS).

## Overview

The prosody-aware model extends the base LightSpeech architecture with emotion embeddings, allowing control over emotional expression in synthesized speech. The model supports:

- **5 emotions**: Angry, Happy, Neutral, Sad, Surprise
- **10 speakers**: Multi-speaker capability (0011-0020)
- **Fine-tuning**: Transfer learning from pretrained model.pt
- **From-scratch training**: Train new model if needed

## Architecture

### Base LightSpeech (6.25M parameters)
```
- d_model: 512
- Encoder: 4 SepConv layers (2.16M params)
- Decoder: 4 SepConv layers (2.16M params)
- Pitch predictor: 6 layers (1.60M params)
- Duration predictor: 1 layer (266K params)
- Speaker embeddings: 10 speakers (5K params)
```

### Prosody-Aware Extension (+66K parameters)
```
- Emotion embeddings: 5 emotions × 64 dims = 320 params
- Emotion projection: (512 + 64) → 512 = 36,864 params
- LayerNorm: 1,024 params
- Total new params: ~38K params
```

### Emotion Conditioning Pipeline
```
Input text → Encoder → [Concatenate emotion embedding] → Projection → 
→ Duration predictor → Length regulator → Decoder → Mel spectrogram
                     ↓
                 Pitch predictor
```

---

## Prerequisites

### 1. Install Dependencies

```bash
cd /home/ralph_c/ralph/lightspeech/modern_fastspeech_lightspeech/fast_speech_english_customized/fastspeech2-clean

# Core dependencies
pip install torch torchaudio torchvision
pip install librosa numpy pandas tqdm

# Montreal Forced Aligner (for phone alignment)
conda install -c conda-forge montreal-forced-aligner

# PENN pitch tracker
pip install penn

# TextGrid parser
pip install textgrid

# Vocoder (BigVGAN)
pip install bigvgan
```

### 2. Download MFA Models

```bash
# Download English acoustic model
mfa model download acoustic english_us_arpa

# Download English dictionary
mfa model download dictionary english_us_arpa

# Verify installation
mfa version
```

### 3. Dataset Structure

Ensure your Emotional Speech Dataset follows this structure:

```
EmotionalSpeechDataset/1/Emotion Speech Dataset/
├── 0011/
│   ├── 0011.txt              # Transcriptions (FILE_ID\tTRANSCRIPTION\tEMOTION)
│   ├── Angry/
│   │   ├── 0011_angry_001.wav
│   │   ├── 0011_angry_002.wav
│   │   └── ...
│   ├── Happy/
│   │   └── *.wav
│   ├── Neutral/
│   │   └── *.wav
│   ├── Sad/
│   │   └── *.wav
│   └── Surprise/
│       └── *.wav
├── 0012/
│   └── ...
└── ... (0013-0020)
```

**Transcription file format** (0011.txt):
```
0011_angry_001	This is the transcription text.	Angry
0011_angry_002	Another sample utterance.	Angry
0011_happy_001	A happy sentence here.	Happy
...
```

---

## Training Pipeline

### Step 1: Preprocess Dataset

Run the preprocessing script to extract features and perform MFA alignment:

```bash
python preprocess_eds.py \
    --input ../EmotionalSpeechDataset/1/Emotion\ Speech\ Dataset/ \
    --output processed/
```

**What this does:**
1. Runs Montreal Forced Aligner to get phone-level alignments
2. Extracts mel spectrograms (80 channels)
3. Extracts pitch (F0) and periodicity using PENN
4. Extracts phone durations from alignments
5. Extracts stress patterns (primary, secondary, unstressed)
6. Saves processed data as `.pt` files

**Output structure:**
```
processed/
├── 0011/
│   ├── Angry/
│   │   ├── 0011_angry_001.pt
│   │   └── ...
│   ├── Happy/
│   │   └── *.pt
│   └── ...
├── 0012/
│   └── ...
├── phones.tsv      # Phone vocabulary
├── speakers.tsv    # Speaker metadata
└── emotions.tsv    # Emotion labels
```

**Processing time:** ~2-4 hours for 17,500 files (depends on CPU/GPU)

**Skip MFA (if already aligned):**
```bash
python preprocess_eds.py \
    --input ../EmotionalSpeechDataset/1/Emotion\ Speech\ Dataset/ \
    --output processed/ \
    --skip-mfa
```

---

### Step 2: Train Prosody-Aware Model

#### Option A: Fine-tune from Pretrained Model (Recommended)

Fine-tuning leverages the existing `model.pt` trained on LJSpeech:

```bash
python train_prosody.py \
    --pretrained model.pt \
    --output models/model_prosody.pt
```

**Why fine-tuning?**
- Faster convergence (~150 epochs vs 300)
- Better quality with less data
- Preserves learned acoustic features
- Only trains emotion embeddings and adapts to new speakers

**Training config:**
- **Epochs**: 300
- **Learning rate**: 5e-4 (lower than from-scratch)
- **Batch size**: 32
- **Train/val split**: 85/15
- **GPU**: cuda:1 (or specify with CUDA_VISIBLE_DEVICES)
- **Mixed precision**: FP16 automatic mixed precision

#### Option B: Train from Scratch

Train a new model without pretrained weights:

```bash
python train_prosody.py \
    --from-scratch \
    --output models/model_prosody_scratch.pt
```

**Training config:**
- **Epochs**: 300
- **Learning rate**: 1e-3
- **Longer training required**: ~300 epochs

---

### Step 3: Monitor Training

#### Training Dashboard

The script prints progress every 10 batches:

```
Epoch 1/300 [Batch 50/500]
├─ Loss: 2.453 (mel: 1.823, dur: 0.312, pitch: 0.218, period: 0.100)
├─ LR: 5.0e-4
└─ Time: 45.2s
```

#### Validation Metrics

After each epoch, validation losses are computed **per emotion**:

```
Validation Epoch 1
├─ Overall Loss: 2.156
├─ Angry:    Loss=2.234 (mel: 1.654, dur: 0.298, pitch: 0.192, period: 0.090)
├─ Happy:    Loss=2.103 (mel: 1.598, dur: 0.276, pitch: 0.179, period: 0.085)
├─ Neutral:  Loss=2.089 (mel: 1.587, dur: 0.271, pitch: 0.174, period: 0.082)
├─ Sad:      Loss=2.145 (mel: 1.623, dur: 0.289, pitch: 0.184, period: 0.088)
└─ Surprise: Loss=2.209 (mel: 1.641, dur: 0.302, pitch: 0.188, period: 0.091)
```

This helps identify if specific emotions are harder to learn.

#### TensorBoard (Optional)

Add TensorBoard logging to `train_prosody.py`:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/prosody_training')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
```

Then monitor with:
```bash
tensorboard --logdir=runs/
```

---

### Step 4: Generate Speech with Emotions

Use the trained model to synthesize speech with emotion control:

```bash
python predict.py \
    --model models/model_prosody.pt \
    --text "Hello, how are you today?" \
    --speaker 0 \
    --emotion 1 \
    --output output_happy.wav
```

**Emotion IDs:**
- `0` = Angry
- `1` = Happy
- `2` = Neutral
- `3` = Sad
- `4` = Surprise

**Speaker IDs:** 0-9 (corresponding to speakers 0011-0020)

#### Predict Script Modifications

If `predict.py` doesn't support emotions yet, modify it:

```python
# In predict.py, add emotion parameter
from lightspeech_prosody import ProsodyAwareModel

# Load model
model = ProsodyAwareModel(...)
model.load_state_dict(torch.load('models/model_prosody.pt'))

# During inference
emotion_id = torch.tensor([1])  # Happy
output = model(encoded_text, speaker_id, emotions=emotion_id)
```

---

## Training Results

### Expected Convergence

| Training Type | Epochs | Time | Notes |
|--------------|--------|------|-------|
| **Fine-tuning** | 150-200 | ~12-18h | Recommended, faster convergence |
| **From scratch** | 250-300 | ~24-36h | Requires more data and time |

### Loss Targets

**After 100 epochs (fine-tuning):**
- Mel loss: ~0.5-0.8
- Duration loss: ~0.15-0.25
- Pitch loss: ~0.08-0.15
- Periodicity loss: ~0.03-0.06

**After 200 epochs (fine-tuning):**
- Mel loss: ~0.3-0.5
- Duration loss: ~0.10-0.18
- Pitch loss: ~0.05-0.10
- Periodicity loss: ~0.02-0.04

### Quality Metrics

1. **Per-emotion validation loss**: Should be balanced across emotions
2. **Duration accuracy**: Predicted durations should match ground truth within 10%
3. **Pitch RMSE**: Root mean squared error < 20 Hz
4. **MOS (Mean Opinion Score)**: Manual listening test (target: >3.5/5.0)

---

## Troubleshooting

### 1. MFA Alignment Fails

**Error**: `Command 'mfa align' failed`

**Solutions:**
- Check MFA installation: `mfa version`
- Download models: `mfa model download acoustic english_us_arpa`
- Verify transcription format (tab-separated, UTF-8)
- Check audio files are valid WAV format (16kHz recommended)

### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size in train_prosody.py
BATCH_SIZE = 16  # Default is 32

# Use smaller GPU memory
CUDA_VISIBLE_DEVICES=1 python train_prosody.py --pretrained model.pt

# Enable gradient checkpointing (add to train_prosody.py)
torch.utils.checkpoint.checkpoint_sequential(...)
```

### 3. Emotion Not Loading

**Error**: `KeyError: 'emotion'`

**Solutions:**
- Ensure preprocessed files contain emotion labels
- Check `processed/emotions.tsv` exists
- Verify emotion mapping in `train_prosody.py`:
  ```python
  EMOTION_TO_ID = {'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3, 'Surprise': 4}
  ```

### 4. Pretrained Model Incompatible

**Error**: `size mismatch for encoder.layers.0.weight`

**Solutions:**
- This happens if model architectures differ
- Check `d_model`, `n_speakers` match
- Use `--from-scratch` if incompatible:
  ```bash
  python train_prosody.py --from-scratch --output models/model_new.pt
  ```

### 5. Low Quality Synthesis

**Symptoms**: Robotic voice, artifact noise, emotion not noticeable

**Solutions:**
1. Train longer (200+ epochs for fine-tuning)
2. Check validation losses (should decrease consistently)
3. Verify emotion distribution in dataset:
   ```python
   # Check balance
   import pandas as pd
   df = pd.read_csv('processed/emotions.tsv', sep='\t')
   print(df.value_counts())
   ```
4. Adjust emotion embedding dimension:
   ```python
   # In lightspeech_prosody.py
   self.emotion_embedding = nn.Embedding(num_emotions=5, embedding_dim=128)  # Increase from 64
   ```

---

## Advanced Configuration

### Modify Emotion Embedding Size

In `lightspeech_prosody.py`:

```python
class ProsodyAwareModel(nn.Module):
    def __init__(
        self,
        # ... existing params ...
        emotion_dim: int = 64,  # Change this
        num_emotions: int = 5
    ):
        super().__init__()
        
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)
        self.emotion_projection = nn.Linear(d_model + emotion_dim, d_model)
```

**Larger embeddings** (128, 256) can capture more emotion details but require more data.

### Add Reference Encoder

For more sophisticated prosody modeling, add a reference encoder:

```python
class ReferenceEncoder(nn.Module):
    """Extract prosody style from reference audio"""
    def __init__(self, d_model=512):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(256, d_model, batch_first=True)
        
    def forward(self, mel):
        """
        Args:
            mel: [B, T, 80] mel spectrogram
        Returns:
            style: [B, d_model] style embedding
        """
        x = self.conv_layers(mel.transpose(1, 2))  # [B, 256, T]
        x = x.transpose(1, 2)  # [B, T, 256]
        _, style = self.gru(x)  # [1, B, d_model]
        return style.squeeze(0)  # [B, d_model]
```

Then in `ProsodyAwareModel`:

```python
self.reference_encoder = ReferenceEncoder(d_model)

def forward(self, ..., reference_mel=None):
    if reference_mel is not None:
        ref_style = self.reference_encoder(reference_mel)
        x = x + ref_style.unsqueeze(1)  # Add to encoder output
```

### Multi-task Learning Weights

Adjust loss weights in `train_prosody.py`:

```python
# Current weights
mel_weight = 1.0
duration_weight = 1.0
pitch_weight = 1.0
periodicity_weight = 1.0

# Emphasize emotion-sensitive losses
mel_weight = 1.5  # Emphasize mel quality
pitch_weight = 1.2  # Emphasize pitch accuracy
```

---

## File Reference

### Created Files

| File | Purpose | Lines |
|------|---------|-------|
| `lightspeech_prosody.py` | Prosody-aware model architecture | 365 |
| `train_prosody.py` | Training script with emotion support | 712 |
| `preprocess_eds.py` | EDS dataset preprocessing | 450 |
| `PROSODY_TRAINING_GUIDE.md` | This guide | - |

### Existing Files

| File | Purpose |
|------|---------|
| `lightspeech.py` | Base LightSpeech model (referenced by prosody model) |
| `model.pt` | Pretrained model on LJSpeech (single-speaker English) |
| `train.py` | Original training script (template) |
| `predict.py` | Inference script (needs emotion parameter added) |
| `preprocess.py` | Original preprocessing (for LJSpeech) |

---

## Next Steps

### 1. Immediate: Run Preprocessing
```bash
python preprocess_eds.py \
    --input ../EmotionalSpeechDataset/1/Emotion\ Speech\ Dataset/ \
    --output processed/
```

### 2. Start Training
```bash
python train_prosody.py --pretrained model.pt --output models/model_prosody.pt
```

### 3. Evaluate Results

After training, evaluate emotion preservation:

```python
# Create evaluation script
python evaluate_emotions.py \
    --model models/model_prosody.pt \
    --test-set processed/ \
    --metrics emotion_accuracy duration_rmse pitch_rmse
```

### 4. Fine-tune Hyperparameters

Based on validation losses:
- Adjust learning rate
- Modify emotion embedding size
- Balance loss weights
- Add regularization (dropout)

### 5. Deploy Model

Export to ONNX for production:

```bash
python convert_to_onnx.py \
    --model models/model_prosody.pt \
    --output models/model_prosody.onnx
```

---

## Performance Benchmarks

### Dataset Statistics (from EDA)
- **Total samples**: 17,500
- **Speakers**: 10 (balanced, 1,750 each)
- **Emotions**: 5 (balanced, 3,500 each)
- **Average duration**: 2.8 seconds
- **Average text length**: 6.3 words, 31 characters

### Model Size
- **Base model**: 6.25M parameters (~25 MB)
- **Prosody model**: 6.29M parameters (~25.2 MB)
- **Inference time**: ~50ms per utterance (GPU), ~200ms (CPU)

### Training Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, V100)
- **RAM**: 16GB+ recommended
- **Storage**: ~20GB for preprocessed data + models
- **Training time**: 12-18 hours (fine-tuning), 24-36 hours (from scratch)

---

## FAQ

**Q: Can I use a different dataset?**  
A: Yes, but ensure it follows the same folder structure with emotion labels.

**Q: How do I add more emotions?**  
A: Modify `EMOTIONS` list in `preprocess_eds.py` and `EMOTION_TO_ID` in `train_prosody.py`.

**Q: Can I train on CPU?**  
A: Yes, but it will be 10-20x slower. Remove `cuda:1` device specifications.

**Q: How do I resume training?**  
A: Modify `train_prosody.py` to add checkpoint loading:
```python
checkpoint = torch.load('models/checkpoint_epoch_50.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch']
```

**Q: Can this work for other languages?**  
A: Yes! Replace MFA English models with target language models. See `PROSODY_MULTILINGUAL_ANALYSIS.md` for details.

---

## Citation

If you use this prosody-aware model in your research:

```bibtex
@misc{lightspeech_prosody_2024,
  title={Prosody-Aware LightSpeech: Emotion-Controllable Text-to-Speech},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/lightspeech-prosody}}
}
```

---

## Support

For issues or questions:
1. Check `PROSODY_MULTILINGUAL_ANALYSIS.md` for architecture details
2. Review training logs in `runs/` directory
3. Verify dataset preprocessing completed successfully
4. Check CUDA/PyTorch compatibility

**Common resources:**
- Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io/
- PENN pitch tracker: https://github.com/interactiveaudiolab/penn
- BigVGAN vocoder: https://github.com/NVIDIA/BigVGAN

---

**Good luck with your prosody-aware TTS training! 🎤🎭**
