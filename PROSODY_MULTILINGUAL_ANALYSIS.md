# LightSpeech Prosody-Aware Multilingual Training Analysis

## Table of Contents
1. [EDA Results Summary](#eda-results-summary)
2. [Current LightSpeech Architecture](#current-lightspeech-architecture)
3. [Prosody-Aware Training Strategy](#prosody-aware-training-strategy)
4. [Multilingual Training Strategy](#multilingual-training-strategy)
5. [Implementation Recommendations](#implementation-recommendations)

---

## 1. EDA Results Summary

### Dataset Characteristics (Emotional Speech Dataset)

**Overview:**
- **Total Audio Files:** 17,500
- **Speakers:** 10 (IDs: 0011-0020)
- **Emotions:** 5 categories (Angry, Happy, Neutral, Sad, Surprise)
- **Language:** English

**Distribution:**

| Emotion  | Files | Percentage |
|----------|-------|------------|
| Angry    | 3,500 | 20%        |
| Happy    | 3,500 | 20%        |
| Neutral  | 3,500 | 20%        |
| Sad      | 3,500 | 20%        |
| Surprise | 3,500 | 20%        |

**Per Speaker:** 1,750 files each (perfectly balanced)

**Transcription Statistics:**
- **Total Transcriptions:** 17,500
- **Average Text Length:** 31.0 characters (±8.9)
- **Average Word Count:** 6.3 words (±1.7)
- **Range:** 2-12 words per utterance
- **Text Length by Emotion:** Uniform across all emotions (~31 chars)

**Key Findings:**
✅ **Perfect Balance:** All emotions and speakers are perfectly balanced  
✅ **Consistent Length:** Text lengths are uniform across emotions  
✅ **Multi-Speaker:** 10 speakers provide diversity  
✅ **Rich Prosody:** 5 distinct emotional categories with clear labels  
⚠️ **Moderate Size:** 17,500 samples may need augmentation for robust training

---

## 2. Current LightSpeech Architecture

### Model Components (6.25M parameters)

```
Component              Parameters    Purpose
─────────────────────────────────────────────────────────────
speaker_embedding      5,120         Multi-speaker representation
embed_tokens           24,800        Phone/phoneme embeddings
embed_tones            112           Stress/tone patterns
embed_pitch            1,536         Pitch feature integration
encoder                2,158,592     Text → acoustic features
decoder                2,156,544     Acoustic → mel spectrogram
duration_predictor     265,729       Phoneme duration prediction
pitch_predictor        1,598,466     F0 + periodicity prediction
mel_out                41,040        Final mel projection
─────────────────────────────────────────────────────────────
Total                  6,253,987
```

### Current Architecture Strengths

1. **Prosody Modeling:**
   - ✅ Pitch predictor outputs both F0 and periodicity
   - ✅ Duration predictor for phoneme-level timing
   - ✅ Tone/stress embeddings (7 categories)
   - ✅ Speaker embeddings for multi-speaker

2. **Efficient Design:**
   - Separable convolutions reduce parameters
   - No attention mechanism (faster inference)
   - 512-dim hidden states (d_model)

### Current Architecture Limitations

❌ **No Emotion Control:** No emotion embeddings or conditioning  
❌ **No Prosody Variability:** Deterministic pitch/duration generation  
❌ **Single Language:** Tone embeddings designed for stress (English) or tones (Chinese)  
❌ **No Cross-Lingual:** Separate phone vocabularies per language  
❌ **Limited Expressiveness:** No VAE or reference encoder for style

---

## 3. Prosody-Aware Training Strategy

### 3.1 Add Emotion Conditioning

**Approach 1: Emotion Embeddings (Recommended)**

```python
class ProsodyAwareLightSpeech(nn.Module):
    def __init__(self, ..., num_emotions=5, emotion_embedding_dim=64):
        super().__init__()
        # Existing components...
        
        # NEW: Emotion conditioning
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)
        
        # Modified encoder input: d_model + emotion_embedding_dim
        self.encoder_projection = nn.Linear(
            d_model + emotion_embedding_dim, 
            d_model
        )
    
    def forward(self, ..., emotions=None):
        # Embed emotions
        emotion_emb = self.emotion_embedding(emotions).unsqueeze(1)
        
        # Concatenate with encoder output
        encoder_outputs = ...  # existing encoder
        encoder_outputs = torch.cat([encoder_outputs, 
                                     emotion_emb.expand(-1, encoder_outputs.size(1), -1)], 
                                    dim=-1)
        encoder_outputs = self.encoder_projection(encoder_outputs)
```

**Approach 2: Style Reference Encoder (Advanced)**

```python
class ReferenceEncoder(nn.Module):
    """Extract prosody embedding from reference mel-spectrogram"""
    def __init__(self, mel_channels=80, embedding_dim=128):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        ])
        self.gru = nn.GRU(64 * mel_channels // 16, embedding_dim, batch_first=True)
        
    def forward(self, mels):
        # mels: [B, T, mel_channels]
        x = mels.unsqueeze(1)  # [B, 1, T, mel_channels]
        for conv in self.convs:
            x = F.relu(conv(x))
        # ... pool and GRU to get prosody embedding
```

### 3.2 Modify Training Data Loading

**Dataset Modification:**

```python
class EmotionalDataset(Dataset):
    def __init__(self, files, emotion_mapping):
        self.files = files
        # Map: 'Angry'->0, 'Happy'->1, 'Neutral'->2, 'Sad'->3, 'Surprise'->4
        self.emotion_mapping = emotion_mapping
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        
        # Extract emotion from file path
        # e.g., .../0011/Angry/0011_000351.pt -> 'Angry'
        emotion_str = Path(self.files[idx]).parent.name
        emotion_id = self.emotion_mapping[emotion_str]
        
        data['emotion'] = emotion_id
        return data
```

### 3.3 Training Loss Modifications

**Add Prosody Consistency Loss:**

```python
def train_one_epoch_with_prosody(model, train_loader, ...):
    # Existing losses
    mel_loss = l1_loss(mel_pred, mels)
    dur_loss = mse_loss(dur_pred, durations)
    pitch_loss = mse_loss(pitch_pred, pitches)
    
    # NEW: Prosody distribution loss (optional)
    # Encourage different emotions to have distinct prosody distributions
    emotion_prosody_reg = compute_emotion_separation_loss(
        pitch_pred, emotions
    )
    
    loss_all = mel_loss + dur_loss + pitch_loss + 0.1 * emotion_prosody_reg
```

### 3.4 Preprocessing for Emotional Speech Dataset

**Steps:**
1. **Extract speaker from folder structure** (0011-0020)
2. **Extract emotion from folder structure** (Angry/Happy/Neutral/Sad/Surprise)
3. **Use MFA for alignment** (Montreal Forced Aligner)
4. **Generate TextGrid files** with word and phone alignments
5. **Extract F0 with PENN** (existing pipeline)
6. **Save .pt files** with emotion labels included

**Modified preprocess.py:**

```python
def process_emotional_dataset():
    dataset_path = Path("EmotionalSpeechDataset/1/Emotion Speech Dataset")
    
    for speaker_dir in dataset_path.glob("*"):
        speaker_id = speaker_dir.name  # e.g., "0011"
        
        for emotion_dir in speaker_dir.glob("*/"):
            emotion = emotion_dir.name  # e.g., "Angry"
            
            for wav_file in emotion_dir.glob("*.wav"):
                # Load transcription from speaker text file
                transcription = load_transcription(speaker_dir, wav_file.stem)
                
                # Run MFA alignment (or use existing TextGrid)
                # Extract features...
                
                # Save with emotion metadata
                torch.save({
                    'speaker': speaker_to_id[speaker_id],
                    'emotion': emotion_to_id[emotion],
                    'encoded_text': ...,
                    'pitch': ...,
                    'mel': ...,
                    # ... other features
                }, output_file)
```

---

## 4. Multilingual Training Strategy

### 4.1 Unified Phone Representation

**IPA (International Phonetic Alphabet) Mapping:**

```python
# Map language-specific phones to IPA
IPA_MAPPING = {
    'en': {  # English (ARPAbet -> IPA)
        'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ',
        'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ',
        # ... complete mapping
    },
    'zh': {  # Chinese (pinyin phones -> IPA)
        'a': 'a', 'ai': 'aɪ', 'an': 'an', 'ang': 'aŋ',
        'ao': 'ɑʊ', 'b': 'p', 'c': 'tsʰ', 'ch': 'tʂʰ',
        # ... complete mapping
    }
}

def unify_phones(language_phones, language_code):
    """Convert language-specific phones to unified IPA set"""
    return [IPA_MAPPING[language_code].get(p, p) 
            for p in language_phones]
```

**Implementation:**

```python
class MultilingualLightSpeech(nn.Module):
    def __init__(self, num_ipa_phones=200, num_languages=2):
        super().__init__()
        # Unified IPA phone embeddings (shared across languages)
        self.embed_phones = nn.Embedding(num_ipa_phones, d_model - tone_embedding - lang_embedding)
        
        # Language-specific embeddings
        self.language_embedding = nn.Embedding(num_languages, lang_embedding_dim)
        
        # Language-specific tone/stress handling
        self.tone_adapters = nn.ModuleDict({
            'en': nn.Embedding(3, tone_embedding),  # stress patterns
            'zh': nn.Embedding(5, tone_embedding),  # tones 1-5
        })
```

### 4.2 Language-Aware Architecture

**Option 1: Shared Encoder + Language-Specific Heads**

```python
class MultilingualModel(nn.Module):
    def __init__(self):
        # Shared components
        self.shared_encoder = ...
        
        # Language-specific decoders
        self.decoders = nn.ModuleDict({
            'en': DecoderModule(...),
            'zh': DecoderModule(...),
        })
        
    def forward(self, ..., language_id):
        x = self.shared_encoder(...)
        x = self.decoders[language_id](x)
        return x
```

**Option 2: Fully Shared (Recommended for similar languages)**

```python
# Single architecture with language embeddings
# Language-specific behavior learned through embeddings
```

### 4.3 Joint Training Data Pipeline

**Mixed Batch Strategy:**

```python
class MultilingualDataLoader:
    def __init__(self, datasets_by_language, batch_size):
        self.datasets = datasets_by_language
        self.batch_size = batch_size
        
    def __iter__(self):
        # Sample from each language proportionally
        for batch in self.create_mixed_batch():
            yield batch
    
    def create_mixed_batch(self):
        # Option 1: Proportional sampling (based on dataset size)
        # Option 2: Balanced sampling (equal from each language)
        # Option 3: Curriculum learning (start with one, gradually add others)
        pass
```

### 4.4 Cross-Lingual Transfer

**Pre-training Strategy:**
1. **Phase 1:** Train on larger dataset (e.g., English LJSpeech)
2. **Phase 2:** Fine-tune with mixed English + Chinese data
3. **Phase 3:** Add low-resource languages

**Code Example:**

```python
# Phase 1: English pre-training
model = MultilingualLightSpeech(...)
train(model, english_data, epochs=100)

# Phase 2: Multilingual fine-tuning
# Freeze some layers to preserve learned representations
for param in model.shared_encoder[:2].parameters():
    param.requires_grad = False
    
train(model, mixed_data, epochs=50, lr=1e-4)
```

---

## 5. Implementation Recommendations

### 5.1 Immediate Actions for Prosody-Aware Training

**Step 1: Preprocess Emotional Speech Dataset**

```bash
# Create folder structure
mkdir -p processed/EDS
cd fastspeech2-clean

# Modify preprocess.py to handle EDS
python preprocess.py --dataset EDS \
                     --input ../EmotionalSpeechDataset/1/Emotion\ Speech\ Dataset/ \
                     --output processed/EDS
```

**Step 2: Modify LightSpeech Model**

Create `lightspeech_prosody.py`:

```python
from lightspeech import Model as LightSpeechBase

class ProsodyAwareLightSpeech(LightSpeechBase):
    def __init__(self, *args, num_emotions=5, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add emotion embedding
        self.emotion_embedding = nn.Embedding(num_emotions, 32)
        
        # Modify encoder to accept emotion conditioning
        # Add projection layer after concatenating emotion embeddings
        self.emotion_projection = nn.Linear(self.d_model + 32, self.d_model)
    
    def forward(self, speakers, tokens, tones, pitches, periodicity, 
                durations, mels, emotions=None):
        # Call parent forward for base processing
        # ... modify to inject emotion embeddings after encoder
```

**Step 3: Update Training Script**

Modify `train.py`:

```python
# Import new model
from lightspeech_prosody import ProsodyAwareLightSpeech

# Update dataset to include emotions
class EmotionalCustomDataset(CustomDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # data already has 'emotion' from preprocessing
        return data
    
    def collate_fn(self, batch):
        # Add emotions to batch
        emotions = torch.tensor([b['emotion'] for b in batch])
        # ... existing collation
        return (*existing_outputs, emotions)

# Training loop
model = ProsodyAwareLightSpeech(
    num_phones=num_phones,
    num_speakers=num_speakers,
    num_mel_bins=VOCODER.num_mels,
    num_emotions=5  # NEW
)
```

### 5.2 Multilingual Extension Plan

**Phase 1: Prepare Second Language Data (e.g., Chinese)**

```bash
# Download AISHELL-3 or similar Chinese dataset
# Run MFA with Chinese acoustic model
mfa align chinese_data lexicon_chinese.txt pretrained_chinese.zip aligned_output

# Preprocess with IPA mapping
python preprocess.py --dataset AISHELL3 \
                     --language zh \
                     --use-ipa-mapping
```

**Phase 2: Create Multilingual Model**

```python
class MultilingualProsodyLightSpeech(ProsodyAwareLightSpeech):
    def __init__(self, *args, num_languages=2, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add language embedding
        self.language_embedding = nn.Embedding(num_languages, 16)
        
        # Adjust embedding sizes to accommodate language embeddings
        # phone_emb + tone_emb + lang_emb = d_model
```

**Phase 3: Joint Training**

```python
# Combine datasets
train_files = english_files + chinese_files
train_dataset = MultilingualDataset(train_files)

# Train with mixed batches
model = MultilingualProsodyLightSpeech(
    num_phones=200,  # Unified IPA phoneset
    num_speakers=num_en_speakers + num_zh_speakers,
    num_emotions=5,
    num_languages=2
)
```

### 5.3 Training Configuration

**Recommended Hyperparameters:**

```python
# For Emotional Speech Dataset (17,500 samples)
EPOCHS = 300  # More epochs for smaller dataset
BATCH_SIZE = 32
LR_RATE = 1e-3
WARMUP = 10
TRAINING_SPLIT = 0.15  # More for validation (2,625 samples)

# Data augmentation (recommended)
USE_SPECAUGMENT = True
PITCH_AUGMENT_RANGE = 0.1  # ±10% pitch shift
SPEED_AUGMENT_RANGE = 0.05  # ±5% speed change
```

### 5.4 Evaluation Metrics

**Track Additional Metrics:**

```python
def evaluate_emotion_accuracy(model, val_files):
    """
    Evaluate if synthesized speech preserves emotional characteristics
    Uses a pre-trained emotion classifier
    """
    emotion_classifier = load_emotion_classifier()
    
    matches = 0
    total = 0
    
    for file in val_files:
        true_emotion = get_emotion(file)
        synthesized_audio = generate_audio(model, file)
        pred_emotion = emotion_classifier(synthesized_audio)
        
        if pred_emotion == true_emotion:
            matches += 1
        total += 1
    
    return matches / total

# Add to training loop
emotion_accuracy = evaluate_emotion_accuracy(model, val_files)
logger.info(f"Emotion preservation accuracy: {emotion_accuracy:.3f}")
```

### 5.5 Inference Control

**Emotion-Controllable Synthesis:**

```python
def synthesize_with_emotion(text, speaker_id, emotion='neutral'):
    """Generate speech with specific emotion"""
    
    emotion_id = emotion_to_id[emotion]
    
    # Prepare inputs
    tokens = text_to_tokens(text)
    
    # Generate
    with torch.inference_mode():
        mel_pred, _, _, _ = model(
            speakers=torch.tensor([speaker_id]),
            tokens=tokens,
            emotions=torch.tensor([emotion_id]),
            # ... other inputs
        )
    
    # Vocoder
    audio = vocoder(mel_pred)
    return audio

# Usage
audio = synthesize_with_emotion(
    "Hello, how are you?", 
    speaker_id=0, 
    emotion='happy'
)
```

---

## 6. Expected Outcomes

### With Prosody-Aware Training:
✅ Controllable emotional expression  
✅ Better prosody variability (pitch, rhythm, intensity)  
✅ More natural-sounding speech  
✅ Emotion-specific speaking styles  

### With Multilingual Training:
✅ Cross-lingual voice cloning  
✅ Code-switching capability  
✅ Reduced data requirements for new languages  
✅ Better generalization  

### Combined Benefits:
✅ Emotional multilingual TTS  
✅ Universal prosody representations  
✅ Rich expressive synthesis across languages  

---

## 7. Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Emotion imbalance across speakers** | Use weighted sampling or data augmentation |
| **Smaller dataset (17.5K vs 13.1K LJSpeech)** | Use transfer learning from pre-trained model |
| **Cross-lingual phone mapping complexity** | Use IPA with language-specific adapters |
| **Increased model complexity** | Start with emotion only, add multilingual later |
| **Training instability** | Use gradient clipping, warmup schedule |
| **Overfitting on emotions** | Mixup augmentation, dropout |

---

## 8. Next Steps

### Immediate (Week 1-2):
1. ✅ Complete EDA on Emotional Speech Dataset
2. ⬜ Run MFA alignment on EDS data
3. ⬜ Modify preprocessing pipeline for emotion extraction
4. ⬜ Implement basic emotion embeddings in LightSpeech

### Short-term (Week 3-4):
5. ⬜ Train prosody-aware model on EDS
6. ⬜ Evaluate emotion preservation metrics
7. ⬜ Fine-tune hyperparameters

### Medium-term (Month 2):
8. ⬜ Prepare second language dataset (Chinese)
9. ⬜ Implement IPA phone mapping
10. ⬜ Create multilingual model architecture

### Long-term (Month 3+):
11. ⬜ Joint training on English + Chinese
12. ⬜ Evaluate cross-lingual transfer
13. ⬜ Deploy demo with emotion + language control

---

## References & Resources

**Prosody-Aware TTS:**
- GST-Tacotron: Style Tokens for Expressive TTS
- MelodyTTS: Explicit Prosody Modeling
- EmotiVoice: Multi-emotion Multi-speaker TTS

**Multilingual TTS:**
- YourTTS: Zero-shot Multi-speaker Multi-lingual TTS
- XTTS: Massive Multilingual TTS
- Mega-TTS: Zero-shot Cross-lingual TTS

**Datasets:**
- Emotional Speech Dataset (EDS) - Current focus
- AISHELL-3 - Chinese multi-speaker
- RAVDESS - English emotional speech
- Common Voice - 100+ languages

---

**Document Version:** 1.0  
**Date:** February 22, 2026  
**Author:** Analysis based on EDA results and LightSpeech architecture review
