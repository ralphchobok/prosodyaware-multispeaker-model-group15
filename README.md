# 🎤 Prosody-Aware Multi-Speaker LightSpeech TTS

This capstone project focuses on extending the LightSpeech Text-to-Speech (TTS) model to support:

- 🎶 **Prosody awareness** (improved rhythm, stress, and intonation)
- 🗣️ **Multi-speaker capability** (handling multiple voices within a single model)

---

## 📌 Project Overview

The goal of this project is to enhance the baseline LightSpeech model by:

1. Training a base TTS model
2. Introducing prosodic features for more natural speech synthesis
3. Enabling multi-speaker support using curated datasets

---

## ⚙️ Setup Instructions

### 1. Create Environment

Create a Conda environment with Python 3.11 and install dependencies:
```bash
conda create -n lightspeech python=3.11
conda activate lightspeech
pip install -r requirements.txt
```

### 2. Download Datasets

Run the scripts inside the `download_dataset` directory:
```bash
cd download_dataset
# run the appropriate download scripts
```

### 3. Clean Dataset

Remove Chinese speakers from the ESD dataset:

- Chinese speaker IDs range from `0000` to `0010`
- Delete these speakers before proceeding

---

## 🧹 Data Preprocessing

### 4. MFA Alignment Preparation

Prepare datasets for alignment using scripts in `preprocess_datasets`:
```bash
cd preprocess_datasets

python prepare_esd.py
python prepare_ljspeech.py
```

These scripts perform dataset preparation for **Montreal Forced Aligner (MFA)**.

### 5. Dataset Processing
✅ Extracts mel-spectrograms, pitch, periodicity
✅ Handles path differences (TextGrids in textgrids/, WAVs in datasets/)
✅ Generates unified phones.tsv, words.tsv, speakers.tsv

Process aligned datasets:
```bash
python process_esd.py
python process_ljspeech.py
```

---

## 🧠 Model Training

### 6. Train Base Model

Train the initial LightSpeech model:
```bash
cd training
python train_base_model.py
```

### 7. Train Prosody Model

Enhance the model with prosody awareness:
```bash
python train_prosody.py
```

---

## 📁 Project Structure
```
.
├── datasets/
├── download_dataset/
├── preprocess_datasets/
│   ├── prepare_esd.py
│   ├── prepare_ljspeech.py
│   ├── process_esd.py
│   └── process_ljspeech.py
├── models/
├── output_examples/
├── training/
│   ├── train_base_model.py
│   └── train_prosody.py
├── requirements.txt
└── README.md
```

---

## 🚀 Key Features

- ✅ Multi-speaker TTS training pipeline
- ✅ Prosody-aware speech synthesis
- ✅ Support for ESD and LJSpeech datasets
- ✅ Modular preprocessing and training scripts

---

## 📌 Notes

- Ensure datasets are properly aligned before processing
- Double-check removal of Chinese speakers from ESD dataset
- Training may require significant GPU resources