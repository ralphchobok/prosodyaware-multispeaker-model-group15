# TTS Model Comparison Streamlit App

Compare two Text-to-Speech models side-by-side:
- **LightSpeech** (custom model)
- **Parler-TTS** (Hugging Face model with emotion control)

## Installation

```bash
cd streamlit_app
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```
**Important:** Models are loaded automatically when Streamlit starts. Wait for the "✅ Models loaded and ready for inference!" message before generating audio.

### Workflow:
1. **Startup** - Models load automatically (happens once, ~10-30 seconds)
2. **Ready** - Green status indicator appears
3. **Generate** - Click button to run inference (fast, no loading)
4. **Results** - Both models run concurrently and display results
Then:
1. Enter text in the input box
2. Optionally adjust the emotion/style for Parler-TTS (e.g., "happy", "sad", "excited")
3. Click "🎵 Generate Speech"
4. Listen to both outputs and download if needed

## GPU Memory Management

The app implements smart memory management to prevent CUDA out-of-memory errors:

### Strategy:
- **Both models (LightSpeech + Parler-TTS)**: 
  - Loaded on CPU initially
  - Moved to GPU only during inference
  - Moved back to CPU after generation
  - CUDA cache cleared after each inference

This means **zero GPU memory usage when idle** and minimal memory during inference!

### If you still get OOM errors:

**Option 1: Run Parler-TTS on CPU only**

Modify line ~286 in `app.py`:
```python
# Change this:
parler_device = device if use_gpu else "cpu"

# To this:
parler_device = "cpu"  # Force CPU
```

**Option 2: Use smaller batch/sequence length**

Add generation parameters:
```python
generation = parler_model.generate(
    input_ids=input_ids, 
    prompt_input_ids=prompt_input_ids,
    max_length=512,  # Limit output length
)
```

**Option 3: Use only one model at a time**

Comment out one model's loading section to free up memory.

## Troubleshooting

### CUDA Out of Memory
- Run with CPU: `CUDA_VISIBLE_DEVICES="" streamlit run app.py`
- Reduce GPU memory usage by implementing Option 1 or 2 above
- Check GPU memory: `nvidia-smi`

### Model Loading Errors
- Ensure `../fastspeech2-clean/model.pt` exists
- Check Python path includes `fastspeech2-clean` directory
- Verify all dependencies are installed

### Audio Generation Fails
- Check the error message in the Streamlit interface
- Verify input text is valid English
- Try shorter text inputs initially

## Features

✅ Single text input generates audio from both models  
✅ Two-column side-by-side comparison  
✅ Audio players for immediate playback  
✅ Download buttons for generated audio  
✅ Emotion/style control for Parler-TTS  
✅ Automatic GPU memory management  
✅ Model caching for fast reloads  

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works)
- 8GB+ GPU memory (for both models on GPU)
- 16GB+ RAM

## Notes

- First run downloads Parler-TTS (~1GB) and vocoder models
- Subsequent runs are faster due to Streamlit's caching
- The app keeps models in memory between generations (faster inference)
- GPU memory is managed automatically to prevent OOM errors
