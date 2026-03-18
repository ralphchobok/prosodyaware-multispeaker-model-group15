import streamlit as st
import sys
import os
import torch
import numpy as np
import wave
from pathlib import Path
import soundfile as sf
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the training directory to path
sys.path.insert(0, '../training')

# Set device
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Emotion mapping for prosody model
EMOTION_TO_ID = {
    'Angry': 0,
    'Happy': 1,
    'Neutral': 2,
    'Sad': 3,
    'Surprise': 4
}

# ===========================
# LightSpeech Helper Functions
# ===========================

def load_lightspeech_model(model_path, model_class='lightspeech'):
    """Load a TTS model from checkpoint on CPU."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if model_class == 'fastspeech2':
        from fastspeech2 import Model
    elif model_class == 'lightspeech':
        from lightspeech import Model
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    model = Model(
        num_phones=checkpoint['num_phones'],
        num_speakers=checkpoint['num_speakers'],
        num_mel_bins=checkpoint['num_mel_bins'],
        d_model=checkpoint.get('d_model', 512)
    ).eval()  # Keep on CPU initially
    
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    return (
        model,
        checkpoint['phone_dict'],
        checkpoint['speaker_dict'],
        checkpoint.get('words_dict', checkpoint.get('pinyin_dict', {})),
        {
            'num_phones': checkpoint['num_phones'],
            'num_speakers': checkpoint['num_speakers'],
            'num_mel_bins': checkpoint['num_mel_bins'],
            'd_model': checkpoint.get('d_model', 512)
        }
    )


def load_vocoder(vocoder_name='hifigan_lj_ft_t2_v1'):
    """Load HiFi-GAN vocoder for mel-to-audio conversion on CPU."""
    vocoder = torch.hub.load(
        'lars76/bigvgan-mirror',
        vocoder_name,
        trust_repo=True,
        pretrained=True,
        verbose=False
    ).eval()  # Keep on CPU initially
    return vocoder


def convert_text_to_phonemes(text):
    """Convert English text to IPA phonemes using g2p_en."""
    from g2p_en import G2p
    
    arpabet_to_ipa = {
        'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ',
        'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
        'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ',
        'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
        'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
        'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
        'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
        'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
    }
    
    g2p = G2p()
    arpabet_phones = g2p(text)
    
    ipa_phones = []
    for phone in arpabet_phones:
        if not phone:
            continue
        base_phone = phone.rstrip('012')
        stress = phone[-1] if phone and phone[-1] in '012' else '0'
        
        if base_phone in arpabet_to_ipa:
            ipa_phone = arpabet_to_ipa[base_phone]
            ipa_phones.append(ipa_phone + stress)
    
    return " ".join(ipa_phones)


def phones_to_tokens(phone_text, phone_to_id):
    """Convert phoneme string to token IDs and stress IDs."""
    token_ids = []
    stress_ids = []
    phonemes = []
    
    sorted_phonemes = sorted(phone_to_id.keys(), key=len, reverse=True)
    
    for k in phone_text.split():
        if k in phone_to_id:
            phonemes.append(k)
            token_ids.append(phone_to_id[k])
            stress_ids.append(1)
            continue
        
        if k[-1].isdigit():
            stress_level = int(k[-1])
            stress_id = [1, 2, 3][min(stress_level, 2)]
            phone_key = k[:-1]
        else:
            stress_id = 1
            phone_key = k
        
        i = 0
        while i < len(phone_key):
            matched = False
            for phoneme in sorted_phonemes:
                if phone_key[i:].startswith(phoneme):
                    phonemes.append(phoneme)
                    token_ids.append(phone_to_id[phoneme])
                    stress_ids.append(stress_id)
                    i += len(phoneme)
                    matched = True
                    break
            if not matched:
                break
    
    return token_ids, stress_ids, phonemes


def run_lightspeech_inference(model, vocoder, text, phone_dict, speaker_id=0, add_silence=True, use_gpu=True):
    """Run inference on input text."""
    # Determine device
    inference_device = device if use_gpu and torch.cuda.is_available() else 'cpu'
    
    # Move models to device if needed
    if use_gpu and torch.cuda.is_available():
        model.to(inference_device)
        vocoder.to(inference_device)
    
    phone_text = convert_text_to_phonemes(text.lower())
    token_ids, stress_ids, phonemes = phones_to_tokens(phone_text, phone_dict)
    
    if add_silence:
        sil = phone_dict.get('<sil>', phone_dict.get('sil', 0))
        token_ids = [sil] + token_ids + [sil]
        stress_ids = [1] + stress_ids + [1]
        phonemes = ['<sil>'] + phonemes + ['<sil>']
    
    token_ids = torch.tensor([token_ids], dtype=torch.long).to(inference_device)
    stress_ids = torch.tensor([stress_ids], dtype=torch.long).to(inference_device)
    speaker_tensor = torch.tensor([speaker_id], dtype=torch.long).to(inference_device)
    
    with torch.inference_mode():
        mel, dur, pitch, _ = model(speaker_tensor, token_ids, stress_ids)
        audio = vocoder(mel.transpose(1, 2))
    
    audio = audio.cpu()
    
    # Move models back to CPU
    if use_gpu and torch.cuda.is_available():
        model.to('cpu')
        vocoder.to('cpu')
    
    return audio.flatten().numpy()


def run_prosody_inference(model, vocoder, text, phone_dict, emotion_id=2, speaker_id=0, add_silence=True, use_gpu=True):
    """Run inference on prosody-aware model with emotion."""
    # Determine device
    inference_device = device if use_gpu and torch.cuda.is_available() else 'cpu'
    
    # Move models to device if needed
    if use_gpu and torch.cuda.is_available():
        model.to(inference_device)
        vocoder.to(inference_device)
    
    phone_text = convert_text_to_phonemes(text.lower())
    token_ids, stress_ids, phonemes = phones_to_tokens(phone_text, phone_dict)
    
    if add_silence:
        sil = phone_dict.get('<sil>', phone_dict.get('sil', 0))
        token_ids = [sil] + token_ids + [sil]
        stress_ids = [1] + stress_ids + [1]
        phonemes = ['<sil>'] + phonemes + ['<sil>']
    
    token_ids = torch.tensor([token_ids], dtype=torch.long).to(inference_device)
    stress_ids = torch.tensor([stress_ids], dtype=torch.long).to(inference_device)
    speaker_tensor = torch.tensor([speaker_id], dtype=torch.long).to(inference_device)
    emotion_tensor = torch.tensor([emotion_id], dtype=torch.long).to(inference_device)
    
    with torch.inference_mode():
        mel, dur, pitch, _ = model(speaker_tensor, token_ids, stress_ids, emotion_tensor)
        audio = vocoder(mel.transpose(1, 2))
    
    audio = audio.cpu()
    
    # Move models back to CPU
    if use_gpu and torch.cuda.is_available():
        model.to('cpu')
        vocoder.to('cpu')
    
    return audio.flatten().numpy()


# ===========================
# Streamlit App
# ===========================

st.set_page_config(
    page_title="TTS Model Comparison - Base vs Prosody",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Text-to-Speech Model Comparison")
st.markdown("Compare two TTS models side-by-side: **Base Model** vs **Prosody-Aware Model**")

# ===========================
# Model Loading (runs once at startup)
# ===========================

@st.cache_resource(show_spinner=False)
def load_base_model():
    """Load Base LightSpeech model - cached after first load."""
    model_path = '../models/model.pt'
    model, phone_dict, speaker_dict, words_dict, config = load_lightspeech_model(
        model_path, 
        model_class='lightspeech'
    )
    vocoder = load_vocoder()
    return model, phone_dict, vocoder


@st.cache_resource(show_spinner=False)
def load_prosody_model():
    """Load Prosody-Aware LightSpeech model - cached after first load."""
    from lightspeech_prosody import ProsodyAwareModel
    
    model_path = '../models/model2.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = ProsodyAwareModel(
        num_phones=checkpoint['num_phones'],
        num_speakers=checkpoint['num_speakers'],
        num_mel_bins=checkpoint['num_mel_bins'],
        num_emotions=checkpoint.get('num_emotions', 5),
        d_model=checkpoint.get('d_model', 512),
        emotion_embedding_dim=checkpoint.get('emotion_embedding_dim', 64)
    ).eval()  # Keep on CPU initially
    
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    # Use the same vocoder as base model
    vocoder = load_vocoder()
    return model, checkpoint['phone_dict'], vocoder


# Initialize session state for model loading status
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.load_time = None

# Load models once at startup
if not st.session_state.models_loaded:
    load_start = time.time()
    with st.spinner("⏳ Loading models at startup... This happens only once."):
        try:
            base_model, base_phone_dict, base_vocoder = load_base_model()
            st.sidebar.success("✅ Base Model ready")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load Base Model: {e}")
            base_model, base_phone_dict, base_vocoder = None, None, None
        
        try:
            prosody_model, prosody_phone_dict, prosody_vocoder = load_prosody_model()
            st.sidebar.success("✅ Prosody-Aware Model ready")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load Prosody-Aware Model: {e}")
            prosody_model, prosody_phone_dict, prosody_vocoder = None, None, None
        
        st.session_state.models_loaded = True
        st.session_state.load_time = time.time() - load_start
        st.sidebar.info(f"🎉 Models loaded in {st.session_state.load_time:.1f}s! Ready for inference.")
else:
    # Models already loaded, just retrieve from cache
    try:
        base_model, base_phone_dict, base_vocoder = load_base_model()
    except Exception as e:
        st.sidebar.error(f"❌ Base Model error: {e}")
        base_model, base_phone_dict, base_vocoder = None, None, None
    
    try:
        prosody_model, prosody_phone_dict, prosody_vocoder = load_prosody_model()
    except Exception as e:
        st.sidebar.error(f"❌ Prosody-Aware Model error: {e}")
        prosody_model, prosody_phone_dict, prosody_vocoder = None, None, None

# Input section
st.markdown("---")

# Show model status
if st.session_state.models_loaded and base_model is not None and prosody_model is not None:
    st.success("✅ Models loaded and ready for inference!")
elif st.session_state.models_loaded:
    st.warning("⚠️ Some models failed to load. Check sidebar for details.")
else:
    st.info("⏳ Loading models...")

text_input = st.text_area(
    "Enter text to synthesize:",
    value="Hello world, this is a test of the speech synthesis system.",
    height=100
)

# Emotion input for prosody control
emotion_input = st.selectbox(
    "Emotion/Style for Prosody-Aware Model:",
    options=list(EMOTION_TO_ID.keys()),
    index=2  # Default to "Neutral"
)

generate_button = st.button(
    "🎵 Generate Speech", 
    type="primary", 
    use_container_width=True,
    disabled=not st.session_state.models_loaded or base_model is None or prosody_model is None
)

# Output sections
if generate_button and text_input:
    st.markdown("---")
    
    # Initialize timing variables
    base_time = None
    prosody_time = None
    audio1 = None
    audio2 = None
    error1 = None
    error2 = None
    
    # Define inference functions for concurrent execution
    def run_base_model():
        """Run Base Model inference."""
        try:
            start_time = time.time()
            # Use GPU for inference
            use_gpu = True
            
            audio = run_lightspeech_inference(
                base_model, 
                base_vocoder, 
                text_input, 
                base_phone_dict, 
                use_gpu=use_gpu
            )
            
            inference_time = time.time() - start_time
            return audio, inference_time, None
        except Exception as e:
            import traceback
            return None, None, (e, traceback.format_exc())
    
    def run_prosody_model_inference():
        """Run Prosody-Aware Model inference."""
        try:
            start_time = time.time()
            # Use GPU for inference
            use_gpu = True
            
            # Get emotion ID from selection
            emotion_id = EMOTION_TO_ID.get(emotion_input, 2)  # Default to Neutral
            
            audio = run_prosody_inference(
                prosody_model, 
                prosody_vocoder, 
                text_input, 
                prosody_phone_dict,
                emotion_id=emotion_id,
                speaker_id=0,
                use_gpu=use_gpu
            )
            
            inference_time = time.time() - start_time
            return audio, inference_time, None
        except Exception as e:
            import traceback
            return None, None, (e, traceback.format_exc())
    
    # Run both models concurrently
    with st.spinner("🎵 Generating audio from both models..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future1 = executor.submit(run_base_model) if base_model is not None else None
            future2 = executor.submit(run_prosody_model_inference) if prosody_model is not None else None
            
            # Wait for results
            if future1:
                audio1, base_time, error1 = future1.result()
            if future2:
                audio2, prosody_time, error2 = future2.result()
    
    col1, col2 = st.columns(2)
    
    # ========== Base Model Section ==========
    with col1:
        st.subheader("🔊 Model 1: Base Model")
        
        if base_model is not None and base_phone_dict is not None and base_vocoder is not None:
            if error1:
                st.error(f"❌ Error generating Base Model audio: {error1[0]}")
                st.code(error1[1])
            elif audio1 is not None and base_time is not None:
                # Save audio
                output_file1 = 'output_base_model.wav'
                audio1_int16 = np.int16(audio1 * 32767)
                sf.write(output_file1, audio1_int16, base_vocoder.sampling_rate)
                
                # Display audio player
                st.audio(output_file1, format='audio/wav')
                st.success(f"✅ Audio generated in **{base_time:.2f}s**")
                
                # Display timing metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Inference Time", f"{base_time:.2f}s")
                with col_b:
                    audio_duration = len(audio1) / base_vocoder.sampling_rate
                    rtf = base_time / audio_duration if audio_duration > 0 else 0
                    st.metric("Real-time Factor", f"{rtf:.2f}x")
                
                # Download button
                with open(output_file1, 'rb') as f:
                    st.download_button(
                        label="📥 Download Base Model Audio",
                        data=f,
                        file_name="base_model_output.wav",
                        mime="audio/wav"
                    )
        else:
            st.warning("⚠️ Base Model not loaded")
    
    # ========== Prosody-Aware Model Section ==========
    with col2:
        st.subheader("🔊 Model 2: Prosody-Aware Model")
        
        if prosody_model is not None and prosody_phone_dict is not None and prosody_vocoder is not None:
            if error2:
                st.error(f"❌ Error generating Prosody-Aware Model audio: {error2[0]}")
                st.code(error2[1])
            elif audio2 is not None and prosody_time is not None:
                # Save audio
                output_file2 = 'output_prosody_model.wav'
                audio2_int16 = np.int16(audio2 * 32767)
                sf.write(output_file2, audio2_int16, prosody_vocoder.sampling_rate)
                
                # Display audio player
                st.audio(output_file2, format='audio/wav')
                st.success(f"✅ Audio generated in **{prosody_time:.2f}s**")
                
                # Display timing metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Inference Time", f"{prosody_time:.2f}s")
                with col_b:
                    audio_duration = len(audio2) / prosody_vocoder.sampling_rate
                    rtf = prosody_time / audio_duration if audio_duration > 0 else 0
                    st.metric("Real-time Factor", f"{rtf:.2f}x")
                
                # Download button
                with open(output_file2, 'rb') as f:
                    st.download_button(
                        label="📥 Download Prosody-Aware Audio",
                        data=f,
                        file_name="prosody_model_output.wav",
                        mime="audio/wav"
                    )
        else:
            st.warning("⚠️ Prosody-Aware Model not loaded")
    
    # ========== Performance Comparison ==========
    if base_time is not None and prosody_time is not None:
        st.markdown("---")
        st.subheader("⚡ Performance Comparison")
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            st.metric(
                "Base Model", 
                f"{base_time:.2f}s",
                delta=f"{base_time - prosody_time:.2f}s" if base_time < prosody_time else None,
                delta_color="inverse"
            )
        
        with comp_col2:
            st.metric(
                "Prosody-Aware", 
                f"{prosody_time:.2f}s",
                delta=f"{prosody_time - base_time:.2f}s" if prosody_time < base_time else None,
                delta_color="inverse"
            )
        
        with comp_col3:
            if base_time < prosody_time:
                speedup = prosody_time / base_time
                st.metric("Winner", "Base Model 🏆", f"{speedup:.2f}x faster")
            else:
                speedup = base_time / prosody_time
                st.metric("Winner", "Prosody-Aware 🏆", f"{speedup:.2f}x faster")
        
        # Visual comparison bar
        st.markdown("#### Inference Time Comparison")
        max_time = max(base_time, prosody_time)
        
        # Base Model bar
        base_percentage = (base_time / max_time) * 100
        st.markdown(f"**Base Model:** {base_time:.2f}s")
        st.progress(base_percentage / 100)
        
        # Prosody-Aware bar
        prosody_percentage = (prosody_time / max_time) * 100
        st.markdown(f"**Prosody-Aware Model:** {prosody_time:.2f}s")
        st.progress(prosody_percentage / 100)

# Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st.sidebar.markdown("**Models:**")
st.sidebar.markdown("- Base: `models/model1.pt`")
st.sidebar.markdown("- Prosody: `models/model2.pt`")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Emotion Support")
st.sidebar.markdown("Prosody model supports:")
for emotion in EMOTION_TO_ID.keys():
    st.sidebar.markdown(f"- {emotion}")
st.sidebar.markdown("---")

if st.session_state.models_loaded:
    st.sidebar.markdown("**Status:** 🟢 Ready")
    st.sidebar.markdown("**Mode:** Inference only")
    if st.session_state.load_time:
        st.sidebar.markdown(f"**Load time:** {st.session_state.load_time:.1f}s")
else:
    st.sidebar.markdown("**Status:** 🟡 Loading...")
