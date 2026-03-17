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

# Add the fastspeech2-clean directory to path
sys.path.insert(0, '../fastspeech2-clean')

# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:1" if torch.cuda.is_available() else "cpu"
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


# ===========================
# Streamlit App
# ===========================

st.set_page_config(
    page_title="TTS Model Comparison",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Text-to-Speech Model Comparison")
st.markdown("Compare two TTS models side-by-side: **LightSpeech** vs **Prosody-Aware Model**")

# ===========================
# Model Loading (runs once at startup)
# ===========================

@st.cache_resource(show_spinner=False)
def load_model1():
    """Load LightSpeech model - cached after first load."""
    model_path = '../fastspeech2-clean/model.pt'
    model, phone_dict, speaker_dict, words_dict, config = load_lightspeech_model(
        model_path, 
        model_class='lightspeech'
    )
    vocoder = load_vocoder()
    return model, phone_dict, vocoder


@st.cache_resource(show_spinner=False)
def load_model2():
    """Load Prosody-Aware model - cached after first load."""
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    
    # Load model with bfloat16 for speed/memory efficiency
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-expresso",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")
    
    # Move to GPU if available and keep it there permanently
    if torch.cuda.is_available():
        model.to(device)
    
    return model, tokenizer


# Initialize session state for model loading status
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.load_time = None

# Load models once at startup
if not st.session_state.models_loaded:
    load_start = time.time()
    with st.spinner("⏳ Loading models at startup... This happens only once."):
        try:
            model1, phone_dict1, vocoder = load_model1()
            st.sidebar.success("✅ LightSpeech ready")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load LightSpeech: {e}")
            model1, phone_dict1, vocoder = None, None, None
        
        try:
            prosody_model, prosody_tokenizer = load_model2()
            st.sidebar.success("✅ Prosody-Aware Model ready")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load Prosody-Aware Model: {e}")
            prosody_model, prosody_tokenizer = None, None
        
        st.session_state.models_loaded = True
        st.session_state.load_time = time.time() - load_start
        st.sidebar.info(f"🎉 Models loaded in {st.session_state.load_time:.1f}s! Ready for inference.")
else:
    # Models already loaded, just retrieve from cache
    try:
        model1, phone_dict1, vocoder = load_model1()
    except Exception as e:
        st.sidebar.error(f"❌ LightSpeech error: {e}")
        model1, phone_dict1, vocoder = None, None, None
    
    try:
        prosody_model, prosody_tokenizer = load_model2()
    except Exception as e:
        st.sidebar.error(f"❌ Prosody-Aware Model error: {e}")
        prosody_model, prosody_tokenizer = None, None

# Input section
st.markdown("---")

# Show model status
if st.session_state.models_loaded and model1 is not None and prosody_model is not None:
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
    options=["Angry", "Happy", "Neutral", "Sad", "Surprise"],
    index=1  # Default to "Happy"
)

generate_button = st.button(
    "🎵 Generate Speech", 
    type="primary", 
    use_container_width=True,
    disabled=not st.session_state.models_loaded or model1 is None or prosody_model is None
)

# Output sections
if generate_button and text_input:
    st.markdown("---")
    
    # Initialize timing variables
    lightspeech_time = None
    prosody_time = None
    audio1 = None
    audio2 = None
    error1 = None
    error2 = None
    
    # Define inference functions for concurrent execution
    def run_lightspeech():
        """Run LightSpeech inference."""
        try:
            start_time = time.time()
            # Always keep LightSpeech on CPU
            use_gpu = False
            
            audio = run_lightspeech_inference(
                model1, 
                vocoder, 
                text_input, 
                phone_dict1, 
                speaker_id=0,
                use_gpu=use_gpu
            )
            
            inference_time = time.time() - start_time
            return audio, inference_time, None
        except Exception as e:
            import traceback
            return None, None, (e, traceback.format_exc())
    
    def run_prosody():
        """Run Prosody-Aware model inference with streaming."""
        try:
            from transformers import set_seed
            from parler_tts import ParlerTTSStreamer
            from threading import Thread
            
            # Map UI emotion to model description
            emotion_mapping = {
                "Angry": "angry man speaking loudly",
                "Happy": "Happy woman speaking happily",
                "Neutral": "Neutral",
                "Sad": "Sad woman speaking saddly",
                "Surprise": "Surprised man speaking surprinsingly"
            }
            description = emotion_mapping.get(emotion_input, emotion_input)
            
            # Model is already on GPU, just determine device for tensors
            prosody_device = device if torch.cuda.is_available() else "cpu"
            
            input_ids = prosody_tokenizer(description, return_tensors="pt").input_ids.to(prosody_device)
            prompt_input_ids = prosody_tokenizer(text_input, return_tensors="pt").input_ids.to(prosody_device)
            
            set_seed(42)
            
            # Use streaming for faster generation
            streamer = ParlerTTSStreamer(prosody_model, device=prosody_device, play_steps=10)
            
            generation_kwargs = {
                "input_ids": input_ids,
                "prompt_input_ids": prompt_input_ids,
                "streamer": streamer,
            }
            
            # Start timing right before generation
            start_time = time.time()
            
            # Generate in a separate thread
            thread = Thread(target=prosody_model.generate, kwargs=generation_kwargs)
            thread.start()

            # Collect audio chunks as they're generated
            audio_chunks = []
            for new_audio in streamer:
                audio_chunks.append(new_audio)
            
            # End timing right before concatenating chunks
            inference_time = time.time() - start_time
            
            thread.join()
            
            # Combine all chunks
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
            else:
                audio = np.array([])
            
            # Clear cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return audio, inference_time, None
        except Exception as e:
            import traceback
            return None, None, (e, traceback.format_exc())
    
    # Run both models concurrently
    with st.spinner("🎵 Generating audio from both models..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future1 = executor.submit(run_lightspeech) if model1 is not None else None
            future2 = executor.submit(run_prosody) if prosody_model is not None else None
            
            # Wait for results
            if future1:
                audio1, lightspeech_time, error1 = future1.result()
            if future2:
                audio2, prosody_time, error2 = future2.result()
    
    col1, col2 = st.columns(2)
    
    # ========== LightSpeech Section ==========
    with col1:
        st.subheader("🔊 Model 1: LightSpeech")
        
        if model1 is not None and phone_dict1 is not None and vocoder is not None:
            if error1:
                st.error(f"❌ Error generating LightSpeech audio: {error1[0]}")
                st.code(error1[1])
            elif audio1 is not None and lightspeech_time is not None:
                # Save audio
                output_file1 = 'output_lightspeech.wav'
                audio1_int16 = np.int16(audio1 * 32767)
                sf.write(output_file1, audio1_int16, vocoder.sampling_rate)
                
                # Display audio player
                st.audio(output_file1, format='audio/wav')
                st.success(f"✅ Audio generated in **{lightspeech_time:.2f}s**")
                
                # Display timing metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Inference Time", f"{lightspeech_time:.2f}s")
                # with col_b:
                #     audio_duration = len(audio1) / vocoder.sampling_rate
                    # rtf = lightspeech_time / audio_duration if audio_duration > 0 else 0
                    # st.metric("Real-time Factor", f"{rtf:.2f}x")
                
                # Download button
                with open(output_file1, 'rb') as f:
                    st.download_button(
                        label="📥 Download LightSpeech Audio",
                        data=f,
                        file_name="lightspeech_output.wav",
                        mime="audio/wav"
                    )
        else:
            st.warning("⚠️ LightSpeech model not loaded")
    
    # ========== Prosody-Aware Model Section ==========
    with col2:
        st.subheader("🔊 Model 2: Prosody-Aware Model")
        
        if prosody_model is not None and prosody_tokenizer is not None:
            if error2:
                st.error(f"❌ Error generating Prosody-Aware Model audio: {error2[0]}")
                st.code(error2[1])
            elif audio2 is not None and prosody_time is not None:
                # Save audio
                output_file2 = 'output_prosody.wav'
                sf.write(output_file2, audio2, prosody_model.config.sampling_rate)
                
                # Display audio player
                st.audio(output_file2, format='audio/wav')
                st.success(f"✅ Audio generated in **{prosody_time:.2f}s**")
                
                # Display timing metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Inference Time", f"{prosody_time:.2f}s")
                # with col_b:
                #     audio_duration = len(audio2) / prosody_model.config.sampling_rate
                #     rtf = prosody_time / audio_duration if audio_duration > 0 else 0
                #     st.metric("Real-time Factor", f"{rtf:.2f}x")
                
                # Download button
                with open(output_file2, 'rb') as f:
                    st.download_button(
                        label="📥 Download Prosody-Aware Audio",
                        data=f,
                        file_name="prosody_aware_output.wav",
                        mime="audio/wav"
                    )
        else:
            st.warning("⚠️ Prosody-Aware Model not loaded")
    
    # ========== Performance Comparison ==========
    # if lightspeech_time is not None and prosody_time is not None:
    #     st.markdown("---")
    #     st.subheader("⚡ Performance Comparison")
        
    #     comp_col1, comp_col2, comp_col3 = st.columns(3)
        
    #     with comp_col1:
    #         st.metric(
    #             "LightSpeech", 
    #             f"{lightspeech_time:.2f}s",
    #             delta=f"{lightspeech_time - prosody_time:.2f}s" if lightspeech_time < prosody_time else None,
    #             delta_color="inverse"
    #         )
        
    #     with comp_col2:
    #         st.metric(
    #             "Prosody-Aware", 
    #             f"{prosody_time:.2f}s",
    #             delta=f"{prosody_time - lightspeech_time:.2f}s" if prosody_time < lightspeech_time else None,
    #             delta_color="inverse"
    #         )
        
    #     with comp_col3:
    #         if lightspeech_time < prosody_time:
    #             speedup = prosody_time / lightspeech_time
    #             st.metric("Winner", "LightSpeech 🏆", f"{speedup:.2f}x faster")
    #         else:
    #             speedup = lightspeech_time / prosody_time
    #             st.metric("Winner", "Prosody-Aware 🏆", f"{speedup:.2f}x faster")
        
    #     # Visual comparison bar
    #     st.markdown("#### Inference Time Comparison")
    #     max_time = max(lightspeech_time, prosody_time)
        
    #     # LightSpeech bar
    #     ls_percentage = (lightspeech_time / max_time) * 100
    #     st.markdown(f"**LightSpeech:** {lightspeech_time:.2f}s")
    #     st.progress(ls_percentage / 100)
        
    #     # Prosody-Aware bar
    #     pt_percentage = (prosody_time / max_time) * 100
    #     st.markdown(f"**Prosody-Aware Model:** {prosody_time:.2f}s")
    #     st.progress(pt_percentage / 100)

# Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
if st.session_state.models_loaded:
    st.sidebar.markdown("**Status:** 🟢 Ready")
    st.sidebar.markdown("**Mode:** Inference only")
    if st.session_state.load_time:
        st.sidebar.markdown(f"**Load time:** {st.session_state.load_time:.1f}s")
else:
    st.sidebar.markdown("**Status:** 🟡 Loading...")
