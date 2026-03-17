"""
Inference Script for Prosody-Aware Multi-Speaker LightSpeech

Generate expressive speech with emotion and speaker control.

Usage:
    # Single text synthesis
    python inference_prosody.py \
        --model models/model_prosody.pt \
        --text "Hello, how are you today?" \
        --speaker 0 \
        --emotion happy \
        --output test_happy.wav
    
    # Interactive mode
    python inference_prosody.py \
        --model models/model_prosody.pt \
        --interactive

Available Emotions: angry, happy, neutral, sad, surprise
Available Speakers: 0-9 (corresponding to EDS speakers 0011-0020)
"""

import torch
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")

from lightspeech_prosody import ProsodyAwareModel
from preprocess import VOCODER

# Import text processing functions from predict.py
from predict import (
    convert_english_to_phonemes,
    convert_phones_to_tokens,
)

# Emotion mapping (must match training)
EMOTION_TO_ID = {
    'angry': 0,
    'happy': 1,
    'neutral': 2,
    'sad': 3,
    'surprise': 4
}

# Note: Text-to-phoneme conversion functions are imported from predict.py
# which already has proper g2p_en integration with ARPAbet to IPA mapping


def load_phone_vocab(phones_file="processed/phones.tsv"):
    """Load phone vocabulary."""
    try:
        phone_df = pd.read_csv(phones_file, sep='\t', keep_default_na=False)
        phone_to_id = dict(zip(phone_df['text'], phone_df['phone_id']))
        print(f"✓ Loaded {len(phone_to_id)} phones from {phones_file}")
        return phone_to_id
    except Exception as e:
        print(f"✗ Error loading phone vocabulary: {e}")
        print("Using default phone set...")
        # Fallback minimal phone set
        return {'<sil>': 0, '<unk>': 1, 'AH': 2, 'B': 3, 'D': 4, 'EH': 5}


def text_to_phones(text, phone_to_id):
    """
    Convert English text to phone IDs and tone IDs using predict.py functions.
    
    Args:
        text: Input English text
        phone_to_id: Dictionary mapping phonemes to IDs
        
    Returns:
        Tuple of (phone_ids, tone_ids)
    """
    # Convert text to IPA phonemes (using predict.py function)
    ipa_text = convert_english_to_phonemes(text)
    
    # Convert IPA phonemes to token IDs and stress IDs
    phone_ids, stress_ids, phonemes = convert_phones_to_tokens(phone_to_id, ipa_text)
    
    # For this model, we use stress_ids as tone_ids (compatible with training)
    # Note: stress_ids are 1-based (1=unstressed, 2=primary, 3=secondary)
    # Convert to 0-based for tone embeddings
    tone_ids = [s - 1 for s in stress_ids]
    
    return phone_ids, tone_ids


def load_model(checkpoint_path, device="cuda:1"):
    """Load trained prosody-aware model."""
    print(f"\n{'='*60}")
    print(f"Loading model from: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found at: {checkpoint_path}")
    
    # Move VOCODER to the same device
    VOCODER.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration
    num_phones = checkpoint.get('num_phones', 87)
    num_speakers = checkpoint.get('num_speakers', 10)
    num_emotions = checkpoint.get('num_emotions', 5)
    pitch_mean_std = checkpoint.get('pitch_mean_std', [0.0, 1.0])
    periodicity_range = checkpoint.get('periodicity_range', [0.0, 1.0])
    
    print(f"Model Configuration:")
    print(f"  • Phones: {num_phones}")
    print(f"  • Speakers: {num_speakers} (IDs: 0-{num_speakers-1})")
    print(f"  • Emotions: {num_emotions} ({', '.join(EMOTION_TO_ID.keys())})")
    print(f"  • Pitch stats: mean={pitch_mean_std[0]:.2f}, std={pitch_mean_std[1]:.2f}")
    print(f"  • Device: {device}\n")
    
    # Create model
    model = ProsodyAwareModel(
        num_phones=num_phones,
        num_speakers=num_speakers,
        num_mel_bins=VOCODER.num_mels,
        num_emotions=num_emotions,
        emotion_embedding_dim=64,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully!\n")
    
    return model, checkpoint


def synthesize(
    model,
    text,
    speaker_id,
    emotion,
    phone_to_id,
    checkpoint,
    device="cuda:1",
    duration_scale=1.0,
    pitch_scale=1.0,
):
    """Generate speech from text."""
    
    # Validate inputs
    if emotion.lower() not in EMOTION_TO_ID:
        raise ValueError(f"Invalid emotion '{emotion}'. Choose from: {list(EMOTION_TO_ID.keys())}")
    
    max_speaker = checkpoint.get('num_speakers', 10) - 1
    if not 0 <= speaker_id <= max_speaker:
        raise ValueError(f"Speaker ID must be 0-{max_speaker}, got {speaker_id}")
    
    emotion_id = EMOTION_TO_ID[emotion.lower()]
    
    print(f"Synthesizing:")
    print(f"  • Text: \"{text}\"")
    print(f"  • Speaker: {speaker_id}")
    print(f"  • Emotion: {emotion}")
    print(f"  • Duration scale: {duration_scale}x")
    print(f"  • Pitch scale: {pitch_scale}x\n")
    
    # Convert text to phone IDs and tone IDs using predict.py functions
    phones, tones = text_to_phones(text, phone_to_id)
    
    print(f"  • Phonemes: {len(phones)} phones generated")
    
    # Prepare tensors
    speaker_tensor = torch.tensor([speaker_id], dtype=torch.long).to(device)
    emotion_tensor = torch.tensor([emotion_id], dtype=torch.long).to(device)
    phones_tensor = torch.tensor([phones], dtype=torch.long).to(device)
    tones_tensor = torch.tensor([tones], dtype=torch.long).to(device)
    
    # Inference
    with torch.inference_mode():
        mel_pred, dur_pred, pitch_pred, periodicity_pred = model(
            speaker_tensor,
            phones_tensor,
            tones_tensor,
            emotion_tensor,
        )
        
        # Apply duration scaling
        if duration_scale != 1.0:
            dur_pred = dur_pred * duration_scale
        
        # Apply pitch scaling
        if pitch_scale != 1.0:
            pitch_pred = pitch_pred * pitch_scale
        
        # NOTE: Mel correction (+2.26) removed after fixing EDS preprocessing
        # If audio is still unclear, this indicates preprocessing issue
        # See NORMALIZATION_FIX_GUIDE.md for details
        
        # Generate audio from mel-spectrogram
        # Model returns [B, T, mel_bins], VOCODER expects [B, mel_bins, T]
        mel_pred = mel_pred.transpose(1, 2)  # [B, T, mel_bins] -> [B, mel_bins, T]
        audio = VOCODER(mel_pred).squeeze().cpu().numpy()
    
    duration_sec = len(audio) / VOCODER.sampling_rate
    print(f"  • Generated: {duration_sec:.2f}s of audio")
    print(f"  • Mel frames: {mel_pred.size(1)}\n")
    
    return audio


def save_audio(audio, output_path, sample_rate=22050):
    """Save audio to WAV file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Normalize to int16
    audio = np.clip(audio, -1, 1)
    audio = (audio * 32767).astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio)
    print(f"✓ Audio saved to: {output_path}\n")


def interactive_mode(model, checkpoint, phone_to_id, device):
    """Interactive TTS mode."""
    print(f"\n{'='*60}")
    print("INTERACTIVE PROSODY-AWARE TTS")
    print(f"{'='*60}\n")
    print("Commands:")
    print("  • Type text to synthesize")
    print("  • 'speaker N' - change speaker (0-9)")
    print("  • 'emotion NAME' - change emotion (angry/happy/neutral/sad/surprise)")
    print("  • 'speed N' - change speed (0.5-2.0, default 1.0)")
    print("  • 'pitch N' - change pitch (0.5-2.0, default 1.0)")
    print("  • 'quit' - exit\n")
    
    current_speaker = 0
    current_emotion = 'neutral'
    duration_scale = 1.0
    pitch_scale = 1.0
    
    while True:
        try:
            prompt = f"[Speaker {current_speaker}, {current_emotion}, speed={duration_scale:.1f}x, pitch={pitch_scale:.1f}x] > "
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nGoodbye!\n")
                break
            
            # Handle commands
            if user_input.lower().startswith('speaker '):
                try:
                    new_speaker = int(user_input.split()[1])
                    max_speaker = checkpoint['num_speakers'] - 1
                    if 0 <= new_speaker <= max_speaker:
                        current_speaker = new_speaker
                        print(f"✓ Speaker: {current_speaker}\n")
                    else:
                        print(f"✗ Speaker must be 0-{max_speaker}\n")
                except (ValueError, IndexError):
                    print("✗ Usage: speaker <0-9>\n")
                continue
            
            if user_input.lower().startswith('emotion '):
                try:
                    new_emotion = user_input.split()[1].lower()
                    if new_emotion in EMOTION_TO_ID:
                        current_emotion = new_emotion
                        print(f"✓ Emotion: {current_emotion}\n")
                    else:
                        print(f"✗ Choose from: {', '.join(EMOTION_TO_ID.keys())}\n")
                except IndexError:
                    print("✗ Usage: emotion <angry/happy/neutral/sad/surprise>\n")
                continue
            
            if user_input.lower().startswith('speed '):
                try:
                    duration_scale = float(user_input.split()[1])
                    duration_scale = max(0.5, min(2.0, duration_scale))
                    print(f"✓ Speed: {duration_scale:.1f}x\n")
                except (ValueError, IndexError):
                    print("✗ Usage: speed <0.5-2.0>\n")
                continue
            
            if user_input.lower().startswith('pitch '):
                try:
                    pitch_scale = float(user_input.split()[1])
                    pitch_scale = max(0.5, min(2.0, pitch_scale))
                    print(f"✓ Pitch: {pitch_scale:.1f}x\n")
                except (ValueError, IndexError):
                    print("✗ Usage: pitch <0.5-2.0>\n")
                continue
            
            # Synthesize speech
            audio = synthesize(
                model, user_input, current_speaker, current_emotion,
                phone_to_id, checkpoint, device, duration_scale, pitch_scale
            )
            
            # Save with timestamp
            import time
            timestamp = int(time.time())
            output_path = f"interactive_{current_speaker}_{current_emotion}_{timestamp}.wav"
            save_audio(audio, output_path, VOCODER.sampling_rate)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"✗ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Prosody-Aware TTS Inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--text", type=str,
                        help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0,
                        help="Speaker ID (0-9)")
    parser.add_argument("--emotion", type=str, default="neutral",
                        help="Emotion (angry/happy/neutral/sad/surprise)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio file")
    parser.add_argument("--duration_scale", type=float, default=1.0,
                        help="Duration scale (>1=slower, <1=faster)")
    parser.add_argument("--pitch_scale", type=float, default=1.0,
                        help="Pitch scale (>1=higher, <1=lower)")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device (cuda:0, cuda:1, cpu)")
    parser.add_argument("--interactive", action='store_true',
                        help="Interactive mode")
    
    args = parser.parse_args()
    
    # Load phone vocabulary
    phone_to_id = load_phone_vocab()
    
    # Load model
    model, checkpoint = load_model(args.model, args.device)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, checkpoint, phone_to_id, args.device)
        return
    
    # Single synthesis
    if not args.text:
        print("✗ Error: --text required (or use --interactive)")
        return
    
    audio = synthesize(
        model, args.text, args.speaker, args.emotion,
        phone_to_id, checkpoint, args.device,
        args.duration_scale, args.pitch_scale
    )
    
    save_audio(audio, args.output, VOCODER.sampling_rate)


if __name__ == "__main__":
    main()
