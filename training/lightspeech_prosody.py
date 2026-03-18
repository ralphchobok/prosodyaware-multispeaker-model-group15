"""
Prosody-Aware LightSpeech Model with Emotion Conditioning

This extends the base LightSpeech model with emotion embeddings for
prosody-aware multi-speaker text-to-speech synthesis.

Key Features:
- Emotion embeddings (5 emotions: Angry, Happy, Neutral, Sad, Surprise)
- Multi-speaker support (inherited from base model)
- Pitch and duration prediction with emotion conditioning
"""

from typing import List, Optional, Tuple

import math
import torch
from torch import nn, Tensor


class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvSeparable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0,
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * out_channels))
        nn.init.normal_(self.depthwise_conv.weight, mean=0, std=std)
        nn.init.normal_(self.pointwise_conv.weight, mean=0, std=std)
        nn.init.zeros_(self.pointwise_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class SepConvLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layer_norm = LayerNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = nn.ReLU(inplace=False)
        self.conv1 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)
        self.conv2 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.activation_fn(self.conv1(x))
        x = self.dropout(x)
        x = self.activation_fn(self.conv2(x))
        x = self.dropout(x)
        return residual + x


class ProsodyAwareModel(nn.Module):
    """
    Prosody-Aware LightSpeech Model
    
    Extends base LightSpeech with emotion conditioning for expressive TTS.
    """
    def __init__(
        self,
        num_phones: int,
        num_speakers: int,
        num_mel_bins: int,
        num_emotions: int = 5,  # NEW: Number of emotion categories
        emotion_embedding_dim: int = 64,  # NEW: Emotion embedding size
        num_tones: int = 7,
        tone_embedding: int = 16,
        d_model: int = 512,
        layer_dropout: float = 0.2,
        encoder_kernel_sizes: List[int] = [5, 25, 13, 9],
        decoder_kernel_sizes: List[int] = [17, 21, 9, 3],
        duration_layers: int = 1,
        duration_kernel_size: int = 3,
        duration_dropout: float = 0.25,
        pitch_layers: int = 6,
        pitch_kernel_size: int = 5,
        pitch_dropout: float = 0.25,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.num_emotions = num_emotions
        self.emotion_embedding_dim = emotion_embedding_dim

        # Speaker embeddings
        self.num_speakers = num_speakers
        if self.num_speakers > 1:
            self.speaker_embedding = nn.Embedding(self.num_speakers, d_model)
        
        # NEW: Emotion embeddings
        self.emotion_embedding = nn.Embedding(self.num_emotions, emotion_embedding_dim)
        
        # Phone and tone embeddings
        self.embed_tokens = nn.Embedding(
            num_phones, d_model - tone_embedding, padding_idx=self.padding_idx
        )
        self.embed_tones = nn.Embedding(
            num_tones, tone_embedding, padding_idx=self.padding_idx
        )
        
        self.dropout = nn.Dropout(layer_dropout)
        self.embed_pitch = nn.Conv1d(2, d_model, kernel_size=1)

        # Encoder
        self.encoder = nn.ModuleList(
            [
                SepConvLayer(d_model, kernel_size, layer_dropout)
                for kernel_size in encoder_kernel_sizes
            ]
        )
        
        # NEW: Projection layer to combine encoder output with emotion
        self.emotion_projection = nn.Linear(d_model + emotion_embedding_dim, d_model)
        
        # Decoder
        self.decoder = nn.ModuleList(
            [
                SepConvLayer(d_model, kernel_size, layer_dropout)
                for kernel_size in decoder_kernel_sizes
            ]
        )

        # Predictors
        self.duration_predictor = self._make_predictor(
            hidden_size=d_model,
            out_dim=1,
            num_layers=duration_layers,
            kernel_size=duration_kernel_size,
            dropout=duration_dropout,
        )
        self.pitch_predictor = self._make_predictor(
            hidden_size=d_model,
            out_dim=2,
            num_layers=pitch_layers,
            kernel_size=pitch_kernel_size,
            dropout=pitch_dropout,
        )

        self.layer_norm = LayerNorm1d(d_model)
        self.layer_norm2 = LayerNorm1d(d_model)
        self.mel_out = nn.Conv1d(d_model, num_mel_bins, kernel_size=1)

    @staticmethod
    def _make_predictor(
        hidden_size: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        kernel_size: int = 3,
    ):
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    ConvSeparable(hidden_size, hidden_size, kernel_size),
                    nn.ReLU(inplace=False),
                    LayerNorm1d(hidden_size),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Conv1d(hidden_size, out_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def _length_regulator(self, x: Tensor, mel_time: int, durations: Tensor) -> Tensor:
        bsz, time, feats = x.shape
        if bsz > 1:
            cumulative_durations = torch.cumsum(durations, dim=1)

            # Create a range tensor for each batch item
            expanded_range = (
                torch.arange(mel_time, device=x.device).unsqueeze(0).expand(bsz, -1)
            )

            # Create a mask for valid positions
            mask = expanded_range.unsqueeze(1) >= cumulative_durations.unsqueeze(2)

            # Calculate source indices
            source_indices = mask.long().sum(dim=1)

            # Clamp the indices to handle cases where mel_time > total_duration
            source_indices = torch.clamp(source_indices, 0, time - 1)

            # Create the gather indices tensor
            gather_indices = source_indices.unsqueeze(-1).expand(-1, -1, feats)

            # Gather the input tensor based on the calculated indices
            return torch.gather(x, 1, gather_indices)
        else:
            indices = torch.arange(time, device=x.device)
            repeated_indices = torch.repeat_interleave(
                indices, durations[0].long(), dim=0
            )
            return x[:, repeated_indices]

    def forward(
        self,
        speakers: Tensor,
        tokens: Tensor,
        tones: Tensor,
        emotions: Tensor,  # NEW: Emotion IDs [batch_size]
        pitches: Optional[Tensor] = None,
        periodicity: Optional[Tensor] = None,
        durations: Optional[Tensor] = None,
        mels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass with emotion conditioning.
        
        Args:
            speakers: Speaker IDs [batch_size]
            tokens: Phone tokens [batch_size, seq_len]
            tones: Tone/stress tokens [batch_size, seq_len]
            emotions: Emotion IDs [batch_size] - NEW
            pitches: Target pitch contours [batch_size, mel_len] (optional, for training)
            periodicity: Pitch periodicity [batch_size, mel_len] (optional, for training)
            durations: Phone durations [batch_size, seq_len] (optional, for training)
            mels: Target mel spectrograms [batch_size, mel_len, mel_bins] (optional, for training)
        
        Returns:
            mel_pred: Predicted mel spectrogram
            duration_pred: Predicted durations
            pitch_pred: Predicted pitch contour
            periodicity_pred: Predicted periodicity
        """
        # Embed phones and tones
        x = torch.cat(
            (self.embed_tokens(tokens), self.embed_tones(tones)), dim=-1
        ).transpose(1, 2)

        # Encode
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        encoder_outputs = self.layer_norm(x).transpose(1, 2)  # [B, T, d_model]

        # Add speaker embedding
        if self.num_speakers > 1:
            encoder_outputs = encoder_outputs + self.speaker_embedding(speakers.long()).unsqueeze(1)
        
        # NEW: Add emotion conditioning
        # Get emotion embeddings [B, emotion_dim]
        emotion_emb = self.emotion_embedding(emotions.long())
        # Expand to sequence length [B, T, emotion_dim]
        emotion_emb_expanded = emotion_emb.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        # Concatenate and project
        encoder_outputs_with_emotion = torch.cat([encoder_outputs, emotion_emb_expanded], dim=-1)
        encoder_outputs = self.emotion_projection(encoder_outputs_with_emotion)  # [B, T, d_model]

        # Duration prediction
        duration_prediction = self.duration_predictor(
            encoder_outputs.transpose(1, 2)
        ).squeeze(1)

        # Length regulation
        if mels is not None and durations is not None:
            durations = torch.clamp(torch.round(durations), min=0).long()
            mel_time = mels.shape[1]
            assert torch.max(torch.sum(durations, dim=1)).item() == mel_time
        else:
            duration_prediction = torch.exp(duration_prediction) - 1
            durations = torch.clamp(torch.round(duration_prediction), min=0).long()
            mel_time = torch.max(torch.sum(durations, dim=1)).long()

        decoder_inp = self._length_regulator(encoder_outputs, mel_time, durations)
        decoder_inp = self.dropout(decoder_inp).transpose(1, 2)

        # Pitch prediction
        pitch_feat = self.pitch_predictor(decoder_inp)
        new_feat = (
            torch.stack((pitches, periodicity), dim=2).transpose(1, 2)
            if pitches is not None
            else pitch_feat.clone()
        )
        new_feat = new_feat.detach()

        decoder_inp = decoder_inp + self.embed_pitch(new_feat)

        # Decode
        for decoder_layer in self.decoder:
            decoder_inp = decoder_layer(decoder_inp)
        decoder_outputs = self.layer_norm2(decoder_inp)

        # Mel output
        decoder_outputs = self.mel_out(decoder_outputs).transpose(1, 2)

        return decoder_outputs, duration_prediction, pitch_feat[:, 0], pitch_feat[:, 1]


# Alias for easier import
Model = ProsodyAwareModel
