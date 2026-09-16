"""
Module 6: Conversational context encoder. ⭐

Models each utterance's representation as a function of a configurable
window of *previous* utterances plus the speaker embedding, using a
Transformer encoder over the utterance sequence (Module 6 lists this as the
preferred approach over BiLSTM/GRU).

Input to this module is a per-utterance "content" embedding -- a
lightweight fusion of that utterance's own text/audio/visual signal, done
upstream so the context transformer reasons over a fixed-size sequence
rather than three separate modality streams. (The *final* prediction fusion
in `fusion.py` then combines this context vector back together with the
raw text/audio/visual embeddings for the current utterance, matching the
Module 14 diagram where "Context embedding" is itself one of the four
fusion inputs.)

A causal mask is used by default (only past utterances, matching real-time
use); an optional bidirectional mode is available for offline analysis
where subsequent context is also allowed (Module 6 explicitly permits this
for offline analysis).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from app.config import settings


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        speaker_dim: int = settings.SPEAKER_EMBED_DIM,
        hidden_dim: int = settings.FUSED_DIM,
        num_layers: int = 2,
        num_heads: int = 4,
        context_window: int | None = None,
        causal: bool = True,
    ):
        super().__init__()
        self.context_window = context_window or settings.CONTEXT_WINDOW
        self.causal = causal
        self.output_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim + speaker_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        utterance_embeddings: torch.Tensor,   # (B, T, input_dim)  T = context_window (+1 for current)
        speaker_embeddings: torch.Tensor,     # (B, T, speaker_dim)
        padding_mask: torch.Tensor | None = None,  # (B, T) True where PADDED (to ignore)
    ) -> torch.Tensor:
        """
        Returns the context representation for the *last* position in the
        sequence (i.e. the current utterance, informed by everything before
        it), shape (B, hidden_dim).
        """
        x = torch.cat([utterance_embeddings, speaker_embeddings], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        seq_len = x.size(1)
        causal_mask = None
        if self.causal:
            # bool mask (True = masked) to match the bool dtype of src_key_padding_mask;
            # PyTorch's MultiheadAttention warns/deprecates mixing float and bool masks.
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
            )

        encoded = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return encoded[:, -1, :]  # representation of the current (last) utterance

    @staticmethod
    def missing_embedding(batch_size: int, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, settings.FUSED_DIM, device=device)
