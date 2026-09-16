"""
Module 7: Speaker embeddings.

Speakers are represented purely as anonymous indices within a conversation
(Speaker 1, Speaker 2, ...) -- never real identities (see Module 13). This
module maps those indices to a learned embedding used inside the context
encoder so the model can distinguish "who is speaking" without knowing who
they are.

A conversation may have more speakers than seen during training; indices
beyond `max_speakers` are clamped to a shared "unknown speaker" slot.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.config import settings


class SpeakerEmbedding(nn.Module):
    def __init__(self, max_speakers: int = 16, embed_dim: int | None = None):
        super().__init__()
        embed_dim = embed_dim or settings.SPEAKER_EMBED_DIM
        self.max_speakers = max_speakers
        # +1 slot reserved for "unknown / overflow" speaker index
        self.embedding = nn.Embedding(max_speakers + 1, embed_dim)
        self.output_dim = embed_dim

    def forward(self, speaker_indices: torch.Tensor) -> torch.Tensor:
        clamped = speaker_indices.clamp(max=self.max_speakers)
        return self.embedding(clamped)

    @staticmethod
    def parse_speaker_label(label: str) -> int:
        """'Speaker 3' -> 2 (zero-indexed). Falls back to 0 for unparsable labels."""
        try:
            return int(label.strip().split()[-1]) - 1
        except (ValueError, IndexError):
            return 0
