"""
Module 4: Audio encoder.

Takes handcrafted prosodic/spectral features (from
`app/preprocessing/audio.py`) and projects them through a small learned
network (Linear -> BiGRU-style temporal smoothing is unnecessary at the
utterance level, so we use an MLP with residual-ish structure) into the
shared audio embedding space. This keeps compute CPU-friendly while still
being a trained neural encoder rather than raw handcrafted features passed
straight to the classifier, matching Module 10's requirement.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.config import settings
from app.preprocessing.audio import AudioSegmentFeatures


class AudioEncoder(nn.Module):
    def __init__(self, input_dim: int | None = None, output_dim: int | None = None):
        super().__init__()
        input_dim = input_dim or AudioSegmentFeatures.vector_dim()
        output_dim = output_dim or settings.AUDIO_EMBED_DIM

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, feature_batch: torch.Tensor) -> torch.Tensor:
        """feature_batch: (B, input_dim) raw handcrafted feature vectors."""
        return self.net(feature_batch)

    @staticmethod
    def missing_embedding(batch_size: int, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, settings.AUDIO_EMBED_DIM, device=device)
