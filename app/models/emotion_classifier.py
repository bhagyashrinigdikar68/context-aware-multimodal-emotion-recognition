"""
Emotion classifier head: takes the fused representation and produces logits
over `settings.EMOTION_LABELS`. Output dimensionality is derived from the
config, never hard-coded to 7, so adding/removing emotion classes only
requires editing `settings.EMOTION_LABELS`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import settings


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim: int = settings.FUSED_DIM, num_classes: int | None = None, hidden_dim: int = 128):
        super().__init__()
        num_classes = num_classes or settings.num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, num_classes) -- apply softmax outside for probabilities."""
        return self.net(fused)

    @staticmethod
    def predict(logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)
        return pred_idx, confidence, probs
