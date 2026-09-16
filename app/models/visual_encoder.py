"""
Module 5: Visual/facial encoder.

A small CNN over cropped, aligned face images (224x224x3). Deliberately kept
lightweight (a few conv blocks, not a full ResNet) so it trains and infers
on CPU within the stated computational constraints. The class is written so
`self.backbone` can be swapped for a pretrained facial-expression model
(e.g. a HF vision model or a pretrained FER checkpoint) without touching any
other module -- fusion and context code only depend on `output_dim` and the
`forward()` contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.config import settings


class SmallFaceCNN(nn.Module):
    """Lightweight CNN backbone: 4 conv blocks + global average pool."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224), normalized to [0,1]
        feats = self.features(x).flatten(1)
        return self.proj(feats)


class VisualEncoder(nn.Module):
    def __init__(self, output_dim: int | None = None, backbone: nn.Module | None = None):
        super().__init__()
        output_dim = output_dim or settings.VISUAL_EMBED_DIM
        self.output_dim = output_dim
        self.backbone = backbone or SmallFaceCNN(output_dim)

    def forward(self, face_batch: torch.Tensor) -> torch.Tensor:
        """face_batch: (B, 3, 224, 224) float tensor in [0, 1]."""
        return self.backbone(face_batch)

    @staticmethod
    def missing_embedding(batch_size: int, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, settings.VISUAL_EMBED_DIM, device=device)
