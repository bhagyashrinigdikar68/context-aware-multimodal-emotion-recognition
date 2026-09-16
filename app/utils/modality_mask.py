"""
Module 15: Missing-modality handling helpers.

Centralizes the logic for building a modality-availability mask and for
substituting a well-defined zero embedding when a modality is missing, so
every module (fusion, context, pipeline) agrees on the same convention:

    mask[modality][i] == 1.0  -> modality i is available (real signal)
    mask[modality][i] == 0.0  -> modality i is missing (zero-filled embedding)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class ModalityAvailability:
    text: bool = True
    audio: bool = True
    visual: bool = True
    context: bool = True

    def as_dict(self) -> Dict[str, bool]:
        return {"text": self.text, "audio": self.audio, "visual": self.visual, "context": self.context}

    def summary(self) -> str:
        checks = {True: "✓", False: "✗"}
        return " ".join(f"{checks[v]} {k.capitalize()}" for k, v in self.as_dict().items())


def build_mask_tensor(availabilities: list[ModalityAvailability], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Convert a batch of per-sample ModalityAvailability into mask tensors per modality."""
    modalities = ("text", "audio", "visual", "context")
    out = {}
    for m in modalities:
        out[m] = torch.tensor([float(getattr(a, m)) for a in availabilities], device=device)
    return out
