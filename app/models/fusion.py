"""
Module 8: Multimodal fusion.

Implements an attention-based ("gated") fusion mechanism: each modality
embedding is projected into a shared space, and a small gating network
computes a per-modality, per-utterance importance weight (softmax over
available modalities). This lets the model learn that e.g. visual+audio
matter more than text for a particular utterance, rather than fixing equal
weights or simple concatenation.

Missing modalities are handled via an explicit mask (Module 15): a missing
modality contributes a zero vector AND is excluded from the softmax so it
cannot silently drag the fused representation toward zero.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import settings

MODALITIES = ("text", "audio", "visual", "context")


class AttentionFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = settings.TEXT_EMBED_DIM,
        audio_dim: int = settings.AUDIO_EMBED_DIM,
        visual_dim: int = settings.VISUAL_EMBED_DIM,
        context_dim: int = settings.FUSED_DIM,
        fused_dim: int = settings.FUSED_DIM,
    ):
        super().__init__()
        self.fused_dim = fused_dim
        dims = {"text": text_dim, "audio": audio_dim, "visual": visual_dim, "context": context_dim}

        # project every modality into the same shared space
        self.projections = nn.ModuleDict({m: nn.Linear(dims[m], fused_dim) for m in MODALITIES})

        # gating network: scores each modality's relevance given its own projected vector
        self.gate = nn.ModuleDict({m: nn.Linear(fused_dim, 1) for m in MODALITIES})

        self.output_norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        embeddings: dict of modality -> (B, dim) tensors (zero vectors allowed for missing).
        modality_mask: dict of modality -> (B,) bool/0-1 tensor, True/1 = available.

        Returns:
            fused: (B, fused_dim)
            attn_weights: dict modality -> (B,) tensor of attention weight per sample,
                          useful for explainability (Module 17). These are model-derived
                          relevance indicators, NOT causal explanations.
        """
        projected = {m: self.projections[m](embeddings[m]) for m in MODALITIES}
        raw_scores = {m: self.gate[m](projected[m]).squeeze(-1) for m in MODALITIES}  # (B,)

        # mask out unavailable modalities with -inf before softmax
        stacked_scores = torch.stack(
            [raw_scores[m].masked_fill(modality_mask[m] < 0.5, float("-inf")) for m in MODALITIES],
            dim=1,
        )  # (B, 4)

        # if ALL modalities are missing for a sample (degenerate case), fall back to uniform
        all_missing = torch.isinf(stacked_scores).all(dim=1)
        if all_missing.any():
            stacked_scores[all_missing] = 0.0

        weights = F.softmax(stacked_scores, dim=1)  # (B, 4)

        fused = torch.zeros_like(projected["text"])
        attn_weights = {}
        for i, m in enumerate(MODALITIES):
            w = weights[:, i].unsqueeze(-1)  # (B, 1)
            fused = fused + w * projected[m]
            attn_weights[m] = weights[:, i]

        fused = self.output_norm(fused)
        return fused, attn_weights
