"""
Module 3: Text encoder.

Wraps a pretrained transformer (default: roberta-base) to produce a
contextual embedding per utterance. This is a genuine transformer forward
pass -- not keyword matching -- and the model name is fully configurable via
`settings.TEXT_MODEL_NAME`, so swapping to BERT/DeBERTa/any HF encoder
requires no code change.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger("models.text_encoder", settings.LOG_DIR)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str | None = None, freeze_base: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        model_name = model_name or settings.TEXT_MODEL_NAME
        logger.info("Loading text encoder backbone '%s' ...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.output_dim = self.backbone.config.hidden_size

        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Return a (batch, hidden_size) embedding per utterance (mean-pooled)."""
        device = next(self.backbone.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        ).to(device)

        outputs = self.backbone(**encoded)
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        # mean-pool over real tokens only (mask out padding)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        embedding = summed / counts
        return embedding  # (B, H)

    @staticmethod
    def missing_embedding(batch_size: int, device: str = "cpu") -> torch.Tensor:
        """Zero-vector placeholder used when text is unavailable for an utterance."""
        return torch.zeros(batch_size, settings.TEXT_EMBED_DIM, device=device)
