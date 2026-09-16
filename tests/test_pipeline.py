"""
Tests for the modality-mask utility and light integration checks on the
multimodal model's missing-modality handling (Module 15). These avoid
requiring a real video/audio file or downloaded transformer weights by
constructing tensors directly, matching how `app/pipeline/inference.py`
assembles them.
"""
from __future__ import annotations

import torch

from app.config import settings
from app.utils.modality_mask import ModalityAvailability, build_mask_tensor


def test_modality_availability_summary_format():
    avail = ModalityAvailability(text=True, audio=False, visual=True, context=True)
    summary = avail.summary()
    assert "✓ Text" in summary
    assert "✗ Audio" in summary


def test_build_mask_tensor_matches_batch_order():
    availabilities = [
        ModalityAvailability(text=True, audio=True, visual=False, context=True),
        ModalityAvailability(text=False, audio=True, visual=True, context=True),
    ]
    mask = build_mask_tensor(availabilities)
    assert torch.equal(mask["text"], torch.tensor([1.0, 0.0]))
    assert torch.equal(mask["visual"], torch.tensor([0.0, 1.0]))
    assert mask["context"].shape == (2,)


def test_emotion_labels_are_extensible_without_code_changes():
    """
    Sanity check that model output dimensionality is derived from
    settings.EMOTION_LABELS, not hard-coded -- adding a class should not
    require touching classifier code (Module 4's extensibility requirement).
    """
    from app.models.emotion_classifier import EmotionClassifier

    original_labels = settings.EMOTION_LABELS
    try:
        settings.EMOTION_LABELS = original_labels + ["contempt"]
        clf = EmotionClassifier(num_classes=settings.num_classes)
        logits = clf(torch.randn(2, settings.FUSED_DIM))
        assert logits.shape == (2, len(original_labels) + 1)
    finally:
        settings.EMOTION_LABELS = original_labels
