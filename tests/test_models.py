"""
Unit tests for the model components. Uses tiny dummy tensors so tests run on
CPU in seconds without downloading any pretrained weights (the text encoder,
which does require a HF download, is tested separately and skipped when
offline -- see test_text_encoder_shapes).
"""
from __future__ import annotations

import torch

from app.config import settings
from app.models.audio_encoder import AudioEncoder
from app.models.context_encoder import ContextEncoder
from app.models.emotion_classifier import EmotionClassifier
from app.models.fusion import MODALITIES, AttentionFusion
from app.models.speaker import SpeakerEmbedding
from app.models.visual_encoder import VisualEncoder
from app.preprocessing.audio import AudioSegmentFeatures


def test_audio_encoder_output_shape():
    enc = AudioEncoder()
    x = torch.randn(4, AudioSegmentFeatures.vector_dim())
    out = enc(x)
    assert out.shape == (4, settings.AUDIO_EMBED_DIM)


def test_visual_encoder_output_shape():
    enc = VisualEncoder()
    x = torch.rand(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, settings.VISUAL_EMBED_DIM)


def test_speaker_embedding_clamps_overflow_indices():
    emb = SpeakerEmbedding(max_speakers=4)
    idx = torch.tensor([0, 4, 999])  # 999 should clamp to the unknown slot without erroring
    out = emb(idx)
    assert out.shape == (3, emb.output_dim)


def test_speaker_label_parsing():
    assert SpeakerEmbedding.parse_speaker_label("Speaker 1") == 0
    assert SpeakerEmbedding.parse_speaker_label("Speaker 3") == 2
    assert SpeakerEmbedding.parse_speaker_label("garbage") == 0


def test_context_encoder_output_shape():
    batch, seq_len, input_dim = 3, 5, settings.FUSED_DIM
    enc = ContextEncoder(input_dim=input_dim)
    utterances = torch.randn(batch, seq_len, input_dim)
    speakers = torch.randn(batch, seq_len, settings.SPEAKER_EMBED_DIM)
    out = enc(utterances, speakers)
    assert out.shape == (batch, enc.output_dim)


def test_attention_fusion_weights_sum_to_one_and_respect_mask():
    fusion = AttentionFusion()
    batch = 2
    embeddings = {
        "text": torch.randn(batch, settings.TEXT_EMBED_DIM),
        "audio": torch.randn(batch, settings.AUDIO_EMBED_DIM),
        "visual": torch.randn(batch, settings.VISUAL_EMBED_DIM),
        "context": torch.randn(batch, settings.FUSED_DIM),
    }
    # sample 0: all modalities available; sample 1: visual missing
    mask = {
        "text": torch.tensor([1.0, 1.0]),
        "audio": torch.tensor([1.0, 1.0]),
        "visual": torch.tensor([1.0, 0.0]),
        "context": torch.tensor([1.0, 1.0]),
    }
    fused, attn = fusion(embeddings, mask)
    assert fused.shape == (batch, settings.FUSED_DIM)

    total_weight = sum(attn[m] for m in MODALITIES)
    assert torch.allclose(total_weight, torch.ones(batch), atol=1e-5)
    # missing modality should get exactly zero weight for that sample
    assert attn["visual"][1].item() == 0.0


def test_emotion_classifier_predict_returns_valid_distribution():
    clf = EmotionClassifier()
    logits = torch.randn(5, settings.num_classes)
    pred_idx, confidence, probs = EmotionClassifier.predict(logits)
    assert pred_idx.shape == (5,)
    assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)
