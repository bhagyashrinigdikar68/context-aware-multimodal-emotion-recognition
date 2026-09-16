"""
Wires together every component into the full Context-Aware Multimodal
Emotion Recognition model:

    text/audio/visual encoders -> per-utterance "content" embedding
                                -> ContextEncoder over the utterance window
                                -> AttentionFusion(text, audio, visual, context)
                                -> EmotionClassifier

Also exposes reduced configurations (text-only, +audio, +audio+visual) used
by the research experiments in `training/` (Module 18) and the ablation
study (Module 19), by simply toggling which modalities are fed in vs.
zeroed out -- the architecture itself doesn't change between experiments,
only which inputs are masked, which keeps the comparison fair.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from app.config import settings
from app.models.audio_encoder import AudioEncoder
from app.models.context_encoder import ContextEncoder
from app.models.emotion_classifier import EmotionClassifier
from app.models.fusion import AttentionFusion
from app.models.speaker import SpeakerEmbedding
from app.models.text_encoder import TextEncoder
from app.models.visual_encoder import VisualEncoder


class MultimodalEmotionModel(nn.Module):
    def __init__(
        self,
        use_text: bool = True,
        use_audio: bool = True,
        use_visual: bool = True,
        use_context: bool = True,
        freeze_text_backbone: bool = False,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_visual = use_visual
        self.use_context = use_context

        self.text_encoder = TextEncoder(freeze_base=freeze_text_backbone) if use_text else None
        self.audio_encoder = AudioEncoder() if use_audio else None
        self.visual_encoder = VisualEncoder() if use_visual else None
        self.speaker_embedding = SpeakerEmbedding()

        # "content" embedding fed into the context transformer: concat of
        # whichever of text/audio/visual are enabled, projected to a common size.
        content_dim = 0
        if use_text:
            content_dim += settings.TEXT_EMBED_DIM
        if use_audio:
            content_dim += settings.AUDIO_EMBED_DIM
        if use_visual:
            content_dim += settings.VISUAL_EMBED_DIM
        content_dim = max(content_dim, 1)
        self.content_proj = nn.Linear(content_dim, settings.FUSED_DIM)

        self.context_encoder = (
            ContextEncoder(input_dim=settings.FUSED_DIM, context_window=settings.CONTEXT_WINDOW)
            if use_context
            else None
        )

        self.fusion = AttentionFusion()
        self.classifier = EmotionClassifier()

    def encode_content(
        self,
        texts: Optional[list[str]],
        audio_features: Optional[torch.Tensor],
        face_images: Optional[torch.Tensor],
        batch_size: int,
        device: str,
    ) -> torch.Tensor:
        parts = []
        if self.use_text:
            parts.append(self.text_encoder(texts) if texts is not None else TextEncoder.missing_embedding(batch_size, device))
        if self.use_audio:
            parts.append(self.audio_encoder(audio_features) if audio_features is not None else AudioEncoder.missing_embedding(batch_size, device))
        if self.use_visual:
            parts.append(self.visual_encoder(face_images) if face_images is not None else VisualEncoder.missing_embedding(batch_size, device))
        if not parts:
            return torch.zeros(batch_size, 1, device=device)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        current: Dict,
        history: Optional[Dict] = None,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        `current` is a dict with keys among {texts, audio_features, face_images,
        speaker_indices} for the utterance being classified.

        `history` (optional) holds the same keys but with an extra leading
        time dimension for previous utterances in the context window, used
        only when `use_context=True`.

        `modality_mask` maps modality name -> (B,) tensor, 1=available.
        Defaults to all-available if not provided.
        """
        device = next(self.parameters()).device
        batch_size = len(current.get("texts") or current.get("speaker_indices"))

        if modality_mask is None:
            ones = torch.ones(batch_size, device=device)
            modality_mask = {"text": ones, "audio": ones, "visual": ones, "context": ones}

        # ---- per-utterance modality embeddings (for the FINAL fusion step) ----
        text_emb = self.text_encoder(current["texts"]) if self.use_text and current.get("texts") is not None else TextEncoder.missing_embedding(batch_size, device)
        audio_emb = self.audio_encoder(current["audio_features"]) if self.use_audio and current.get("audio_features") is not None else AudioEncoder.missing_embedding(batch_size, device)
        visual_emb = self.visual_encoder(current["face_images"]) if self.use_visual and current.get("face_images") is not None else VisualEncoder.missing_embedding(batch_size, device)

        # ---- context embedding from the utterance history ----
        if self.use_context and history is not None:
            content_seq = []
            T = history["speaker_indices"].shape[1]
            for t in range(T):
                step_texts = history.get("texts")
                step_texts_t = [seq[t] for seq in step_texts] if step_texts is not None else None
                step_audio_t = history["audio_features"][:, t] if history.get("audio_features") is not None else None
                step_face_t = history["face_images"][:, t] if history.get("face_images") is not None else None
                content_seq.append(self.encode_content(step_texts_t, step_audio_t, step_face_t, batch_size, device))
            content_seq = torch.stack(content_seq, dim=1)  # (B, T, content_dim)
            content_seq = self.content_proj(content_seq)

            speaker_seq = self.speaker_embedding(history["speaker_indices"])  # (B, T, speaker_dim)
            padding_mask = history.get("padding_mask")  # (B, T) True=pad

            context_emb = self.context_encoder(content_seq, speaker_seq, padding_mask)
        else:
            context_emb = ContextEncoder.missing_embedding(batch_size, device)
            modality_mask = {**modality_mask, "context": torch.zeros(batch_size, device=device)}

        embeddings = {"text": text_emb, "audio": audio_emb, "visual": visual_emb, "context": context_emb}
        fused, attn_weights = self.fusion(embeddings, modality_mask)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "fused": fused,
            "attention_weights": attn_weights,
            "modality_mask": modality_mask,
        }
