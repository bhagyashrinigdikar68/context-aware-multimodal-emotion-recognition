"""
Shared PyTorch Dataset for training/evaluation: reads the JSONL manifests
produced by `dataset/prepare_meld.py`, groups utterances by dialogue so a
configurable context window of preceding utterances can be attached to each
training example, and returns modality tensors with correct missing-modality
handling (an utterance with no audio file, e.g. extraction failed, is marked
unavailable rather than dropped).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from app.config import settings
from app.models.speaker import SpeakerEmbedding
from app.preprocessing import audio as audio_prep
from app.utils.modality_mask import ModalityAvailability


class ConversationEmotionDataset(Dataset):
    """
    Each __getitem__ returns a single utterance PLUS its preceding context
    window (both drawn from the same dialogue_id), matching what
    MultimodalEmotionModel.forward expects for `current` and `history`.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        context_window: int | None = None,
        load_audio_features: bool = True,
        load_faces: bool = False,  # face crops require a separate offline extraction step; see training/train_visual.py
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. Run dataset/prepare_meld.py first."
            )
        self.context_window = context_window or settings.CONTEXT_WINDOW
        self.load_audio_features = load_audio_features
        self.load_faces = load_faces

        self.records = [json.loads(line) for line in open(self.manifest_path)]
        # group by dialogue, preserving original order, for context lookups
        self.by_dialogue: dict[str, List[dict]] = {}
        for r in self.records:
            self.by_dialogue.setdefault(r["dialogue_id"], []).append(r)
        for dialogue in self.by_dialogue.values():
            dialogue.sort(key=lambda r: int(r["utterance_id"]))

        self._audio_cache: dict[str, Optional[torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _audio_feature_vector(self, record: dict) -> torch.Tensor:
        if not self.load_audio_features or not record.get("audio_path"):
            return torch.zeros(audio_prep.AudioSegmentFeatures.vector_dim())
        key = record["audio_path"]
        if key not in self._audio_cache:
            waveform = audio_prep.load_waveform(key)
            feats = audio_prep.extract_segment_features(waveform, start_sec=0.0, end_sec=None) if waveform is not None else audio_prep.AudioSegmentFeatures.missing()
            self._audio_cache[key] = torch.tensor(feats.to_vector())
        return self._audio_cache[key]

    def _build_utterance(self, record: dict) -> dict:
        audio_vec = self._audio_feature_vector(record)
        label = settings.label_to_index(record["emotion"])
        speaker_idx = SpeakerEmbedding.parse_speaker_label(record["speaker"])
        availability = ModalityAvailability(
            text=bool(record["text"].strip()),
            audio=bool(record.get("audio_path")),
            visual=False,  # face crops not wired into the base dataset; see train_visual.py for a visual-only loader
            context=True,
        )
        return {
            "text": record["text"],
            "audio_features": audio_vec,
            "speaker_idx": speaker_idx,
            "label": label,
            "availability": availability,
        }

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        dialogue = self.by_dialogue[record["dialogue_id"]]
        pos = next(i for i, r in enumerate(dialogue) if r["utterance_id"] == record["utterance_id"])

        start = max(0, pos - self.context_window)
        context_records = dialogue[start:pos]  # strictly preceding utterances only (causal)

        current = self._build_utterance(record)
        history = [self._build_utterance(r) for r in context_records]
        return {"current": current, "history": history}


def collate_conversation_batch(batch: List[dict], device: str = "cpu") -> dict:
    """
    Pads variable-length context windows (left-padding, oldest-first) and
    stacks everything into the tensor shapes MultimodalEmotionModel expects.
    """
    batch_size = len(batch)
    max_hist_len = max((len(b["history"]) for b in batch), default=0)

    current_texts = [b["current"]["text"] for b in batch]
    current_audio = torch.stack([b["current"]["audio_features"] for b in batch]).to(device)
    current_speaker = torch.tensor([b["current"]["speaker_idx"] for b in batch]).to(device)
    labels = torch.tensor([b["current"]["label"] for b in batch]).to(device)

    availabilities = [b["current"]["availability"] for b in batch]

    # history: pad on the left with zero vectors + a padding mask
    hist_audio = torch.zeros(batch_size, max(max_hist_len, 1), audio_prep.AudioSegmentFeatures.vector_dim())
    hist_speaker = torch.zeros(batch_size, max(max_hist_len, 1), dtype=torch.long)
    hist_texts = [["" for _ in range(max(max_hist_len, 1))] for _ in range(batch_size)]
    padding_mask = torch.ones(batch_size, max(max_hist_len, 1), dtype=torch.bool)  # True = PAD

    for i, b in enumerate(batch):
        h = b["history"]
        offset = max_hist_len - len(h)  # left-pad
        for j, step in enumerate(h):
            pos = offset + j
            hist_audio[i, pos] = step["audio_features"]
            hist_speaker[i, pos] = step["speaker_idx"]
            hist_texts[i][pos] = step["text"]
            padding_mask[i, pos] = False
        # also append current utterance as the final step of the "history" sequence
        # so ContextEncoder attends over history + current together.
    # append current step at the end of the sequence
    hist_audio = torch.cat([hist_audio, current_audio.unsqueeze(1)], dim=1)
    hist_speaker = torch.cat([hist_speaker, current_speaker.unsqueeze(1)], dim=1)
    for i in range(batch_size):
        hist_texts[i].append(current_texts[i])
    padding_mask = torch.cat([padding_mask, torch.zeros(batch_size, 1, dtype=torch.bool)], dim=1)

    history = {
        "texts": hist_texts,
        "audio_features": hist_audio.to(device),
        "speaker_indices": hist_speaker.to(device),
        "padding_mask": padding_mask.to(device),
    }
    current = {
        "texts": current_texts,
        "audio_features": current_audio,
        "face_images": None,
        "speaker_indices": current_speaker,
    }
    return {"current": current, "history": history, "labels": labels, "availabilities": availabilities}
