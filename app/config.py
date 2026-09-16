"""
Central configuration for the emotion-recognition system.

All modules read configuration from a single `Settings` instance so behaviour
(device, batch size, model names, emotion label set, etc.) can be changed in
one place -- via environment variables or a `.env` file -- without touching
code. Nothing here is hard-coded in a way that blocks extension: in
particular `EMOTION_LABELS` can be edited/extended without changing model
code, because every model reads `len(settings.EMOTION_LABELS)` for its output
dimension rather than a literal `7`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


@dataclass
class Settings:
    # ---- Emotion taxonomy (extensible: add/remove classes here only) ----
    EMOTION_LABELS: List[str] = field(
        default_factory=lambda: [
            "anger",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "neutral",
        ]
    )

    # ---- Device / performance ----
    DEVICE: str = field(default_factory=lambda: _resolve_device(os.getenv("DEVICE", "auto")))
    BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", 8)))
    MAX_SEQUENCE_LENGTH: int = field(default_factory=lambda: int(os.getenv("MAX_SEQUENCE_LENGTH", 128)))
    FRAME_RATE: int = field(default_factory=lambda: int(os.getenv("FRAME_RATE", 2)))
    CONTEXT_WINDOW: int = field(default_factory=lambda: int(os.getenv("CONTEXT_WINDOW", 8)))

    # ---- Model identifiers ----
    TEXT_MODEL_NAME: str = field(default_factory=lambda: os.getenv("TEXT_MODEL_NAME", "roberta-base"))
    WHISPER_MODEL_SIZE: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "base"))
    HUGGINGFACE_TOKEN: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", ""))

    # ---- Paths ----
    DATA_DIR: Path = field(default_factory=lambda: _env_path("DATA_DIR", "./dataset/meld"))
    MODEL_DIR: Path = field(default_factory=lambda: _env_path("MODEL_DIR", "./models"))
    LOG_DIR: Path = field(default_factory=lambda: _env_path("LOG_DIR", "./logs"))
    TMP_DIR: Path = field(default_factory=lambda: _env_path("TMP_DIR", "./tmp"))

    # ---- API ----
    API_HOST: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = field(default_factory=lambda: int(os.getenv("API_PORT", 8000)))

    # ---- Demo mode ----
    ALLOW_DEMO_MODE: bool = field(default_factory=lambda: _env_bool("ALLOW_DEMO_MODE", True))

    # ---- Fusion / context model dims (shared so all encoders agree) ----
    TEXT_EMBED_DIM: int = 768   # roberta-base hidden size
    AUDIO_EMBED_DIM: int = 128
    VISUAL_EMBED_DIM: int = 128
    SPEAKER_EMBED_DIM: int = 32
    FUSED_DIM: int = 256

    def __post_init__(self) -> None:
        for p in (self.MODEL_DIR, self.LOG_DIR, self.TMP_DIR):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def num_classes(self) -> int:
        return len(self.EMOTION_LABELS)

    def label_to_index(self, label: str) -> int:
        return self.EMOTION_LABELS.index(label.lower())

    def index_to_label(self, index: int) -> str:
        return self.EMOTION_LABELS[index]


settings = Settings()
