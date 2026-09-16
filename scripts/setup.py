"""
One-time environment setup / sanity check.

    python scripts/setup.py

Checks: Python/torch/CUDA availability, ffmpeg on PATH, required directories
exist, and reports whether the MELD dataset has been prepared. Does not
download anything automatically (dataset licensing requires the user to
accept terms directly from the source).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "✓" if ok else "✗"
    print(f"[{mark}] {name}{': ' + detail if detail else ''}")


def main():
    print("=== Context-Aware Multimodal Emotion Recognition -- Setup Check ===\n")

    check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])

    try:
        import torch

        check("PyTorch installed", True, torch.__version__)
        check("CUDA available", torch.cuda.is_available(), "using GPU" if torch.cuda.is_available() else "CPU-only mode")
    except ImportError:
        check("PyTorch installed", False, "run: pip install -r requirements.txt")

    check("ffmpeg on PATH", shutil.which("ffmpeg") is not None, "required for audio extraction")

    for name, path in [("Model dir", settings.MODEL_DIR), ("Log dir", settings.LOG_DIR), ("Tmp dir", settings.TMP_DIR)]:
        path.mkdir(parents=True, exist_ok=True)
        check(name, path.exists(), str(path))

    meld_raw = settings.DATA_DIR / "raw"
    meld_ok = meld_raw.exists() and any(meld_raw.iterdir()) if meld_raw.exists() else False
    check("MELD raw dataset present", meld_ok, str(meld_raw) if meld_ok else "see dataset/README.md")

    checkpoint = settings.MODEL_DIR / "multimodal_best.pt"
    check("Trained checkpoint present", checkpoint.exists(), str(checkpoint) if checkpoint.exists() else "will run in DEMO MODE")

    print(f"\nDevice that will be used: {settings.DEVICE}")
    print(f"Emotion classes: {settings.EMOTION_LABELS}")


if __name__ == "__main__":
    main()
