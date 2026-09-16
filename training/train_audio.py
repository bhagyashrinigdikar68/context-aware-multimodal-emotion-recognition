"""
Audio-only and Text+Audio configurations (used by Experiment B in Module 18
and by several rows of the ablation study in Module 19). Thin wrapper around
`training.train_multimodal.build_model_and_data` with modalities toggled.

Usage:
    # audio only
    python training/train_audio.py --train ... --val ... --mode audio_only
    # text + audio (Experiment B)
    python training/train_audio.py --train ... --val ... --mode text_audio
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from training.engine import train_model
from training.train_multimodal import build_model_and_data

MODES = {
    "audio_only": dict(use_text=False, use_audio=True, use_visual=False, use_context=False),
    "text_audio": dict(use_text=True, use_audio=True, use_visual=False, use_context=False),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--mode", choices=list(MODES.keys()), default="text_audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or Path(f"outputs/experiments/{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = settings.DEVICE

    modality_flags = MODES[args.mode]
    model, train_loader, val_loader = build_model_and_data(
        args.train, args.val, batch_size=args.batch_size, device=device, **modality_flags,
    )

    result = train_model(
        model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device,
        checkpoint_path=settings.MODEL_DIR / f"{args.mode}_best.pt",
    )

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.mode}] Best val Macro F1: {result['best_val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
