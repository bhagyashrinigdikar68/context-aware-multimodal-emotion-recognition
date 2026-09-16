"""
Isolates the effect of conversational context (Module 6 / Module 19):
trains Text+Context vs. Text-only using the exact same data and optimizer
settings, so the comparison directly answers "does context help?"

Usage:
    python training/train_context.py --train ... --val ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from training.engine import train_model
from training.train_multimodal import build_model_and_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    args = parser.parse_args()

    device = settings.DEVICE
    results = {}

    for name, use_context in [("text_only", False), ("text_plus_context", True)]:
        model, train_loader, val_loader = build_model_and_data(
            args.train, args.val,
            use_text=True, use_audio=False, use_visual=False, use_context=use_context,
            batch_size=args.batch_size, device=device,
        )
        result = train_model(
            model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device,
            checkpoint_path=settings.MODEL_DIR / f"{name}_context_ablation_best.pt",
        )
        results[name] = result["best_val_f1_macro"]
        print(f"[{name}] best val Macro F1: {result['best_val_f1_macro']:.4f}")

    out_dir = Path("outputs/experiments/context_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    delta = results["text_plus_context"] - results["text_only"]
    print(f"\nContext improved Macro F1 by {delta:+.4f} ({'helps' if delta > 0 else 'does not help'} on this data/run).")


if __name__ == "__main__":
    main()
