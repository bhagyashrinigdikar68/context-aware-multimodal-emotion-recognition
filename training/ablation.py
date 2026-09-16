"""
Module 19: Ablation study.

Trains and evaluates every modality combination requested:

    1. Text only
    2. Audio only
    3. Visual only              (requires pre-extracted faces, see train_visual.py)
    4. Text + Audio
    5. Text + Visual            (requires pre-extracted faces)
    6. Audio + Visual           (requires pre-extracted faces)
    7. Text + Audio + Visual    (requires pre-extracted faces)
    8. Text + Audio + Visual + Context (requires pre-extracted faces)

Configs that need face crops are skipped with a clear message (not silently
faked) if `dataset/meld/processed/faces/` hasn't been populated via
`train_visual.py --extract_faces` first.

Writes `ablation_results.csv` and a bar-chart comparison PNG so results feed
directly into the README experiments table (Module 27).

Usage:
    python training/ablation.py --train dataset/meld/processed/train.jsonl \
                                 --val dataset/meld/processed/dev.jsonl
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from app.config import settings
from app.utils.logging import get_logger
from app.utils.visualization import plot_experiment_comparison
from training.engine import train_model
from training.train_multimodal import build_model_and_data

logger = get_logger("training.ablation", settings.LOG_DIR)

CONFIGS = {
    "text_only":            dict(use_text=True,  use_audio=False, use_visual=False, use_context=False),
    "audio_only":           dict(use_text=False, use_audio=True,  use_visual=False, use_context=False),
    "visual_only":          dict(use_text=False, use_audio=False, use_visual=True,  use_context=False),
    "text_audio":           dict(use_text=True,  use_audio=True,  use_visual=False, use_context=False),
    "text_visual":          dict(use_text=True,  use_audio=False, use_visual=True,  use_context=False),
    "audio_visual":         dict(use_text=False, use_audio=True,  use_visual=True,  use_context=False),
    "text_audio_visual":    dict(use_text=True,  use_audio=True,  use_visual=True,  use_context=False),
    "text_audio_visual_ctx": dict(use_text=True, use_audio=True,  use_visual=True,  use_context=True),
}

REQUIRES_FACES = {"visual_only", "text_visual", "audio_visual", "text_audio_visual", "text_audio_visual_ctx"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--faces_dir", type=Path, default=Path("dataset/meld/processed/faces"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ablation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    faces_available = (args.faces_dir / "train").exists() and (args.faces_dir / "dev").exists()

    all_results = {}
    for name, flags in CONFIGS.items():
        if name in REQUIRES_FACES and not faces_available:
            logger.warning(
                "Skipping '%s': requires pre-extracted face crops at %s. "
                "Run: python training/train_visual.py --extract_faces --train %s --val %s",
                name, args.faces_dir, args.train, args.val,
            )
            continue

        logger.info("=== Ablation config: %s ===", name)
        model, train_loader, val_loader = build_model_and_data(
            args.train, args.val, batch_size=args.batch_size, device=settings.DEVICE, **flags,
        )
        result = train_model(
            model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=settings.DEVICE,
            checkpoint_path=settings.MODEL_DIR / f"ablation_{name}_best.pt",
        )
        # also compute full metric set on val for the final epoch's best state
        from training.engine import run_epoch
        _, val_metrics = run_epoch(model, val_loader, None, settings.DEVICE)
        all_results[name] = val_metrics.as_dict()

    # write CSV
    csv_path = args.out_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"])
        for name, m in all_results.items():
            writer.writerow([name, m["accuracy"], m["precision_macro"], m["recall_macro"], m["f1_macro"], m["f1_weighted"]])
    logger.info("Wrote %s", csv_path)

    if all_results:
        fig = plot_experiment_comparison(all_results, metric="f1_macro")
        try:
            fig.write_image(str(args.out_dir / "ablation_f1_macro.png"))
        except Exception as exc:
            logger.warning("Could not write PNG (kaleido not installed?): %s", exc)

    print("\n=== Ablation Study Results (Macro F1) ===")
    for name, m in sorted(all_results.items(), key=lambda kv: -kv[1]["f1_macro"]):
        print(f"{name:25s} f1_macro={m['f1_macro']:.4f}  acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
