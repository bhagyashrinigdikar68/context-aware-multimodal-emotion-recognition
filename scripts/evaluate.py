"""
Module 27: Research report generation.

Loads a trained checkpoint, evaluates it on a held-out split, and writes:
    metrics.csv, confusion_matrix.png, classification_report.txt

Usage:
    python scripts/evaluate.py --checkpoint models/multimodal_best.pt \
                                --test dataset/meld/processed/test.jsonl \
                                --config text_audio_visual_ctx
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.config import settings  # noqa: E402
from app.models.multimodal_model import MultimodalEmotionModel  # noqa: E402
from app.utils.metrics import compute_confusion_matrix, compute_metrics, full_classification_report  # noqa: E402
from app.utils.modality_mask import build_mask_tensor  # noqa: E402
from app.utils.visualization import plot_confusion_matrix  # noqa: E402
from training.data import ConversationEmotionDataset, collate_conversation_batch  # noqa: E402
from training.ablation import CONFIGS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="text_audio_visual_ctx")
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train a model first (see training/) -- "
            "this script refuses to evaluate an untrained model as though it were a real result."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = settings.DEVICE

    flags = CONFIGS[args.config]
    model = MultimodalEmotionModel(**flags).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    context_window = settings.CONTEXT_WINDOW if flags["use_context"] else 0
    test_ds = ConversationEmotionDataset(args.test, context_window=context_window, load_audio_features=flags["use_audio"])
    collate = lambda b: collate_conversation_batch(b, device=device)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            mask = build_mask_tensor(batch["availabilities"], device=device)
            output = model(batch["current"], history=batch["history"], modality_mask=mask)
            preds = output["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    labels = settings.EMOTION_LABELS

    with open(args.out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics.as_dict().keys()))
        writer.writerow(list(metrics.as_dict().values()))

    with open(args.out_dir / "classification_report.txt", "w") as f:
        f.write(full_classification_report(all_labels, all_preds, labels))

    cm = compute_confusion_matrix(all_labels, all_preds, labels)
    fig = plot_confusion_matrix(cm, labels)
    try:
        fig.write_image(str(args.out_dir / "confusion_matrix.png"))
    except Exception as exc:
        print(f"(Could not write PNG -- install `kaleido` for static image export: {exc})")

    print("\n=== Evaluation Results ===")
    for k, v in metrics.as_dict().items():
        print(f"{k}: {v:.4f}")
    print(f"\nFull report + confusion matrix written to {args.out_dir}")


if __name__ == "__main__":
    main()
