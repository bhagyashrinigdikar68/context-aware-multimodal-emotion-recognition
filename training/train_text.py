"""
Experiment A (Module 18): Text-only baseline.

    Text -> Text Encoder -> Emotion Classifier

Trains and evaluates the text-only configuration and writes results to
`outputs/experiments/text_only/`.

Usage:
    python training/train_text.py --train dataset/meld/processed/train.jsonl \
                                   --val dataset/meld/processed/dev.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.config import settings
from app.models.multimodal_model import MultimodalEmotionModel
from training.data import ConversationEmotionDataset, collate_conversation_batch
from training.engine import train_model


def build_dataloaders(train_path: Path, val_path: Path, batch_size: int, device: str):
    train_ds = ConversationEmotionDataset(train_path, context_window=0)  # no context for the text-only baseline
    val_ds = ConversationEmotionDataset(val_path, context_window=0)

    collate = lambda b: collate_conversation_batch(b, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/experiments/text_only"))
    args = parser.parse_args()

    device = settings.DEVICE
    train_loader, val_loader = build_dataloaders(args.train, args.val, args.batch_size, device)

    model = MultimodalEmotionModel(use_text=True, use_audio=False, use_visual=False, use_context=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = train_model(
        model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device,
        checkpoint_path=settings.MODEL_DIR / "text_only_best.pt",
    )

    with open(args.out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Best val Macro F1: {result['best_val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
