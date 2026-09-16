"""
Experiment D (Module 18): full multimodal + conversational context model.

    Text + Audio + Visual + Context -> Attention Fusion -> Context Encoder -> Emotion

Also used as the general-purpose trainer: `--use_audio/--use_visual/--use_context`
flags let this same script run Experiments B and C too (Module 18), and the
ablation study (Module 19) reuses `build_model_and_data` directly.

Usage:
    python training/train_multimodal.py --train dataset/meld/processed/train.jsonl \
                                         --val dataset/meld/processed/dev.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from app.config import settings
from app.models.multimodal_model import MultimodalEmotionModel
from training.data import ConversationEmotionDataset, collate_conversation_batch
from training.engine import train_model


def build_model_and_data(
    train_path: Path,
    val_path: Path,
    use_text: bool = True,
    use_audio: bool = True,
    use_visual: bool = True,
    use_context: bool = True,
    batch_size: int = settings.BATCH_SIZE,
    device: str = settings.DEVICE,
):
    context_window = settings.CONTEXT_WINDOW if use_context else 0
    train_ds = ConversationEmotionDataset(train_path, context_window=context_window, load_audio_features=use_audio)
    val_ds = ConversationEmotionDataset(val_path, context_window=context_window, load_audio_features=use_audio)

    collate = lambda b: collate_conversation_batch(b, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = MultimodalEmotionModel(use_text=use_text, use_audio=use_audio, use_visual=use_visual, use_context=use_context)
    return model, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--use_audio", action="store_true", default=True)
    parser.add_argument("--use_visual", action="store_true", default=True)
    parser.add_argument("--use_context", action="store_true", default=True)
    parser.add_argument("--no_visual", dest="use_visual", action="store_false",
                         help="Disable visual modality (note: base MELD manifest doesn't wire faces in by default; see README limitations).")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/experiments/multimodal_context"))
    args = parser.parse_args()

    device = settings.DEVICE
    model, train_loader, val_loader = build_model_and_data(
        args.train, args.val,
        use_audio=args.use_audio, use_visual=args.use_visual, use_context=args.use_context,
        batch_size=args.batch_size, device=device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = train_model(
        model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device,
        checkpoint_path=settings.MODEL_DIR / "multimodal_best.pt",
    )

    with open(args.out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Best val Macro F1: {result['best_val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
