"""
Shared train/eval loop so `train_text.py`, `train_multimodal.py`,
`train_context.py`, and `ablation.py` don't each reimplement the same
boilerplate. Keeping ONE engine also guarantees the research experiments in
Module 18 are compared fairly (same optimizer, same metric computation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import settings
from app.utils.logging import get_logger
from app.utils.metrics import ClassificationMetrics, compute_metrics
from app.utils.modality_mask import build_mask_tensor

logger = get_logger("training.engine", settings.LOG_DIR)


def run_epoch(model, dataloader: DataLoader, optimizer=None, device: str = "cpu") -> tuple[float, ClassificationMetrics]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()

    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(dataloader, desc="train" if is_train else "eval", leave=False):
            labels = batch["labels"].to(device)
            mask = build_mask_tensor(batch["availabilities"], device=device)

            output = model(batch["current"], history=batch["history"], modality_mask=mask)
            loss = criterion(output["logits"], labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            preds = output["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return total_loss / max(n_batches, 1), metrics


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 2e-5,
    device: str = "cpu",
    checkpoint_path: Optional[Path] = None,
    patience: int = 3,
) -> dict:
    """
    Trains with early stopping on validation Macro F1 (prioritized over
    accuracy per Module 18, since MELD is class-imbalanced).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1, best_state, epochs_without_improvement = -1.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_loader, None, device)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_f1m=%.4f | val_loss=%.4f val_f1m=%.4f val_acc=%.4f",
            epoch, epochs, train_loss, train_metrics.f1_macro, val_loss, val_metrics.f1_macro, val_metrics.accuracy,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.as_dict().items()}})

        if val_metrics.f1_macro > best_f1:
            best_f1 = val_metrics.f1_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": best_state, "val_f1_macro": best_f1}, checkpoint_path)
                logger.info("Saved new best checkpoint (val_f1_macro=%.4f) to %s", best_f1, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_f1_macro": best_f1, "history": history}
