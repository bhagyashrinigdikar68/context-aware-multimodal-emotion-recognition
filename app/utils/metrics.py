"""
Module 18/19 support: evaluation metrics shared by training, evaluation, and
ablation scripts so every experiment reports numbers the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> ClassificationMetrics:
    """
    Computes standard classification metrics. Macro F1 is emphasized per
    Module 18, since conversational emotion datasets like MELD are heavily
    imbalanced (neutral dominates) and accuracy alone is misleading.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision_macro=float(precision_m),
        recall_macro=float(recall_m),
        f1_macro=float(f1_m),
        f1_weighted=float(f1_w),
    )


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels: List[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))


def full_classification_report(y_true: Sequence[int], y_pred: Sequence[int], labels: List[str]) -> str:
    return classification_report(y_true, y_pred, target_names=labels, labels=list(range(len(labels))), zero_division=0)
