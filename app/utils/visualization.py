"""
Module 21/27 support: shared plotting helpers for the dashboard and for
research-report generation (confusion_matrix.png etc).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import plotly.graph_objects as go


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: str | Path | None = None) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis_autorange="reversed",
    )
    if save_path:
        fig.write_image(str(save_path))
    return fig


def plot_emotion_timeline(timestamps: Sequence[float], emotions: Sequence[str], labels: List[str]) -> go.Figure:
    """Categorical scatter/line showing emotion at each utterance over time."""
    y_positions = [labels.index(e) for e in emotions]
    fig = go.Figure(
        data=go.Scatter(
            x=list(timestamps),
            y=y_positions,
            mode="lines+markers",
            text=emotions,
            hovertemplate="t=%{x:.1f}s<br>emotion=%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Emotion Timeline",
        xaxis_title="Time (s)",
        yaxis=dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
    )
    return fig


def plot_emotion_distribution(emotion_counts: dict[str, int]) -> go.Figure:
    fig = go.Figure(data=go.Bar(x=list(emotion_counts.keys()), y=list(emotion_counts.values())))
    fig.update_layout(title="Emotion Distribution", xaxis_title="Emotion", yaxis_title="Count")
    return fig


def plot_modality_attention(attn_weights: dict[str, float]) -> go.Figure:
    """Bar chart of per-modality attention weight for a single prediction (Module 17)."""
    fig = go.Figure(data=go.Bar(x=list(attn_weights.keys()), y=list(attn_weights.values())))
    fig.update_layout(
        title="Modality Contribution (model-derived, not causal)",
        yaxis_title="Attention Weight",
        yaxis_range=[0, 1],
    )
    return fig


def plot_experiment_comparison(results: dict[str, dict[str, float]], metric: str = "f1_macro") -> go.Figure:
    """Bar chart comparing experiments (Experiment A/B/C/D or ablation configs) on one metric."""
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    fig = go.Figure(data=go.Bar(x=names, y=values))
    fig.update_layout(title=f"Experiment Comparison ({metric})", yaxis_title=metric)
    return fig
