"""Model evaluation utilities — pure functions, no MLflow/model dependencies.

  - compute_metrics(): a dict of scalar metrics for MLflow to log
  - confusion_matrix_figure(): a matplotlib Figure to log as an artifact

Dependency-free (numpy/sklearn/matplotlib) so it can be unit-tested with toy
arrays and reused by training, drift checks, and retraining.

PR-AUC and ROC-AUC are computed from probabilities (threshold-independent);
precision / recall / f1 are computed from binary predictions at an explicit
threshold (threshold-dependent).
"""
from __future__ import annotations

import matplotlib

# Force the non-interactive Agg backend before importing pyplot, so plotting
# works headless (in containers / CI) without trying to open a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute classification metrics for an imbalanced binary problem.

    Args:
        y_true: Ground-truth labels, shape (n,), values in {0, 1}.
        y_proba: Predicted probability of the positive class, shape (n,),
            values in [0, 1]. Not binary predictions.
        threshold: Cutoff applied to y_proba for the threshold-dependent
            metrics (precision/recall/f1).

    Returns:
        Dict with keys: pr_auc, roc_auc (from probabilities); precision,
        recall, f1 (at `threshold`).
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        # average_precision_score is PR-AUC — the primary metric given the
        # 0.17% positive rate. Takes probabilities, not binary predictions.
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        # zero_division=0: report 0 (not a warning/NaN) if nothing is predicted positive.
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_matrix_figure(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Build a labelled 2x2 confusion-matrix figure.

    The caller is responsible for closing the returned figure
    (`plt.close(fig)`) after use to avoid leaking memory.
    """
    y_pred = (y_proba >= threshold).astype(int)
    # labels=[0, 1] pins cell order even if a class is absent from y_pred.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cell_meaning = {
        (0, 0): "True Negative\n(legit ok)",
        (0, 1): "False Positive\n(false alarm)",
        (1, 0): "False Negative\n(missed fraud)",
        (1, 1): "True Positive\n(caught fraud)",
    }
    thresh = cm.max() / 2.0  # text colour switch for contrast
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}\n{cell_meaning[(i, j)]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=9,
            )

    ax.set_xticks([0, 1], labels=["Pred: legit", "Pred: fraud"])
    ax.set_yticks([0, 1], labels=["True: legit", "True: fraud"])
    ax.set_title(f"{title}\n(threshold = {threshold:.2f})")
    fig.tight_layout()
    return fig
