"""Model evaluation utilities — pure functions, no MLflow/model dependencies.

Splits cleanly into two kinds of output:
  - compute_metrics(): a dict of scalar metrics for MLflow to log
  - confusion_matrix_figure(): a matplotlib Figure for MLflow to log as an artifact

Kept dependency-free (just numpy/sklearn/matplotlib) so it can be unit-tested
with toy arrays and reused by training (Phase 3), drift checks (Phase 5), and
retraining (Phase 6).

Key distinction baked into the API:
  - PR-AUC and ROC-AUC are computed from PROBABILITIES (threshold-independent).
  - precision / recall / f1 are computed from BINARY predictions at a chosen
    threshold (threshold-dependent). The threshold is an explicit argument so
    this dependency is never hidden.
"""
from __future__ import annotations

import matplotlib

# LEARN: Force the non-interactive "Agg" backend BEFORE importing pyplot.
# Training runs headless (no display, and later inside containers). Without
# this, matplotlib may try to open a GUI window and crash or hang.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
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
        y_proba: Predicted probability of the POSITIVE class, shape (n,),
            values in [0, 1]. NOT binary predictions.
        threshold: Cutoff applied to y_proba to produce binary predictions
            for the threshold-dependent metrics (precision/recall/f1).

    Returns:
        Dict of metric name -> value. Keys:
            pr_auc, roc_auc  (threshold-independent, from probabilities)
            precision, recall, f1  (threshold-dependent, at `threshold`)
    """
    # LEARN: Binarize once, here, so it's obvious that precision/recall/f1
    # all use the SAME threshold. y_pred is {0,1}; y_proba stays continuous.
    y_pred = (y_proba >= threshold).astype(int)

    return {
        # --- Threshold-INDEPENDENT: the headline metrics for model selection ---
        # LEARN: average_precision_score IS PR-AUC (area under precision-recall
        # curve). It takes probabilities, not binary preds. This is our
        # primary metric given the 0.17% positive rate.
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        # ROC-AUC reported as a sanity check / convention, not for selection.
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        # --- Threshold-DEPENDENT: what the model does at THIS cutoff ---
        # LEARN: zero_division=0 means "if we predicted zero positives, report
        # precision as 0 rather than raising a warning/NaN." Relevant early in
        # training when a model might predict all-negative.
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
    """Build a labelled 2x2 confusion-matrix figure for MLflow logging.

    Args:
        y_true: Ground-truth labels, shape (n,), values in {0, 1}.
        y_proba: Predicted probability of the positive class, shape (n,).
        threshold: Cutoff applied to y_proba to produce binary predictions.
        title: Figure title (e.g. include the model name).

    Returns:
        A matplotlib Figure. Caller is responsible for closing it
        (plt.close(fig)) after logging, to avoid leaking memory.
    """
    y_pred = (y_proba >= threshold).astype(int)
    # LEARN: labels=[0, 1] pins cell order so it's deterministic even if one
    # class is absent from y_pred (possible with a degenerate model).
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    # LEARN: imshow draws the matrix as a heatmap. The color scale will be
    # dominated by the huge TN cell — that's expected and fine; the per-cell
    # text annotations below are what humans actually read.
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Human-readable meaning of each quadrant, in fraud terms.
    cell_meaning = {
        (0, 0): "True Negative\n(legit ok)",
        (0, 1): "False Positive\n(false alarm)",
        (1, 0): "False Negative\n(missed fraud)",
        (1, 1): "True Positive\n(caught fraud)",
    }
    # LEARN: Choose text color per-cell for contrast — dark text on light
    # cells, white on dark. Threshold at half the max count.
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{cell_meaning[(i, j)]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_xticks([0, 1], labels=["Pred: legit", "Pred: fraud"])
    ax.set_yticks([0, 1], labels=["True: legit", "True: fraud"])
    ax.set_title(f"{title}\n(threshold = {threshold:.2f})")
    fig.tight_layout()
    return fig
