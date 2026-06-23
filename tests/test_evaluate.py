"""Tests for src/models/evaluate.py — pure metric functions."""
from __future__ import annotations

import numpy as np

from src.models.evaluate import compute_metrics, confusion_matrix_figure


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_metrics(y_true, y_proba)
    assert set(m) == {"pr_auc", "roc_auc", "precision", "recall", "f1"}
    assert all(isinstance(v, float) for v in m.values())


def test_perfect_separation_scores_one():
    # All positives rank strictly above all negatives -> perfect ranking.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    m = compute_metrics(y_true, y_proba, threshold=0.5)
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_threshold_changes_recall_not_ranking():
    # Same ranking, but a high threshold misses a true positive scoring 0.6.
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.95])
    low = compute_metrics(y_true, y_proba, threshold=0.5)
    high = compute_metrics(y_true, y_proba, threshold=0.9)
    # Ranking-based metrics are threshold-independent.
    assert low["pr_auc"] == high["pr_auc"]
    assert low["roc_auc"] == high["roc_auc"]
    # Recall drops when the threshold rises past a positive's score.
    assert high["recall"] < low["recall"]


def test_all_negative_predictions_no_crash():
    # Every probability below threshold -> zero positives predicted.
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.05, 0.3])
    m = compute_metrics(y_true, y_proba, threshold=0.9)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_confusion_matrix_figure_returns_figure():
    import matplotlib.figure

    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])
    fig = confusion_matrix_figure(y_true, y_proba, threshold=0.5, title="t")
    assert isinstance(fig, matplotlib.figure.Figure)
