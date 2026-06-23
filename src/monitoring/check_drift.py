"""Detect data drift between a reference and a current dataset using Evidently.

Compares the feature distributions of a reference dataset (what the model was
trained on) against a current batch. Produces:
  - a human-readable HTML report (reports/drift/<name>.html)
  - a machine-readable JSON signal (reports/drift/<name>_signal.json)
  - a clear stdout summary + return dict

The JSON signal (drift_detected, share of drifted columns, which columns) is
what Phase 6's auto-retraining flow will consume to decide whether to retrain.

Usage:
    # Drifted batch (should detect drift):
    python -m src.monitoring.check_drift --current data/processed/test_drifted.parquet
    # Control batch from the same distribution (should NOT detect drift):
    python -m src.monitoring.check_drift --current data/processed/test.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILE = PROJECT_ROOT / "data" / "processed" / "train.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports" / "drift"
TARGET_COLUMN = "Class"

# LEARN: Dataset-level drift fires when the SHARE of drifted columns crosses
# this. Evidently's default is 0.5 ("more than half the features"), but that's
# too blunt for fraud: drifting even a few of the model's MOST IMPORTANT
# features is dangerous. We use 0.1 — react when ~10%+ of features shift. This
# cleanly separates our control (0% drift) from the drifted batch (17%).
DRIFT_SHARE_THRESHOLD = 0.1
# Importance-weighted share that triggers drift. Because important-feature
# drift concentrates the weighted score, this sits higher than the plain
# column-share threshold.
WEIGHTED_DRIFT_THRESHOLD = 0.2
# Sample size per dataset — distribution comparison on a representative sample
# is statistically sound and far faster than using every row.
SAMPLE_SIZE = 10_000


def _parse_column(metric_name: str) -> str | None:
    m = re.search(r"column=([^,)]+)", metric_name)
    return m.group(1) if m else None


def _parse_threshold(metric_name: str) -> float | None:
    m = re.search(r"threshold=([\d.]+)", metric_name)
    return float(m.group(1)) if m else None


def load_staging_importances(
    tracking_uri: str = "http://localhost:5000",
    model_name: str = "fraud-classifier",
    alias: str = "staging",
) -> dict[str, float]:
    """Load the @staging model's feature importances, normalized to sum to 1.

    LEARN: This is the drift analogue of scale_pos_weight. Instead of treating
    all 29 features equally, we weight each feature's drift contribution by how
    much the model actually relies on it — so drift in V14 (≈37% of the model)
    counts far more than drift in a feature the model barely uses.

    Production note: you'd snapshot these importances WITH the model version so
    the weights can't silently fall out of sync with the served model. Here we
    read them live from @staging for simplicity.
    """
    import mlflow
    import numpy as np

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):       # tree models (XGBoost, RF)
        raw = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):                     # linear models
        raw = np.abs(clf.coef_).ravel()
    else:
        return {}
    # Feature order: prefer the names the model was fit with, else the known order.
    cols = list(getattr(clf, "feature_names_in_", [])) or (
        [f"V{i}" for i in range(1, 29)] + ["log_amount"]
    )
    total = float(raw.sum()) or 1.0
    return {c: float(w) / total for c, w in zip(cols, raw)}


def check_drift(
    current_file: Path,
    reference_file: Path = REFERENCE_FILE,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
    sample_size: int = SAMPLE_SIZE,
    seed: int = 42,
    weights: dict[str, float] | None = None,
    weighted_threshold: float = WEIGHTED_DRIFT_THRESHOLD,
) -> dict:
    """Run Evidently data-drift detection and return a signal dict.

    If `weights` (feature -> importance) is given, the drift decision is based
    on the IMPORTANCE-WEIGHTED share of drifted features rather than a plain
    column count — so drift in features the model relies on dominates.

    Returns a dict with keys including: drift_detected, decision_basis,
    n_drifted, share_drifted, weighted_share, drifted_columns.
    """
    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset

    if not current_file.exists():
        raise FileNotFoundError(f"Current dataset not found: {current_file}")
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference dataset not found: {reference_file}")

    # LEARN: Drop the target — DATA drift compares INPUT feature distributions,
    # not the label. Both datasets must share the same feature columns.
    ref = pd.read_parquet(reference_file).drop(columns=[TARGET_COLUMN], errors="ignore")
    cur = pd.read_parquet(current_file).drop(columns=[TARGET_COLUMN], errors="ignore")

    feature_cols = [c for c in ref.columns if c in cur.columns]
    ref = ref[feature_cols].sample(min(sample_size, len(ref)), random_state=seed)
    cur = cur[feature_cols].sample(min(sample_size, len(cur)), random_state=seed)
    logger.info("Comparing %d features | ref=%d rows, cur=%d rows",
                len(feature_cols), len(ref), len(cur))

    data_def = DataDefinition(numerical_columns=feature_cols)
    ref_ds = Dataset.from_pandas(ref, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur, data_definition=data_def)

    # LEARN: Pass our project threshold INTO the preset so Evidently's own
    # dataset-drift verdict (shown in the HTML report header) matches the
    # signal we emit — otherwise the report uses Evidently's blunt 0.5 default
    # and confusingly says "NOT detected" while our pipeline says "detected".
    report = Report([DataDriftPreset(drift_share=drift_share_threshold)])
    run = report.run(cur_ds, ref_ds)  # (current, reference)

    # --- Save the human-readable HTML report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_name = current_file.stem  # e.g. "test_drifted"
    html_path = REPORTS_DIR / f"{report_name}.html"
    run.save_html(str(html_path))

    # --- Extract the machine-readable signal ---
    result = run.dict()
    metrics = result["metrics"]
    # Dataset-level summary: count + share of drifted columns.
    summary = next(m for m in metrics if m["metric_name"].startswith("DriftedColumnsCount"))
    n_drifted = int(summary["value"]["count"])
    share_drifted = float(summary["value"]["share"])

    # Per-column: a column drifted if its ValueDrift score exceeds its threshold.
    drifted_columns = []
    for m in metrics:
        name = m["metric_name"]
        if name.startswith("ValueDrift"):
            col, thr = _parse_column(name), _parse_threshold(name)
            score = float(m["value"])
            if col is not None and thr is not None and score > thr:
                drifted_columns.append(col)

    # --- Decide drift: importance-weighted if weights given, else plain share ---
    if weights:
        # LEARN: weighted_share = (importance carried by drifted features) /
        # (total importance). Drifting V14 alone (~37%) can exceed the threshold
        # by itself; drifting many trivial features barely moves it.
        total_w = sum(weights.values()) or 1.0
        weighted_share = sum(weights.get(c, 0.0) for c in drifted_columns) / total_w
        drift_detected = weighted_share >= weighted_threshold
        decision_basis = "importance_weighted"
    else:
        weighted_share = None
        drift_detected = share_drifted >= drift_share_threshold
        decision_basis = "column_share"

    signal = {
        "drift_detected": drift_detected,
        "decision_basis": decision_basis,
        "n_drifted": n_drifted,
        "n_features": len(feature_cols),
        "share_drifted": round(share_drifted, 4),
        "drift_share_threshold": drift_share_threshold,
        "weighted_share": round(weighted_share, 4) if weighted_share is not None else None,
        "weighted_threshold": weighted_threshold if weights else None,
        "drifted_columns": sorted(drifted_columns),
        "current": str(current_file),
        "reference": str(reference_file),
        "report_html": str(html_path),
    }

    # --- Persist the JSON signal (Phase 6 will read this) ---
    json_path = REPORTS_DIR / f"{report_name}_signal.json"
    json_path.write_text(json.dumps(signal, indent=2))
    return signal


def main() -> int:
    """CLI entry point."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--current", type=Path,
                        default=PROJECT_ROOT / "data" / "processed" / "test_drifted.parquet",
                        help="Dataset to check for drift.")
    parser.add_argument("--reference", type=Path, default=REFERENCE_FILE,
                        help="Reference (training) dataset.")
    parser.add_argument("--drift-share", type=float, default=DRIFT_SHARE_THRESHOLD,
                        help="Share of drifted columns that triggers dataset drift.")
    parser.add_argument("--weighted", action="store_true",
                        help="Weight drift by @staging feature importances "
                             "(the scale_pos_weight analogue for drift).")
    parser.add_argument("--weighted-threshold", type=float, default=WEIGHTED_DRIFT_THRESHOLD,
                        help="Importance-weighted share that triggers drift.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        weights = None
        if args.weighted:
            weights = load_staging_importances()
            logger.info("Loaded importance weights for %d features from @staging", len(weights))
        signal = check_drift(
            current_file=args.current,
            reference_file=args.reference,
            drift_share_threshold=args.drift_share,
            weights=weights,
            weighted_threshold=args.weighted_threshold,
        )
    except Exception:
        logger.exception("Drift check failed.")
        return 1

    verdict = "⚠️  DRIFT DETECTED" if signal["drift_detected"] else "✅ NO DRIFT"
    if signal["decision_basis"] == "importance_weighted":
        logger.info("%s [importance-weighted] — weighted_share=%.3f (threshold=%.2f) | "
                    "%d/%d columns drifted",
                    verdict, signal["weighted_share"], signal["weighted_threshold"],
                    signal["n_drifted"], signal["n_features"])
    else:
        logger.info("%s [column-share] — %d/%d features drifted (share=%.2f, threshold=%.2f)",
                    verdict, signal["n_drifted"], signal["n_features"],
                    signal["share_drifted"], signal["drift_share_threshold"])
    if signal["drifted_columns"]:
        logger.info("Drifted columns: %s", ", ".join(signal["drifted_columns"]))
    logger.info("HTML report: %s", signal["report_html"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
