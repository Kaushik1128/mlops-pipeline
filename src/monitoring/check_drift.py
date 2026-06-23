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
# Sample size per dataset — distribution comparison on a representative sample
# is statistically sound and far faster than using every row.
SAMPLE_SIZE = 10_000


def _parse_column(metric_name: str) -> str | None:
    m = re.search(r"column=([^,)]+)", metric_name)
    return m.group(1) if m else None


def _parse_threshold(metric_name: str) -> float | None:
    m = re.search(r"threshold=([\d.]+)", metric_name)
    return float(m.group(1)) if m else None


def check_drift(
    current_file: Path,
    reference_file: Path = REFERENCE_FILE,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
    sample_size: int = SAMPLE_SIZE,
    seed: int = 42,
) -> dict:
    """Run Evidently data-drift detection and return a signal dict.

    Returns a dict with keys: drift_detected, n_drifted, n_features,
    share_drifted, drifted_columns, report_html, current, reference.
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

    drift_detected = share_drifted >= drift_share_threshold

    signal = {
        "drift_detected": drift_detected,
        "n_drifted": n_drifted,
        "n_features": len(feature_cols),
        "share_drifted": round(share_drifted, 4),
        "drift_share_threshold": drift_share_threshold,
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        signal = check_drift(
            current_file=args.current,
            reference_file=args.reference,
            drift_share_threshold=args.drift_share,
        )
    except Exception:
        logger.exception("Drift check failed.")
        return 1

    verdict = "⚠️  DRIFT DETECTED" if signal["drift_detected"] else "✅ NO DRIFT"
    logger.info("%s — %d/%d features drifted (share=%.2f, threshold=%.2f)",
                verdict, signal["n_drifted"], signal["n_features"],
                signal["share_drifted"], signal["drift_share_threshold"])
    if signal["drifted_columns"]:
        logger.info("Drifted columns: %s", ", ".join(signal["drifted_columns"]))
    logger.info("HTML report: %s", signal["report_html"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
