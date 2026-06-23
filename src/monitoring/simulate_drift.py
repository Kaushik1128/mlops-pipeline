"""Manufacture a drifted dataset to validate the drift detector.

Takes the processed test set (a stand-in for a fresh batch of production
transactions) and applies controlled distribution shifts to a subset of the
model's important features — simulating a world where transaction behavior has
changed. The target column is left untouched: this is DATA drift (input
distribution), not label/target drift.

Usage:
    python -m src.monitoring.simulate_drift
    python -m src.monitoring.simulate_drift --severity 2.0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "data" / "processed" / "test.parquet"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "test_drifted.parquet"
TARGET_COLUMN = "Class"

# LEARN: We drift the model's most fraud-informative features (from the EDA:
# V14, V17, V12, V10) plus log_amount (simulating inflated transaction sizes).
# Drifting features the model actually relies on is what makes drift dangerous
# — and clearly detectable.
DRIFT_FEATURES = ["V14", "V17", "V12", "V10", "log_amount"]


def simulate_drift(
    in_file: Path = IN_FILE,
    out_file: Path = OUT_FILE,
    features: list[str] | None = None,
    severity: float = 1.5,
    scale: float = 1.2,
) -> Path:
    """Apply a controlled distribution shift to selected features.

    Args:
        in_file: Source parquet (the "production batch" before drift).
        out_file: Where to write the drifted dataset.
        features: Columns to perturb (defaults to DRIFT_FEATURES).
        severity: Mean shift in units of each feature's std dev. Higher = more
            drift. 0 = no mean shift.
        scale: Multiplier on each feature's spread (1.0 = unchanged).

    Returns:
        Path to the drifted parquet.
    """
    if not in_file.exists():
        raise FileNotFoundError(
            f"Source not found at {in_file}. Run `python -m src.data.preprocess` first."
        )
    features = features or DRIFT_FEATURES

    df = pd.read_parquet(in_file)
    drifted = df.copy()

    for col in features:
        if col not in drifted.columns:
            logger.warning("Skipping unknown column: %s", col)
            continue
        std = df[col].std()
        # LEARN: new = value*scale + severity*std. The *scale term widens the
        # distribution; the +severity*std term shifts its center. Both move the
        # distribution away from the reference so Wasserstein distance grows.
        drifted[col] = drifted[col] * scale + severity * std
        logger.info(
            "Drifted %-10s | mean %.3f -> %.3f (shift=%.1fσ, scale=%.2f)",
            col, df[col].mean(), drifted[col].mean(), severity, scale,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    drifted.to_parquet(out_file, engine="pyarrow", compression="snappy", index=False)
    logger.info("Wrote drifted dataset (%d rows) to %s", len(drifted), out_file)
    return out_file


def main() -> int:
    """CLI entry point."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--severity", type=float, default=1.5,
                        help="Mean shift in std-devs (default 1.5).")
    parser.add_argument("--scale", type=float, default=1.2,
                        help="Spread multiplier (default 1.2).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        simulate_drift(severity=args.severity, scale=args.scale)
    except Exception:
        logger.exception("Drift simulation failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
