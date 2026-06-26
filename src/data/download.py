"""Download the Credit Card Fraud Detection dataset to data/raw/.

Data source:
    OpenML dataset id=1597 — a mirror of the Worldline/ULB credit card fraud
    dataset originally published on Kaggle
    (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

    The dataset contains 284,807 European credit card transactions from
    September 2013, with 492 fraud cases (0.172%). Features V1-V28 are the
    output of a PCA transformation applied by the data publishers to
    anonymize the original features; Amount is kept as-is. Target column
    `Class` is 1 for fraud, 0 for legitimate.

Usage:
    python -m src.data.download
    python -m src.data.download --force   # ignore the cached file
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "creditcard.parquet"

# Pin the numeric ID rather than the dataset name — names can be reused, IDs cannot.
OPENML_DATA_ID = 1597


def download_dataset(force: bool = False) -> Path:
    """Fetch the credit-card fraud dataset from OpenML and save as Parquet.

    Args:
        force: If True, re-download even if the cached Parquet file exists.

    Returns:
        Path to the Parquet file that was downloaded or already present.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists() and not force:
        logger.info(
            "Cached dataset already at %s (%.1f MB). Skipping download. "
            "Use --force to re-download.",
            OUTPUT_FILE,
            OUTPUT_FILE.stat().st_size / 1024 / 1024,
        )
        return OUTPUT_FILE

    logger.info("Fetching dataset from OpenML (data_id=%d)...", OPENML_DATA_ID)
    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    df: pd.DataFrame = bunch.frame

    # OpenML may return the target as a string/category; coerce to a clean int.
    df["Class"] = df["Class"].astype(int)

    n_rows = len(df)
    n_fraud = int(df["Class"].sum())
    logger.info(
        "Fetched %d rows, %d columns. Fraud cases: %d (%.4f%%).",
        n_rows, df.shape[1], n_fraud, 100.0 * n_fraud / n_rows,
    )

    logger.info("Writing Parquet to %s ...", OUTPUT_FILE)
    df.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy", index=False)
    logger.info("Done. File size: %.1f MB", OUTPUT_FILE.stat().st_size / 1024 / 1024)
    return OUTPUT_FILE


def main() -> int:
    """CLI entry point. Returns shell exit code (0 on success)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached file exists.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        download_dataset(force=args.force)
    except Exception:
        logger.exception("Download failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
