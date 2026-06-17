"""Download the Credit Card Fraud Detection dataset to data/raw/.

Data source:
    OpenML dataset id=1597 — a mirror of the Worldline/ULB credit card fraud
    dataset originally published on Kaggle
    (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

    The dataset contains 284,807 European credit card transactions from
    September 2013, with 492 fraud cases (0.172%). Features V1-V28 are the
    output of a PCA transformation applied by the data publishers to
    anonymize the original features; Time and Amount are kept as-is.
    Target column `Class` is 1 for fraud, 0 for legitimate.

Usage:
    # From project root, with venv active:
    python -m src.data.download

    # Force re-download (skip the cached file):
    python -m src.data.download --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# LEARN: Module-level logger named after the module path. Lets us configure
# logging globally later (e.g. send to a file in Phase 6) without changing
# every print() call. `print()` in production scripts is an anti-pattern.
logger = logging.getLogger(__name__)

# LEARN: Define paths relative to the project root so the script works
# regardless of where it's invoked from. `Path(__file__)` is the path to
# THIS file; .parents[2] walks up: file -> data/ -> src/ -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "creditcard.parquet"

# OpenML dataset identifier. Pinning the ID (not the name) makes this
# reproducible — names can be reused, IDs cannot.
OPENML_DATA_ID = 1597


def download_dataset(force: bool = False) -> Path:
    """Fetch the credit-card fraud dataset from OpenML and save as Parquet.

    Args:
        force: If True, re-download even if the cached Parquet file exists.

    Returns:
        Path to the Parquet file that was downloaded or already present.
    """
    # LEARN: mkdir(parents=True, exist_ok=True) is the Python equivalent of
    # `mkdir -p`. Creates intermediate dirs, doesn't error if it exists.
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
    # LEARN: as_frame=True returns a pandas DataFrame instead of a numpy
    # array. We want the column names (V1..V28, Amount, Class) preserved.
    # parser="auto" lets sklearn pick the fastest parser (pandas-native).
    bunch = fetch_openml(
        data_id=OPENML_DATA_ID,
        as_frame=True,
        parser="auto",
    )

    # `bunch.frame` is the full DataFrame with both features and target.
    df: pd.DataFrame = bunch.frame

    # LEARN: OpenML may return the target column as a string ("0"/"1") or
    # category dtype. Coerce to int so downstream code can rely on a clean
    # binary integer column.
    df["Class"] = df["Class"].astype(int)

    # Quick stats. Logging these makes the script's output self-documenting.
    n_rows = len(df)
    n_fraud = int(df["Class"].sum())
    fraud_pct = 100.0 * n_fraud / n_rows
    logger.info(
        "Fetched %d rows, %d columns. Fraud cases: %d (%.4f%%).",
        n_rows,
        df.shape[1],
        n_fraud,
        fraud_pct,
    )

    # LEARN: Parquet (via pyarrow) preserves dtypes, compresses well, and
    # reads ~10x faster than CSV in pandas. Snappy compression is the
    # sensible default — small + fast.
    logger.info("Writing Parquet to %s ...", OUTPUT_FILE)
    df.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy", index=False)
    logger.info("Done. File size: %.1f MB", OUTPUT_FILE.stat().st_size / 1024 / 1024)
    return OUTPUT_FILE


def main() -> int:
    """CLI entry point. Returns shell exit code (0 on success)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached file exists.",
    )
    args = parser.parse_args()

    # LEARN: basicConfig only takes effect if the root logger has no
    # handlers yet — safe to call here for script entry points. Format
    # includes timestamp + level + message; production-shape default.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        download_dataset(force=args.force)
    except Exception:
        # LEARN: logger.exception() logs the full traceback at ERROR level.
        # Cleaner than `except: print(traceback.format_exc())` and the
        # standard pattern for top-level error handling in scripts.
        logger.exception("Download failed.")
        return 1

    return 0


if __name__ == "__main__":
    # LEARN: sys.exit(main()) propagates the return code to the shell.
    # `python -m src.data.download` followed by `echo %ERRORLEVEL%` will
    # show 0 (success) or 1 (failure) — useful for CI / Prefect tasks later.
    sys.exit(main())
