"""Preprocess the raw credit-card fraud dataset.

Reads data/raw/creditcard.parquet, performs a stratified 80/20 train/test
split, applies log1p to the Amount column, and writes:
    data/processed/train.parquet
    data/processed/test.parquet

Scaling (StandardScaler) is INTENTIONALLY NOT done here — it belongs inside
the model training pipeline (Phase 3) so that the fitted scaler ships
together with the model as a single sklearn Pipeline artifact. This avoids
having to version and load a separate scaler at inference time.

Usage:
    python -m src.data.preprocess
    python -m src.data.preprocess --test-size 0.25 --seed 7
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "creditcard.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "Class"
# LEARN: Fixed seed makes the split deterministic — same input file, same
# command, same output bytes. This is what makes the DVC story work: rerun
# the pipeline on a clean checkout and the hashes match exactly.
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.20


def preprocess(
    raw_file: Path = RAW_FILE,
    out_dir: Path = PROCESSED_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """Run the preprocessing pipeline end-to-end.

    Args:
        raw_file: Path to the raw Parquet file.
        out_dir: Directory to write train.parquet and test.parquet into.
        test_size: Fraction of rows to allocate to the test set.
        seed: Random seed for the train/test split.

    Returns:
        (train_path, test_path) — paths to the two output files.
    """
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_file}. Run `python -m src.data.download` first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading raw data from %s", raw_file)
    df = pd.read_parquet(raw_file)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    # LEARN: Defensive validation. If the schema ever changes (e.g. an OpenML
    # mirror swap drops a column — as we already discovered with `Time`),
    # we want to fail loudly here rather than silently train on garbage.
    expected_columns = {f"V{i}" for i in range(1, 29)} | {"Amount", TARGET_COLUMN}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}"
        )

    # LEARN: log1p(x) = log(1 + x). The +1 inside the log handles
    # Amount == 0 (which exists in this data — refund/test transactions).
    # Plain log(0) is undefined and would produce -inf. log1p is the
    # numerically stable, NaN-safe choice for non-negative values.
    df["Amount"] = np.log1p(df["Amount"])
    df = df.rename(columns={"Amount": "log_amount"})
    logger.info("Applied log1p to Amount → renamed to log_amount")

    # LEARN: stratify=y ensures the positive class ratio is preserved in
    # both train and test. Without it, with 492 positives out of 284,807,
    # an unlucky random split could leave one half with significantly fewer
    # fraud rows — which would make evaluation high-variance.
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )

    train_df = X_train.assign(**{TARGET_COLUMN: y_train})
    test_df = X_test.assign(**{TARGET_COLUMN: y_test})

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"

    logger.info("Writing train set to %s", train_path)
    train_df.to_parquet(train_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("Writing test set to %s", test_path)
    test_df.to_parquet(test_path, engine="pyarrow", compression="snappy", index=False)

    # LEARN: Log the resulting class balance so we can verify stratification
    # worked. These numbers go into the script's audit trail.
    _log_split_stats(train_df, "TRAIN")
    _log_split_stats(test_df, "TEST")

    return train_path, test_path


def _log_split_stats(df: pd.DataFrame, label: str) -> None:
    """Log row count, fraud count, and fraud percentage for a split."""
    n = len(df)
    n_fraud = int(df[TARGET_COLUMN].sum())
    pct = 100.0 * n_fraud / n
    logger.info("%s: %d rows | fraud=%d (%.4f%%)", label, n, n_fraud, pct)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Fraction of rows for the test set (default {DEFAULT_TEST_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for the split (default {DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        preprocess(test_size=args.test_size, seed=args.seed)
    except Exception:
        logger.exception("Preprocessing failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
