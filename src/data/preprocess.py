"""Preprocess the raw credit-card fraud dataset.

Reads data/raw/creditcard.parquet, performs a stratified 80/20 train/test
split, applies log1p to the Amount column, and writes:
    data/processed/train.parquet
    data/processed/test.parquet

Scaling (StandardScaler) is intentionally NOT done here — it lives inside the
model training pipeline so the fitted scaler ships with the model as a single
sklearn Pipeline artifact, avoiding a separate scaler to version and load at
inference time.

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

    # Fail loudly if the expected schema is missing rather than training on bad data.
    expected_columns = {f"V{i}" for i in range(1, 29)} | {"Amount", TARGET_COLUMN}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}"
        )

    # log1p handles Amount == 0 safely (plain log(0) is undefined) and tames the
    # heavy right skew of transaction amounts.
    df["Amount"] = np.log1p(df["Amount"])
    df = df.rename(columns={"Amount": "log_amount"})

    # Stratify preserves the ~0.17% fraud ratio in both splits.
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y, shuffle=True,
    )

    train_df = X_train.assign(**{TARGET_COLUMN: y_train})
    test_df = X_test.assign(**{TARGET_COLUMN: y_test})

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    train_df.to_parquet(train_path, engine="pyarrow", compression="snappy", index=False)
    test_df.to_parquet(test_path, engine="pyarrow", compression="snappy", index=False)

    _log_split_stats(train_df, "TRAIN")
    _log_split_stats(test_df, "TEST")
    return train_path, test_path


def _log_split_stats(df: pd.DataFrame, label: str) -> None:
    """Log row count, fraud count, and fraud percentage for a split."""
    n = len(df)
    n_fraud = int(df[TARGET_COLUMN].sum())
    logger.info("%s: %d rows | fraud=%d (%.4f%%)", label, n, n_fraud, 100.0 * n_fraud / n)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                        help=f"Fraction of rows for the test set (default {DEFAULT_TEST_SIZE}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for the split (default {DEFAULT_SEED}).")
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
