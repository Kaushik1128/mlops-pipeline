"""Produce a stratified smoke-test subset of the processed test split.

A small (default 1000-row) sample for fast iteration during model development:
it exercises the same code paths as the full test set at negligible cost.

Reads data/processed/test.parquet and writes data/processed/test_smoke.parquet.

Usage:
    python -m src.data.make_smoke_test
    python -m src.data.make_smoke_test --n-rows 500 --seed 7
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE = PROJECT_ROOT / "data" / "processed" / "test.parquet"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "test_smoke.parquet"

TARGET_COLUMN = "Class"
DEFAULT_N_ROWS = 1000
DEFAULT_SEED = 42


def make_smoke_test(
    in_file: Path = IN_FILE,
    out_file: Path = OUT_FILE,
    n_rows: int = DEFAULT_N_ROWS,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Create a stratified subset of the test set for fast iteration.

    Args:
        in_file: Source Parquet file (defaults to the processed test split).
        out_file: Destination Parquet file for the smoke subset.
        n_rows: Number of rows in the smoke sample. Must be <= rows in input.
        seed: Random seed for reproducible sampling.

    Returns:
        Path to the written Parquet file.
    """
    if not in_file.exists():
        raise FileNotFoundError(
            f"Source file not found at {in_file}. "
            "Run `python -m src.data.preprocess` first."
        )

    logger.info("Reading source data from %s", in_file)
    df = pd.read_parquet(in_file)
    logger.info("Source has %d rows, %d columns", len(df), df.shape[1])

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Source is missing the target column '{TARGET_COLUMN}'. "
            f"Got columns: {sorted(df.columns)}"
        )
    if n_rows > len(df):
        raise ValueError(f"n_rows ({n_rows}) cannot exceed source row count ({len(df)}).")

    # train_test_split as a stratified sampler: an integer train_size gives
    # exactly n_rows, and stratify preserves the ~0.17% fraud rate (a plain
    # random sample of 1000 rows could contain zero fraud cases).
    sample_df, _ = train_test_split(
        df, train_size=n_rows, stratify=df[TARGET_COLUMN], random_state=seed,
    )
    sample_df = sample_df.reset_index(drop=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing smoke-test set to %s", out_file)
    sample_df.to_parquet(out_file, engine="pyarrow", compression="snappy", index=False)

    _log_sample_stats(sample_df)
    return out_file


def _log_sample_stats(df: pd.DataFrame) -> None:
    """Log row count and class distribution of the smoke sample."""
    n = len(df)
    n_fraud = int(df[TARGET_COLUMN].sum())
    logger.info("SMOKE: %d rows | fraud=%d (%.4f%%)", n, n_fraud, 100.0 * n_fraud / n)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-rows", type=int, default=DEFAULT_N_ROWS,
                        help=f"Number of rows in the smoke sample (default {DEFAULT_N_ROWS}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for the sample (default {DEFAULT_SEED}).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        make_smoke_test(n_rows=args.n_rows, seed=args.seed)
    except Exception:
        logger.exception("Smoke-test generation failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
