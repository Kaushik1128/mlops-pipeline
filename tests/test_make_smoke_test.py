"""Tests for src/data/make_smoke_test.py — stratified subsampling."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.make_smoke_test import make_smoke_test


def _synthetic(tmp_path):
    # 500 rows: 450 legit, 50 fraud.
    df = pd.DataFrame({
        "V1": range(500),
        "V2": range(500, 1000),
        "Class": [0] * 450 + [1] * 50,
    })
    src = tmp_path / "test.parquet"
    df.to_parquet(src)
    return src


def test_smoke_test_exact_size_and_columns(tmp_path):
    src = _synthetic(tmp_path)
    out = tmp_path / "smoke.parquet"
    make_smoke_test(in_file=src, out_file=out, n_rows=100, seed=42)
    result = pd.read_parquet(out)
    assert len(result) == 100
    assert "Class" in result.columns


def test_smoke_test_preserves_both_classes(tmp_path):
    src = _synthetic(tmp_path)
    out = tmp_path / "smoke.parquet"
    make_smoke_test(in_file=src, out_file=out, n_rows=100, seed=42)
    result = pd.read_parquet(out)
    # Stratified: both classes present, ~10% positive (50/500).
    assert set(result["Class"].unique()) == {0, 1}
    assert result["Class"].sum() == pytest.approx(10, abs=3)


def test_smoke_test_is_deterministic(tmp_path):
    src = _synthetic(tmp_path)
    out1, out2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    make_smoke_test(in_file=src, out_file=out1, n_rows=50, seed=7)
    make_smoke_test(in_file=src, out_file=out2, n_rows=50, seed=7)
    pd.testing.assert_frame_equal(pd.read_parquet(out1), pd.read_parquet(out2))
