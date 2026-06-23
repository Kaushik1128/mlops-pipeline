"""Tests for src/monitoring/check_drift.py parse helpers (no Evidently needed)."""
from __future__ import annotations

from src.monitoring.check_drift import _parse_column, _parse_threshold


def test_parse_column_extracts_name():
    name = "ValueDrift(column=V14,method=Wasserstein distance (normed),threshold=0.1)"
    assert _parse_column(name) == "V14"


def test_parse_threshold_extracts_value():
    name = "ValueDrift(column=V14,method=Wasserstein distance (normed),threshold=0.1)"
    assert _parse_threshold(name) == 0.1


def test_parse_helpers_return_none_when_absent():
    assert _parse_column("DriftedColumnsCount(drift_share=0.5)") is None
    assert _parse_threshold("DriftedColumnsCount(drift_share=0.5)") is None
