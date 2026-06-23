"""Tests for src/models/train.py — pure helpers (no MLflow needed)."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.train import (
    build_pipeline,
    get_data_hash,
    load_data,
    resolve_params,
)


def test_resolve_params_computes_scale_pos_weight():
    y_train = pd.Series([0] * 95 + [1] * 5)  # 95 neg, 5 pos -> ratio 19.0
    resolved = resolve_params("xgboost", {"scale_pos_weight": "auto", "max_depth": 5}, y_train)
    assert resolved["scale_pos_weight"] == 19.0
    assert resolved["max_depth"] == 5  # other params untouched


def test_resolve_params_passthrough_for_non_xgboost():
    y_train = pd.Series([0, 1, 0, 1])
    params = {"C": 1.0, "class_weight": "balanced"}
    resolved = resolve_params("logistic_regression", params, y_train)
    assert resolved == params
    assert resolved is not params  # returns a copy, doesn't mutate input


def test_build_pipeline_logreg_has_scaler():
    pipe = build_pipeline("logistic_regression",
                          {"C": 1.0, "max_iter": 100, "class_weight": "balanced"}, seed=42)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["scaler", "clf"]


def test_build_pipeline_xgboost_has_no_scaler():
    pipe = build_pipeline("xgboost",
                          {"n_estimators": 10, "max_depth": 3, "scale_pos_weight": 5.0}, seed=42)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["clf"]  # trees need no scaling


def test_build_pipeline_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("not_a_model", {}, seed=42)


def test_get_data_hash_reads_dvc_pointer(tmp_path):
    data_file = tmp_path / "data.parquet"
    dvc_file = tmp_path / "data.parquet.dvc"
    dvc_file.write_text("outs:\n- md5: deadbeef1234\n  path: data.parquet\n")
    assert get_data_hash(data_file) == "deadbeef1234"


def test_get_data_hash_untracked(tmp_path):
    assert get_data_hash(tmp_path / "missing.parquet") == "untracked"


def test_load_data_splits_features_and_target(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "Class": [0, 1, 0]})
    path = tmp_path / "d.parquet"
    df.to_parquet(path)
    X, y = load_data(path, target_column="Class")
    assert list(X.columns) == ["a", "b"]
    assert y.tolist() == [0, 1, 0]


def test_load_data_missing_target_raises(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "d.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError, match="not found"):
        load_data(path, target_column="Class")
