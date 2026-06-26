"""Train a fraud-detection model and log everything to MLflow.

Config-driven: hyperparameters and paths come from configs/training.yaml, so
every experiment is a config change (logged by MLflow), not a code edit.

Each run logs the hyperparameters used, tags (git SHA + DVC data hash for
reproducibility lineage), metrics (PR-AUC, ROC-AUC, precision, recall, f1),
and artifacts (the trained sklearn Pipeline + a confusion-matrix plot).

Usage:
    python -m src.models.train --model logistic_regression
    python -m src.models.train --model xgboost
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

# Import evaluate first: it sets matplotlib's Agg backend before pyplot loads.
from src.models.evaluate import compute_metrics, confusion_matrix_figure
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the training YAML config into a dict."""
    with path.open() as f:
        return yaml.safe_load(f)


def get_git_sha() -> str:
    """Return the current git commit SHA, or 'unknown' if unavailable.

    Tagging each run with its commit ties a logged metric back to the exact
    code that produced it.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_data_hash(data_path: Path) -> str:
    """Return the DVC content hash for a data file, or 'untracked'.

    Reads the hash DVC already stored in the sibling `.dvc` pointer file,
    tying the run to the exact DVC-versioned data.
    """
    dvc_file = data_path.with_suffix(data_path.suffix + ".dvc")
    if not dvc_file.exists():
        return "untracked"
    meta = yaml.safe_load(dvc_file.read_text())
    return meta["outs"][0]["md5"]


def load_data(path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Read a parquet file and split into features X and target y."""
    df = pd.read_parquet(path)
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {path}. "
            f"Columns: {sorted(df.columns)}"
        )
    return df.drop(columns=[target_column]), df[target_column]


def resolve_params(model_name: str, model_params: dict, y_train: pd.Series) -> dict:
    """Resolve data-dependent hyperparameter sentinels to concrete values.

    Resolved before logging so MLflow records the actual value used (e.g.
    scale_pos_weight=578.4), not the sentinel 'auto'.
    """
    params = dict(model_params)  # copy — never mutate the loaded YAML
    if model_name == "xgboost" and params.get("scale_pos_weight") == "auto":
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        # negatives / positives makes the positive class's total weight equal
        # the negative class's — XGBoost's recommended default for imbalance.
        params["scale_pos_weight"] = round(n_neg / n_pos, 2)
    return params


def build_pipeline(model_name: str, resolved_params: dict, seed: int):
    """Construct an (unfitted) sklearn Pipeline for the named model.

    Every model returns a Pipeline so the serving interface is identical
    (`pipeline.predict_proba(X)`), regardless of internals.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        # LogReg is scale-sensitive, so it gets a StandardScaler (fit on train
        # only, inside the pipeline).
        clf = LogisticRegression(random_state=seed, **resolved_params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        params = dict(resolved_params)  # copy — don't mutate caller's dict
        clf = XGBClassifier(
            random_state=seed, n_jobs=-1,
            eval_metric=params.pop("eval_metric", "aucpr"), **params,
        )
        # Trees are invariant to feature scaling, so no StandardScaler.
        return Pipeline([("clf", clf)])

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        # Tree-based: no scaler; handles imbalance via class_weight='balanced'.
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **resolved_params)
        return Pipeline([("clf", clf)])

    raise ValueError(f"Unknown model: {model_name}")


def main() -> int:
    """CLI entry point: train the chosen model and log the run to MLflow."""
    # Windows consoles default to cp1252, which can't encode the unicode MLflow
    # prints; force UTF-8 so output never crashes. No-op on UTF-8 systems.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model", choices=list(config["models"].keys()),
        default="logistic_regression",
        help="Which model config to use (from training.yaml).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Imported here rather than at module top: mlflow is a heavy import, and
    # keeping it local lets the helper functions above stay cheap to import/test.
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    train_path = PROJECT_ROOT / config["data"]["train"]
    test_path = PROJECT_ROOT / config["data"]["test"]
    target = config["data"]["target_column"]

    logger.info("Loading train data from %s", train_path)
    X_train, y_train = load_data(train_path, target)
    logger.info("Loading test data from %s", test_path)
    X_test, y_test = load_data(test_path, target)
    logger.info(
        "Train: %d rows (%d fraud) | Test: %d rows (%d fraud)",
        len(y_train), int(y_train.sum()), len(y_test), int(y_test.sum()),
    )

    model_name = args.model
    model_params = config["models"][model_name]

    # start_run guarantees the run is closed (FINISHED/FAILED) even on error.
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags({
            "model_type": model_name,
            "git_sha": get_git_sha(),
            "data_hash_train": get_data_hash(train_path),
            "phase": "3",
        })
        resolved_params = resolve_params(model_name, model_params, y_train)
        mlflow.log_params(resolved_params)
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_test", len(y_test))
        mlflow.log_param("seed", config["seed"])

        logger.info("Started MLflow run %s", run.info.run_id)
        if "scale_pos_weight" in resolved_params:
            logger.info("Resolved scale_pos_weight = %s", resolved_params["scale_pos_weight"])

        pipeline = build_pipeline(model_name, resolved_params, config["seed"])
        logger.info("Fitting %s ...", model_name)
        pipeline.fit(X_train, y_train)

        # Column 1 of predict_proba is P(fraud).
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_proba)
        mlflow.log_metrics(metrics)
        logger.info(
            "Metrics — PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f precision=%.4f recall=%.4f",
            metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
            metrics["precision"], metrics["recall"],
        )

        fig = confusion_matrix_figure(y_test, y_proba, title=model_name)
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)

        # Record the input/output schema with the model for the registry + serving.
        signature = infer_signature(X_test, y_proba)
        mlflow.sklearn.log_model(
            sk_model=pipeline, artifact_path="model",
            signature=signature, input_example=X_test.iloc[:5],
        )
        logger.info("Logged model + plot to run %s", run.info.run_id)

    logger.info(
        "Run complete. View at %s/#/experiments/%s/runs/%s",
        mlflow_cfg["tracking_uri"], run.info.experiment_id, run.info.run_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
