"""Train a fraud-detection model and log everything to MLflow.

Config-driven: hyperparameters and paths come from configs/training.yaml, so
every experiment is a config change (logged by MLflow), not a code edit.

Each run logs:
  - params  : the model hyperparameters actually used
  - tags    : git commit SHA + DVC data hash (full reproducibility lineage)
  - metrics : PR-AUC, ROC-AUC, precision, recall, f1 (added in Step 4)
  - artifacts: the trained sklearn Pipeline + a confusion-matrix plot (Step 4)

Usage:
    python -m src.models.train --model logistic_regression
    python -m src.models.train --model xgboost --smoke
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

# LEARN: Import evaluate FIRST — it calls matplotlib.use("Agg") on import,
# setting the headless backend before pyplot is loaded below.
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
    """Return the current git commit SHA, or 'unknown' if not in a repo.

    LEARN: Tagging each MLflow run with the exact commit that produced it is
    THE key to reproducibility. Three months from now, a run scoring 0.85
    PR-AUC is just a number — unless you can `git checkout <sha>` and get the
    exact code back. We swallow errors (e.g. git not installed) rather than
    crash training over a missing tag.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_data_hash(data_path: Path) -> str:
    """Return the DVC content hash for a data file, or 'untracked'.

    LEARN: Rather than recompute an MD5 over a 57 MB parquet on every run, we
    read the hash DVC already computed and stored in the sibling `.dvc`
    pointer file. This ties the run directly to the DVC-versioned data: the
    tag value matches exactly what's in MinIO. If the .dvc file is missing
    (data not tracked), we degrade gracefully.
    """
    dvc_file = data_path.with_suffix(data_path.suffix + ".dvc")
    if not dvc_file.exists():
        return "untracked"
    meta = yaml.safe_load(dvc_file.read_text())
    return meta["outs"][0]["md5"]


def load_data(
    path: Path, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Read a parquet file and split into features X and target y."""
    df = pd.read_parquet(path)
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {path}. "
            f"Columns: {sorted(df.columns)}"
        )
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def resolve_params(model_name: str, model_params: dict, y_train: pd.Series) -> dict:
    """Resolve any 'auto' / data-dependent hyperparameters to concrete values.

    LEARN: We resolve BEFORE logging to MLflow so the run records the actual
    value used (e.g. scale_pos_weight=578.4), not the sentinel 'auto'. A run
    that logs 'auto' isn't reproducible — you can't tell what was used.

    Args:
        model_name: Model key.
        model_params: Raw hyperparameters from the config.
        y_train: Training labels, used to compute imbalance-based values.

    Returns:
        A new dict with sentinels replaced by concrete numbers.
    """
    params = dict(model_params)  # copy — never mutate the loaded YAML
    if model_name == "xgboost" and params.get("scale_pos_weight") == "auto":
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        # LEARN: scale_pos_weight scales the gradient of the positive class.
        # Setting it to (negatives / positives) makes the positive class's
        # total weight equal the negative class's — the XGBoost-recommended
        # default for imbalanced data. ~578 for us.
        params["scale_pos_weight"] = round(n_neg / n_pos, 2)
    return params


def build_pipeline(model_name: str, resolved_params: dict, seed: int):
    """Construct an sklearn Pipeline for the named model.

    Both models return a Pipeline so the serving interface is identical
    (`pipeline.predict_proba(X_raw)`), even though their internals differ.

    Args:
        model_name: 'logistic_regression' or 'xgboost'.
        resolved_params: Hyperparameters with sentinels already resolved.
        seed: Random seed for reproducibility.

    Returns:
        An unfitted sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(random_state=seed, **resolved_params)
        # LEARN: LogReg is scale-sensitive — coefficients and convergence
        # depend on feature magnitudes. StandardScaler (fit on train only,
        # inside the pipeline) is essential here.
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        params = dict(resolved_params)  # copy — don't mutate caller's dict
        clf = XGBClassifier(
            random_state=seed,
            n_jobs=-1,            # use all CPU cores
            eval_metric=params.pop("eval_metric", "aucpr"),
            **params,
        )
        # LEARN: No StandardScaler. Tree splits are threshold-based, so trees
        # are invariant to monotonic feature scaling — scaling would be wasted
        # compute. We still wrap in a Pipeline so serving code is identical
        # across models (always `pipeline.predict_proba(X)`).
        return Pipeline([("clf", clf)])

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        # LEARN: RandomForest is tree-based too → no StandardScaler. It handles
        # imbalance via class_weight='balanced' (like LogReg), not
        # scale_pos_weight, so resolve_params leaves its config untouched.
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **resolved_params)
        return Pipeline([("clf", clf)])

    # LEARN: This raise is the fallthrough for genuinely-unknown models — it
    # MUST be last, after every real branch, or it makes them unreachable.
    raise ValueError(f"Unknown model: {model_name}")

        


def main() -> int:
    """CLI entry point. Skeleton: opens an MLflow run, logs params + tags.

    Model training, metrics, and artifact logging are added in Step 4.
    """
    # LEARN: Windows consoles default to the cp1252 codepage, which can't
    # encode emoji/unicode that MLflow prints (e.g. the 🏃 in its run-complete
    # message) or em-dashes in our logs. Force UTF-8 so output never crashes
    # regardless of the host console. No-op on systems already using UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        choices=list(config["models"].keys()),
        default="logistic_regression",
        help="Which model config to use (from training.yaml).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # LEARN: Import mlflow inside main(), not at module top. mlflow is a heavy
    # import (~1s) and pulls network config; keeping it here means the cheap
    # helper functions above stay importable/testable without that cost.
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow_cfg = config["mlflow"]
    # LEARN: This points the CLIENT at the tracking server. Every log_* call
    # after this becomes an HTTP request to http://localhost:5000.
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    # LEARN: Creates the experiment if it doesn't exist, else reuses it.
    # All Phase 3 runs group under "fraud-detection" for side-by-side compare.
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # --- Load data ---
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

    # LEARN: `with mlflow.start_run()` opens a run and guarantees it's closed
    # (and marked FINISHED/FAILED) even if an exception fires inside. The
    # run_name shows in the UI; without it MLflow assigns a random name.
    with mlflow.start_run(run_name=model_name) as run:
        # Tags = metadata for filtering/lineage (searchable in the UI).
        mlflow.set_tags({
            "model_type": model_name,
            "git_sha": get_git_sha(),
            "data_hash_train": get_data_hash(train_path),
            "phase": "3",
        })
        # LEARN: Resolve 'auto' sentinels (e.g. scale_pos_weight) to concrete
        # numbers BEFORE logging, so MLflow records what was actually used.
        resolved_params = resolve_params(model_name, model_params, y_train)

        # Params = inputs to the run. Log the resolved hyperparameters plus
        # dataset shape so the run is self-describing.
        mlflow.log_params(resolved_params)
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_test", len(y_test))
        mlflow.log_param("seed", config["seed"])

        logger.info("Started MLflow run %s", run.info.run_id)
        if "scale_pos_weight" in resolved_params:
            logger.info("Resolved scale_pos_weight = %s", resolved_params["scale_pos_weight"])

        # --- Train ---
        pipeline = build_pipeline(model_name, resolved_params, config["seed"])
        logger.info("Fitting %s ...", model_name)
        pipeline.fit(X_train, y_train)

        # --- Evaluate ---
        # LEARN: predict_proba returns shape (n, 2): column 0 = P(legit),
        # column 1 = P(fraud). We want the positive-class probability for
        # PR-AUC etc., so we take [:, 1].
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_proba)
        mlflow.log_metrics(metrics)
        logger.info(
            "Metrics — PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f precision=%.4f recall=%.4f",
            metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
            metrics["precision"], metrics["recall"],
        )

        # --- Log confusion-matrix plot as an artifact ---
        fig = confusion_matrix_figure(y_test, y_proba, title=model_name)
        # LEARN: log_figure uploads the matplotlib figure straight to the
        # artifact store (MinIO) — no temp file on disk needed.
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)  # release the figure (evaluate.py left it open for us)

        # --- Log the trained Pipeline as a model artifact ---
        # LEARN: infer_signature records the input/output schema (column names,
        # dtypes, output shape). MLflow stores it with the model so the
        # registry and serving layer know exactly what inputs to expect.
        signature = infer_signature(X_test, y_proba)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            # input_example helps the UI render a sample and validates schema.
            input_example=X_test.iloc[:5],
        )
        logger.info("Logged model + plot to run %s", run.info.run_id)

    logger.info(
        "Run complete. View at %s/#/experiments/%s/runs/%s",
        mlflow_cfg["tracking_uri"],
        run.info.experiment_id,
        run.info.run_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
