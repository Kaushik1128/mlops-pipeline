"""Select the best run in the experiment and register it in the Model Registry.

Separation of concerns: train.py LOGS models (every run, good or bad); this
script PROMOTES the best one. Promotion is a governance decision, not an
automatic side-effect of training.

Selection metric: PR-AUC (the right headline metric for our 0.17% positive
rate — see the EDA notebook and Step 5 analysis).

Usage:
    python -m src.models.register
    python -m src.models.register --metric pr_auc --alias staging
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the training YAML config."""
    with path.open() as f:
        return yaml.safe_load(f)


def find_best_run(client, experiment_id: str, metric: str):
    """Return the FINISHED run with the highest value of `metric`.

    LEARN: We search server-side, ordered by the metric descending, and filter
    to FINISHED runs only — a crashed/running run shouldn't be promotable.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No FINISHED runs found in experiment {experiment_id}. "
            "Train a model first: python -m src.models.train --model xgboost"
        )
    return runs[0]


def main() -> int:
    """CLI entry point: find best run, register + alias it."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Windows UTF-8, see train.py

    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metric", default="pr_auc",
                        help="Metric to select the best run by (default pr_auc).")
    parser.add_argument("--alias", default="staging",
                        help="Alias to assign the registered version (default staging).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = MlflowClient()

    experiment = client.get_experiment_by_name(mlflow_cfg["experiment_name"])
    if experiment is None:
        raise RuntimeError(f"Experiment '{mlflow_cfg['experiment_name']}' not found.")

    # --- Select the best run ---
    best = find_best_run(client, experiment.experiment_id, args.metric)
    model_name = best.data.tags.get("mlflow.runName", "?")
    score = best.data.metrics.get(args.metric)
    logger.info(
        "Best run: %s (%s) with %s=%.4f",
        best.info.run_id[:12], model_name, args.metric, score,
    )

    # --- Register the model ---
    # LEARN: model_uri 'runs:/<run_id>/model' points at the artifact path we
    # logged in train.py (mlflow.sklearn.log_model used artifact_path="model").
    # register_model creates the registered model if it doesn't exist, then
    # adds a new immutable version (v1, v2, ...).
    registered_name = mlflow_cfg["registered_model_name"]
    model_uri = f"runs:/{best.info.run_id}/model"
    logger.info("Registering %s -> '%s'", model_uri, registered_name)
    version = mlflow.register_model(model_uri=model_uri, name=registered_name)
    logger.info("Created version %s of '%s'", version.version, registered_name)

    # --- Alias + tags (the modern MLflow promotion mechanism) ---
    # LEARN: Aliases replaced the deprecated stage system (None/Staging/
    # Production). An alias is a movable pointer: `@staging` always points at
    # whichever version is current staging. The serving layer (Phase 4) loads
    # `models:/fraud-classifier@staging` and never needs to know the version
    # number — promote a new version by moving the alias.
    client.set_registered_model_alias(registered_name, args.alias, version.version)
    # Tags record WHY this version was promoted — its score and source metric.
    client.set_model_version_tag(registered_name, version.version,
                                 "selection_metric", args.metric)
    client.set_model_version_tag(registered_name, version.version,
                                 f"{args.metric}", f"{score:.4f}")
    client.set_model_version_tag(registered_name, version.version,
                                 "model_type", model_name)
    logger.info("Set alias '@%s' on version %s and tagged it.",
                args.alias, version.version)

    logger.info("Done. View registry at %s/#/models/%s",
                mlflow_cfg["tracking_uri"], registered_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
