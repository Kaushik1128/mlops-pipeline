"""Import the @staging model from the MLflow registry into BentoML's store.

MLflow is the governance registry (which model is blessed); BentoML has its
own local model store that the serving runtime loads from. This script is the
bridge: it pulls models:/fraud-classifier@staging out of MLflow and imports it
into BentoML, carrying source-run metadata along for traceability.

Run this whenever the @staging alias moves to a new version (e.g. after a
Phase 6 retrain) to refresh what the service will serve.

Usage:
    python -m src.serving.import_model
    python -m src.serving.import_model --alias staging
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"

# LEARN: BentoML model names allow [a-z0-9_.-]. We use underscores to match
# BentoML conventions; this is the name the service will reference.
BENTO_MODEL_NAME = "fraud_classifier"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the training YAML config."""
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    """Import the aliased MLflow model into the BentoML model store."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Windows UTF-8, see train.py

    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--alias", default="staging",
                        help="Registry alias to import (default staging).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import bentoml
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow_cfg = config["mlflow"]
    # LEARN: Prefer the MLFLOW_TRACKING_URI env var over the config default.
    # On the host we use http://localhost:5000; inside the Docker network the
    # serving container sets http://mlflow:5000 (service-name DNS). Same code,
    # different address depending on who's calling and from where.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("Using MLflow tracking URI: %s", tracking_uri)
    registered_name = mlflow_cfg["registered_model_name"]

    # LEARN: Resolve the alias to a concrete version so we can record exactly
    # what we imported (and surface its score in the BentoML metadata).
    client = MlflowClient()
    version = client.get_model_version_by_alias(registered_name, args.alias)
    logger.info(
        "Resolved %s@%s -> version %s (run %s)",
        registered_name, args.alias, version.version, version.run_id[:12],
    )

    # LEARN: import_model copies the MLflow model artifacts into BentoML's
    # store and wraps them so BentoML can load/serve them. model_uri uses the
    # alias form so we always import whatever is currently @staging.
    model_uri = f"models:/{registered_name}@{args.alias}"
    bento_model = bentoml.mlflow.import_model(
        BENTO_MODEL_NAME,
        model_uri,
        # Labels are searchable key/values; metadata is free-form provenance.
        labels={"stage": args.alias, "framework": "sklearn-pipeline"},
        metadata={
            "mlflow_version": version.version,
            "mlflow_run_id": version.run_id,
            "pr_auc": version.tags.get("pr_auc", "unknown"),
            "model_type": version.tags.get("model_type", "unknown"),
        },
    )
    logger.info("Imported into BentoML store as: %s", bento_model.tag)
    logger.info("List all with: bentoml models list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
