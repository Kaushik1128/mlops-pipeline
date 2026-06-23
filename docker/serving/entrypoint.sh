#!/bin/sh
# =============================================================================
# Serving container entrypoint.
#   1. Import the current @staging model from the MLflow registry into this
#      container's BentoML store (so the service always serves what's blessed).
#   2. Hand off (exec) to the BentoML server.
#
# Running import at startup means a `docker compose restart fraud-service`
# picks up a newly-promoted model — handy for Phase 6 auto-retraining.
# =============================================================================
set -e

echo "[entrypoint] Importing @staging model from MLflow (${MLFLOW_TRACKING_URI})..."
python -m src.serving.import_model

echo "[entrypoint] Starting BentoML service on :3000 ..."
# LEARN: exec replaces the shell process with bentoml, so signals (SIGTERM
# from `docker stop`) reach the server directly for a clean shutdown.
exec bentoml serve src.serving.service:FraudClassifier --host 0.0.0.0 --port 3000
