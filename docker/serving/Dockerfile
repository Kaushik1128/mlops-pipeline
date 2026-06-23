# =============================================================================
# Fraud-classifier serving image.
#
# Imports the @staging model from MLflow at startup and serves it via BentoML.
# Built by docker-compose (context = project root). Python 3.12 to match the
# version the model's dependencies resolve against.
# =============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BENTOML_DO_NOT_TRACK=True \
    PYTHONPATH=/app
# LEARN: PYTHONPATH=/app so `python -m src.serving...` and
# `bentoml serve src.serving.service:FraudClassifier` resolve the package.

WORKDIR /app

# LEARN: Copy requirements first and install before copying code — keeps the
# expensive pip-install layer cached across code-only changes (same layer-
# ordering lesson as the MLflow Dockerfile in Phase 1).
COPY docker/serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the files the service needs at runtime.
COPY src/ ./src/
COPY configs/ ./configs/
COPY docker/serving/entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 3000
ENTRYPOINT ["./entrypoint.sh"]
