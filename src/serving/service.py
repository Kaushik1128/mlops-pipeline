"""BentoML service exposing the fraud model as a REST API.

Run locally:
    bentoml serve src.serving.service:FraudClassifier --reload

Then POST a transaction to http://localhost:3000/predict, or use the
auto-generated Swagger UI at http://localhost:3000.

The native sklearn Pipeline is loaded (not BentoML's pyfunc wrapper) because we
need predict_proba, not just the 0/1 label. The model expects a `log_amount`
column (log1p of the raw Amount, applied during preprocessing); the API accepts
the raw `amount` and applies log1p here to avoid train/serve skew.
"""
from __future__ import annotations

import os

import bentoml
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

BENTO_MODEL_NAME = "fraud_classifier:latest"

# The decision threshold is a business trade-off (missed fraud vs false alarms),
# so it's read from an env var at startup rather than hard-coded.
DEFAULT_THRESHOLD = 0.5
THRESHOLD_ENV_VAR = "FRAUD_DECISION_THRESHOLD"

# The exact feature-column order the model was trained on.
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
MODEL_COLUMNS = V_COLUMNS + ["log_amount"]

# Custom Prometheus metrics (alongside BentoML's built-in request/latency
# metrics) capturing the business signal: fraud-vs-legit counts and the score
# distribution. Registered via bentoml.metrics so they appear on /metrics.
fraud_predictions_total = bentoml.metrics.Counter(
    name="fraud_predictions_total",
    documentation="Total predictions, labelled by outcome (fraud / legit).",
    labelnames=["result"],
)
fraud_probability = bentoml.metrics.Histogram(
    name="fraud_probability",
    documentation="Distribution of predicted fraud probabilities.",
    buckets=(0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0),
)


class Transaction(BaseModel):
    """One credit-card transaction to score.

    V1–V28 are the PCA-anonymized features from the source dataset. `amount`
    is the raw transaction value in euros — the service log-transforms it
    internally to match preprocessing.
    """

    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float
    V14: float; V15: float; V16: float; V17: float; V18: float; V19: float
    V20: float; V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    amount: float = Field(..., ge=0, description="Raw transaction value in euros (>= 0).")


class Prediction(BaseModel):
    """The scored response returned to the caller."""

    fraud_probability: float = Field(..., description="P(fraud) in [0, 1].")
    is_fraud: bool = Field(..., description="True if probability >= threshold.")
    threshold: float = Field(..., description="Decision threshold applied.")


@bentoml.service(name="fraud_classifier", traffic={"timeout": 10})
class FraudClassifier:
    # Lazy reference: doesn't require the model at import time, and tells
    # `bentoml build` which model version to bundle.
    bento_model = bentoml.models.BentoModel(BENTO_MODEL_NAME)

    def __init__(self) -> None:
        # Runs once per worker at startup: load the native Pipeline + threshold.
        model_ref = bentoml.models.get(BENTO_MODEL_NAME)
        native_path = os.path.join(model_ref.path, "mlflow_model")
        self.model = mlflow.sklearn.load_model(native_path)
        self.threshold = float(os.getenv(THRESHOLD_ENV_VAR, DEFAULT_THRESHOLD))
        print(f"[fraud-service] decision threshold = {self.threshold}")

    @bentoml.api
    def predict(self, transaction: Transaction) -> Prediction:
        """Score a single transaction and return its fraud probability."""
        data = transaction.model_dump()
        # Replicate preprocessing exactly (raw amount -> log_amount) to avoid
        # train/serve skew.
        raw_amount = data.pop("amount")
        data["log_amount"] = float(np.log1p(raw_amount))

        # Single-row DataFrame with columns in the model's trained order.
        row = pd.DataFrame([[data[c] for c in MODEL_COLUMNS]], columns=MODEL_COLUMNS)

        proba = float(self.model.predict_proba(row)[:, 1][0])
        is_fraud = bool(proba >= self.threshold)

        fraud_predictions_total.labels(result="fraud" if is_fraud else "legit").inc()
        fraud_probability.observe(proba)

        return Prediction(
            fraud_probability=round(proba, 6),
            is_fraud=is_fraud,
            threshold=self.threshold,
        )
