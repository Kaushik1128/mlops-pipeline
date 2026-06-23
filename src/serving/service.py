"""BentoML service exposing the fraud model as a REST API.

Run locally:
    bentoml serve src.serving.service:FraudClassifier --reload

Then POST a transaction to http://localhost:3000/predict, or use the
auto-generated Swagger UI at http://localhost:3000.

Design notes:
  - We load the NATIVE sklearn Pipeline (mlflow.sklearn.load_model) rather than
    BentoML's pyfunc wrapper, because we need predict_proba (the fraud
    probability), not just the 0/1 label that pyfunc.predict returns.
  - The model expects a `log_amount` column (log1p of the raw Amount, applied
    in preprocess.py — OUTSIDE the model Pipeline). To avoid train/serve skew,
    the API accepts the RAW `amount` and applies log1p here, mirroring
    preprocessing exactly. (A cleaner design would fold log1p into the Pipeline
    so the model is fully self-contained; noted as a future improvement.)
"""
from __future__ import annotations

import os

import bentoml
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

BENTO_MODEL_NAME = "fraud_classifier:latest"
# LEARN: The decision threshold turns a probability into a yes/no. The right
# value is a business trade-off (missed fraud vs false alarms), NOT a code
# constant — so it's read from an env var at startup. Ops moves along the
# precision/recall curve by changing FRAUD_DECISION_THRESHOLD, no code change,
# no redeploy of the image. 0.5 is just the fallback default.
DEFAULT_THRESHOLD = 0.5
THRESHOLD_ENV_VAR = "FRAUD_DECISION_THRESHOLD"

# The exact feature-column order the model was trained on.
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
MODEL_COLUMNS = V_COLUMNS + ["log_amount"]


class Transaction(BaseModel):
    """One credit-card transaction to score.

    V1–V28 are the PCA-anonymized features from the source dataset. `amount`
    is the RAW transaction value in euros — the service log-transforms it
    internally to match preprocessing.
    """

    # LEARN: Listing all 28 PCA features explicitly makes the API contract
    # self-documenting — the auto-generated Swagger UI shows every field with
    # its type, and Pydantic validates that each is present and numeric.
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


@bentoml.service(
    name="fraud_classifier",
    # LEARN: traffic.timeout caps how long a single request may run before
    # BentoML returns a timeout — protects the service from hanging requests.
    traffic={"timeout": 10},
)
class FraudClassifier:
    # LEARN: BentoModel is a LAZY reference — it doesn't require the model to
    # exist at import time (unlike models.get()). Declaring it as a class
    # attribute still tells BentoML this service DEPENDS on this model, so
    # `bentoml build` bundles the exact version into the deployable artifact.
    bento_model = bentoml.models.BentoModel(BENTO_MODEL_NAME)

    def __init__(self) -> None:
        # LEARN: __init__ runs once per worker at startup, not per request —
        # so the (relatively expensive) model load happens once and is reused
        # across all requests. We resolve the concrete on-disk path at runtime
        # and load the native sklearn Pipeline (for predict_proba).
        model_ref = bentoml.models.get(BENTO_MODEL_NAME)
        native_path = os.path.join(model_ref.path, "mlflow_model")
        self.model = mlflow.sklearn.load_model(native_path)

        # LEARN: Read the threshold once at startup. Reading per-request would
        # let it change mid-flight; reading once gives every request in this
        # deployment a consistent, auditable decision boundary.
        self.threshold = float(os.getenv(THRESHOLD_ENV_VAR, DEFAULT_THRESHOLD))
        print(f"[fraud-service] decision threshold = {self.threshold}")

    @bentoml.api
    def predict(self, transaction: Transaction) -> Prediction:
        """Score a single transaction and return its fraud probability."""
        data = transaction.model_dump()
        # LEARN: Replicate preprocessing — raw amount -> log_amount via log1p,
        # exactly as preprocess.py did at training time. Mismatch here would be
        # silent train/serve skew: the model would see inputs it never trained on.
        raw_amount = data.pop("amount")
        data["log_amount"] = float(np.log1p(raw_amount))

        # LEARN: Build a single-row DataFrame with columns in the EXACT order
        # and naming the Pipeline was trained on. sklearn matches by column
        # name, but keeping order identical avoids any ambiguity.
        row = pd.DataFrame([[data[c] for c in MODEL_COLUMNS]], columns=MODEL_COLUMNS)

        proba = float(self.model.predict_proba(row)[:, 1][0])
        return Prediction(
            fraud_probability=round(proba, 6),
            is_fraud=bool(proba >= self.threshold),
            threshold=self.threshold,
        )
