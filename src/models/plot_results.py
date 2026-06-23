"""Generate result charts for the README from the logged MLflow models.

Produces:
  assets/pr_curves.png         — precision-recall curves for all 3 models
  assets/confusion_matrix.png  — confusion matrix for the @staging model

Run with the stack up:
    python -m src.models.plot_results
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save to file, no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from src.models.evaluate import confusion_matrix_figure  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.parquet"
TRACKING_URI = "http://localhost:5000"
EXPERIMENT = "fraud-detection"
MODELS = ["logistic_regression", "random_forest", "xgboost"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)

    df = pd.read_parquet(TEST_FILE)
    X, y = df.drop(columns=["Class"]), df["Class"]
    ASSETS_DIR.mkdir(exist_ok=True)

    # --- Precision-Recall curves for all three models ---
    fig, ax = plt.subplots(figsize=(7, 5))
    xgb_proba = None
    for name in MODELS:
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.model_type = '{name}' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"], max_results=1,
        )
        if not runs:
            logger.info("No run for %s, skipping", name)
            continue
        model = mlflow.sklearn.load_model(f"runs:/{runs[0].info.run_id}/model")
        proba = model.predict_proba(X)[:, 1]
        ap = average_precision_score(y, proba)
        prec, rec, _ = precision_recall_curve(y, proba)
        ax.plot(rec, prec, linewidth=2, label=f"{name} (PR-AUC={ap:.3f})")
        if name == "xgboost":
            xgb_proba = proba
        logger.info("%s: PR-AUC=%.3f", name, ap)

    # Baseline: a random classifier scores the positive prevalence.
    ax.axhline(y.mean(), linestyle="--", color="grey", alpha=0.7,
               label=f"random ({y.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — test set (0.17% fraud)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "pr_curves.png", dpi=120)
    plt.close(fig)
    logger.info("Saved assets/pr_curves.png")

    # --- Confusion matrix for the champion (XGBoost) ---
    if xgb_proba is not None:
        cm = confusion_matrix_figure(y.values, xgb_proba, threshold=0.5,
                                     title="XGBoost — confusion matrix")
        cm.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=120)
        plt.close(cm)
        logger.info("Saved assets/confusion_matrix.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
