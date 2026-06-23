"""Prefect flow: autonomous drift-triggered retraining.

The loop that makes the model self-maintaining:

    check drift (importance-weighted)
      └─ no drift  -> stop (model still fresh)
      └─ drift     -> retrain
                       └─ evaluate vs current @staging
                            └─ worse/equal -> keep incumbent
                            └─ better       -> promote to @staging
                                                 └─ refresh serving

Run once (manual):
    python -m src.flows.retraining_flow
    python -m src.flows.retraining_flow --current data/processed/test.parquet

Each task gets retries + shows in the Prefect UI at http://localhost:4200.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from prefect import flow, task, get_run_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


_CFG = load_config()
TRACKING_URI = _CFG["mlflow"]["tracking_uri"]
EXPERIMENT = _CFG["mlflow"]["experiment_name"]
REGISTERED_MODEL = _CFG["mlflow"]["registered_model_name"]

# LEARN: Minimum PR-AUC gain required to promote. Two reasons:
#   1. Avoids promoting on rounding/precision noise (a "0.874826 > 0.8748"
#      phantom improvement when the incumbent tag was stored rounded).
#   2. Governance: don't churn a known-good production model — with the risk
#      and cost of a deployment — for a negligible gain. Require a real margin.
MIN_IMPROVEMENT = 0.001


# ---------------------------------------------------------------------------
# Task 1 — detect drift
# ---------------------------------------------------------------------------
@task(retries=2, retry_delay_seconds=10)
def check_for_drift(current_file: str, weighted: bool = True) -> dict:
    """Run the importance-weighted drift check; return the signal dict."""
    logger = get_run_logger()
    from src.monitoring.check_drift import check_drift, load_staging_importances

    weights = load_staging_importances(tracking_uri=TRACKING_URI) if weighted else None
    signal = check_drift(current_file=Path(current_file), weights=weights)
    logger.info(
        "Drift check: detected=%s (basis=%s, weighted_share=%s, columns=%s)",
        signal["drift_detected"], signal["decision_basis"],
        signal.get("weighted_share"), signal["drifted_columns"],
    )
    return signal


# ---------------------------------------------------------------------------
# Task 2 — retrain
# ---------------------------------------------------------------------------
@task(retries=1, retry_delay_seconds=15)
def retrain_model(model_name: str) -> dict:
    """Retrain via the existing CLI, then return the new run's id + PR-AUC."""
    logger = get_run_logger()
    logger.info("Retraining %s ...", model_name)
    # LEARN: We shell out to the proven training CLI (own process, clean exit
    # code) rather than re-implement training in the flow. check=True turns a
    # non-zero exit into an exception, which Prefect surfaces + retries.
    subprocess.run(
        [sys.executable, "-m", "src.models.train", "--model", model_name],
        check=True, cwd=PROJECT_ROOT,
    )

    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    # The run we just created = newest FINISHED run for this model type.
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.model_type = '{model_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"], max_results=1,
    )
    run = runs[0]
    candidate = {"run_id": run.info.run_id, "pr_auc": float(run.data.metrics["pr_auc"])}
    logger.info("Candidate run %s — PR-AUC=%.4f", candidate["run_id"][:12], candidate["pr_auc"])
    return candidate


# ---------------------------------------------------------------------------
# Task 3 — evaluate vs incumbent, promote only if better
# ---------------------------------------------------------------------------
@task(retries=2, retry_delay_seconds=10)
def evaluate_and_promote(candidate: dict, model_name: str) -> dict:
    """Promote the candidate to @staging ONLY if it beats the incumbent."""
    logger = get_run_logger()
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Incumbent PR-AUC (from the @staging version's tag; None if no model yet).
    incumbent_pr_auc = None
    try:
        cur = client.get_model_version_by_alias(REGISTERED_MODEL, "staging")
        incumbent_pr_auc = float(cur.tags.get("pr_auc", "nan"))
    except Exception:
        logger.info("No current @staging model — first promotion.")

    cand_pr_auc = candidate["pr_auc"]
    # LEARN: The governance gate. Promote only if the candidate beats the
    # incumbent by at least MIN_IMPROVEMENT — never on noise, never to displace
    # a known-good production model for a negligible gain.
    better = incumbent_pr_auc is None or cand_pr_auc > incumbent_pr_auc + MIN_IMPROVEMENT
    logger.info("Gate: candidate=%.4f vs incumbent=%s (margin=%.3f) -> %s",
                cand_pr_auc, incumbent_pr_auc, MIN_IMPROVEMENT,
                "PROMOTE" if better else "KEEP INCUMBENT")

    if not better:
        return {"promoted": False, "candidate_pr_auc": cand_pr_auc,
                "incumbent_pr_auc": incumbent_pr_auc}

    # Register the candidate run + move the @staging alias to it.
    version = mlflow.register_model(f"runs:/{candidate['run_id']}/model", REGISTERED_MODEL)
    client.set_registered_model_alias(REGISTERED_MODEL, "staging", version.version)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "pr_auc", f"{cand_pr_auc:.4f}")
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "model_type", model_name)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "promoted_by", "auto_retraining_flow")
    logger.info("Promoted version %s to @staging (PR-AUC %.4f)", version.version, cand_pr_auc)
    return {"promoted": True, "candidate_pr_auc": cand_pr_auc,
            "incumbent_pr_auc": incumbent_pr_auc, "new_version": version.version}


# ---------------------------------------------------------------------------
# Task 4 — refresh the serving container so it re-imports @staging
# ---------------------------------------------------------------------------
@task(retries=1, retry_delay_seconds=10)
def refresh_serving() -> None:
    """Restart fraud-service; its entrypoint re-imports the new @staging model."""
    logger = get_run_logger()
    logger.info("Restarting fraud-service to pick up the new @staging model ...")
    subprocess.run(["docker", "compose", "restart", "fraud-service"],
                   check=True, cwd=PROJECT_ROOT)
    logger.info("Serving refreshed.")


# ---------------------------------------------------------------------------
# The flow — orchestrates the tasks with conditional branching
# ---------------------------------------------------------------------------
@flow(name="auto-retraining")
def auto_retraining_flow(
    current_file: str = str(PROJECT_ROOT / "data" / "processed" / "test_drifted.parquet"),
    model_name: str = "xgboost",
    weighted: bool = True,
) -> dict:
    """Drift-triggered retraining loop. Returns a summary of what it did.

    LEARN: A @flow is just Python — so the branching below (stop on no drift,
    promote only if better, refresh only if promoted) is plain `if`/`return`.
    Prefect records which path each run took, with per-task state in the UI.
    """
    logger = get_run_logger()

    # 1. Detect drift.
    signal = check_for_drift(current_file, weighted=weighted)
    if not signal["drift_detected"]:
        logger.info("No drift detected — model still fresh. Nothing to do.")
        return {"status": "no_drift", "drift": signal}

    # 2. Drift detected -> retrain.
    logger.info("Drift detected -> retraining.")
    candidate = retrain_model(model_name)

    # 3. Promote only if the candidate beats the incumbent.
    decision = evaluate_and_promote(candidate, model_name)
    if not decision["promoted"]:
        logger.info("Candidate did not beat incumbent — keeping current @staging.")
        return {"status": "retrained_no_promotion", "drift": signal, "decision": decision}

    # 4. Promoted -> refresh serving so the new model goes live.
    refresh_serving()
    logger.info("New model promoted and serving refreshed.")
    return {"status": "retrained_and_promoted", "drift": signal, "decision": decision}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--current", default=str(PROJECT_ROOT / "data" / "processed" / "test_drifted.parquet"),
                        help="Current production batch to check for drift.")
    parser.add_argument("--model", default="xgboost", help="Model to retrain.")
    parser.add_argument("--no-weighted", action="store_true",
                        help="Use plain column-share drift instead of importance-weighted.")
    parser.add_argument("--serve", action="store_true",
                        help="Register a SCHEDULED deployment and serve it (blocks).")
    parser.add_argument("--cron", default="0 2 * * *",
                        help="Cron schedule for --serve (default: daily 02:00).")
    args = parser.parse_args()

    if args.serve:
        # LEARN: .serve() registers a Prefect DEPLOYMENT (visible in the UI)
        # bound to a cron schedule, then runs a long-lived process that triggers
        # the flow automatically on schedule. This is what turns "a flow I run
        # by hand" into "a model that maintains itself".
        auto_retraining_flow.serve(
            name="auto-retraining-scheduled",
            cron=args.cron,
            parameters={"current_file": args.current, "model_name": args.model,
                        "weighted": not args.no_weighted},
        )
        return 0

    result = auto_retraining_flow(
        current_file=args.current, model_name=args.model, weighted=not args.no_weighted,
    )
    print(f"\nFlow result: {result['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
