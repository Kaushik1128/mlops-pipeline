# Demo walkthrough

A click-by-click script for a ~4-minute demo video (or a live walkthrough in an
interview). It tells one story: **a fraud model that detects its own decay and
heals itself.** Each step lists what to *do*, what the viewer *sees*, and what
to *say*.

> Before recording: `docker compose up -d` and wait for all 7 services healthy
> (`docker compose ps`). Activate the venv. Have these tabs open: MLflow (:5000),
> Fraud API Swagger (:3000), Prefect (:4200), Grafana (:3001).

---

## 0 · Hook (20s)

**Say:** "Anyone can train a model in a notebook. The hard part is everything
around it — serving it, watching it decay, and retraining it automatically.
This is that system, for credit-card fraud."

**Do:** `docker compose ps` — show **7 services** all healthy.
**Say:** "One `docker compose up` brings up the whole stack: storage, tracking,
serving, orchestration, and monitoring."

---

## 1 · Experiment tracking & the registry (45s)

**Do:** Open **MLflow (:5000)** → the `fraud-detection` experiment → show the
**3 runs** (logistic_regression, random_forest, xgboost). Sort by `pr_auc`.

**Say:** "I compared three models. The data is 0.17% fraud — extreme imbalance —
so I select on **PR-AUC**, not accuracy or ROC-AUC, which look great but lie
here. XGBoost wins at 0.87."

**Do:** Open a run → show params, the **git SHA + data-hash tags**, and the
confusion-matrix artifact. Then the **Models** tab → `fraud-classifier` →
version with the `@staging` alias.

**Say:** "Every run is reproducible — tagged with the exact code commit and data
version. The best model is promoted to `@staging` in the registry."

---

## 2 · The serving API (40s)

**Do:** Open the **Swagger UI (:3000)** → `POST /predict` → "Try it out" → paste
a known-fraud transaction → Execute.

**Say:** "The model is served as a containerized BentoML API that loads whatever
is `@staging`. Inputs are validated with Pydantic — here's a real fraud scoring
0.9999."

**Do:** Send a malformed payload (negative amount) → show the **400 validation
error**.
**Say:** "Bad input is rejected cleanly, not silently mispredicted."

---

## 3 · Drift detection (45s)

**Do:** In the terminal:
```bash
python -m src.monitoring.check_drift --current data/processed/test_drifted.parquet --weighted
```
**Say:** "Evidently compares live data against the training distribution. Crucially,
drift is **weighted by feature importance** — it fires on the features the model
actually relies on, not on noise."

**Do:** Open `reports/drift/test_drifted.html` → show the per-feature
distribution shifts and the "Dataset Drift is detected" banner.

---

## 4 · Auto-retraining (the centerpiece) (50s)

**Do:**
```bash
$env:PREFECT_API_URL="http://localhost:4200/api"
python -m src.flows.retraining_flow --current data/processed/test_drifted.parquet
```
**Do:** Open **Prefect (:4200)** → the flow run → show the **task graph**:
check_drift → retrain → evaluate_and_promote → (refresh_serving).

**Say:** "Drift triggers a Prefect flow: it retrains, then a **governance gate**
compares the candidate to the incumbent and promotes **only if it's genuinely
better** — so a bad retrain can never break production. If it promotes, the
serving container restarts and picks up the new model automatically."

---

## 5 · Live observability (40s)

**Do:**
```bash
python -m src.serving.generate_traffic --seconds 60
```
**Do:** Open **Grafana (:3001)** → the **Fraud Service Monitoring** dashboard.

**Say:** "The API is instrumented with Prometheus. Here's live request rate,
latency percentiles, predictions by outcome, and the **business fraud rate** —
so I'd see a problem on a dashboard, not in a customer complaint."

**Do:** Point at the panels updating in real time.

---

## 6 · Close (15s)

**Say:** "So: weighted drift detection triggers retraining, the gate promotes
only real improvements, and serving refreshes itself — the model maintains
itself, on a schedule, fully observable. That's a complete, production-shaped
MLOps lifecycle."

---

## Recording tips

- Pre-run the slow steps once so images are built / caches warm.
- Set `FRAUD_DECISION_THRESHOLD=0.5` in `.env` and recreate fraud-service for a
  livelier fraud rate during the demo.
- Generate traffic *before* opening Grafana so the graphs already have data.
- Keep terminal font large; hide secrets.
