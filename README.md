# MLOps Pipeline — Credit Card Fraud Detection

> Production-grade MLOps pipeline with experiment tracking, data versioning,
> model registry, drift detection, and auto-retraining. Built end-to-end as a
> portfolio project on a fully local, free, Docker-based stack.

**Status: Phase 4 of 8 complete** — model training, MLflow registry, live serving API.
A full write-up with architecture diagram and demo recording lands in Phase 8.

---

## Build progress

- [x] **Phase 1 — Infrastructure.** Docker Compose stack: MinIO (S3-compatible
  object storage), PostgreSQL, MLflow tracking server, Prefect. Custom bridge
  network, named volumes, healthchecks, init-container bucket provisioning.
- [x] **Phase 2 — Data pipeline.** Dataset download (OpenML), EDA, stratified
  preprocessing, and full data versioning with DVC pushing to MinIO.
  Reproducibility verified (wipe local → `dvc pull` → identical hashes).
- [x] **Phase 3 — Model training.** Config-driven training, MLflow experiment
  tracking, reusable evaluation (PR-AUC / F1 / confusion matrix), and model
  registry with alias-based promotion.
- [x] **Phase 4 — Model serving.** Containerized BentoML REST API that imports
  the `@staging` model from MLflow at startup. Pydantic-validated `/predict`
  endpoint, auto-generated Swagger docs, and a deploy-time decision threshold.
- [ ] **Phase 5 — Drift detection** (Evidently AI)
- [ ] **Phase 6 — Orchestration + auto-retraining** (Prefect)
- [ ] **Phase 7 — Observability** (Prometheus + Grafana)
- [ ] **Phase 8 — Docs, architecture diagram, demo**

---

## Current results

Two models compared on a held-out test set (0.17% fraud — extreme class
imbalance, so **PR-AUC** is the headline metric, not accuracy or ROC-AUC):

| Model | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.708 | 0.971 | 0.058 | 0.918 |
| **XGBoost** (`@staging`) | **0.875** | 0.976 | 0.837 | 0.837 |

XGBoost lifts PR-AUC by +0.167 and precision from 6% → 84%, while ROC-AUC
barely moves — a concrete demonstration of why ROC-AUC is misleading under
heavy imbalance. The winner is registered to MLflow as `fraud-classifier`
and promoted via the `@staging` alias.

---

## Tech stack

| Layer | Tool |
|---|---|
| Model training | Scikit-learn, XGBoost |
| Experiment tracking + registry | MLflow |
| Data versioning | DVC |
| Drift detection | Evidently AI *(Phase 5)* |
| Orchestration | Prefect *(Phase 6)* |
| Model serving | BentoML *(Phase 4)* |
| Observability | Prometheus + Grafana *(Phase 7)* |
| Infrastructure | Docker Compose + MinIO (S3-compatible) |
| Dataset | Credit Card Fraud Detection (Worldline/ULB, via OpenML id=1597) |

---

## Quickstart

```bash
# 1. Bring up the local stack (MinIO, Postgres, MLflow, Prefect)
cp .env.example .env          # then fill in passwords
docker compose up -d

# 2. Create a Python env and install deps
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt

# 3. Get and version the data
python -m src.data.download
python -m src.data.preprocess
dvc push

# 4. Train, evaluate, and register a model
python -m src.models.train --model xgboost
python -m src.models.register

# 5. Serve it — the fraud-service container imports @staging and serves it
docker compose up -d --build fraud-service
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"V1": -1.36, ..., "V28": -0.02, "amount": 149.62}}'
```

Service UIs (after `docker compose up -d`):

| Service | URL |
|---|---|
| **Fraud API + Swagger docs** | **http://localhost:3000** |
| MLflow (experiments + registry) | http://localhost:5000 |
| MinIO console (object storage) | http://localhost:9001 |
| Prefect (orchestration) | http://localhost:4200 |

---

## Repository layout

```
configs/            Training config (hyperparameters, paths)
data/               DVC-tracked datasets (raw + processed)
docker/             Custom Dockerfiles (MLflow image)
notebooks/          Exploratory data analysis
src/data/           Download + preprocessing
src/models/         Training, evaluation, registry
docker-compose.yml  The full local stack
```

> Built as a learning-focused portfolio project. Code favours clarity and
> documented decisions over cleverness.
