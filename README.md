# MLOps Pipeline — Credit Card Fraud Detection

> Production-grade MLOps pipeline with drift detection and auto-retraining.
> Built as a portfolio project. **Work in progress — Phase 1.**

A full README with architecture diagram, demo recording, and setup instructions
will be written in Phase 8. For now, this file just exists so the repo doesn't
look empty.

## Stack (planned)

| Layer | Tool |
|---|---|
| Model training | PyTorch + Scikit-learn |
| Experiment tracking + registry | MLflow |
| Data versioning | DVC |
| Drift detection | Evidently AI |
| Orchestration | Prefect |
| Model serving | BentoML |
| Observability | Prometheus + Grafana |
| Infrastructure | Docker Compose + MinIO (S3) |
| Dataset | UCI Credit Card Fraud Detection |
