# =============================================================================
# Custom MLflow tracking server image.
#
# The official MLflow images don't bundle psycopg2 (Postgres driver) or boto3
# (S3 client). Since our backend store is Postgres and our artifact store is
# MinIO (S3-compatible), we need both. Rather than pip-installing them at
# container start, we bake them into the image at build time.
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

