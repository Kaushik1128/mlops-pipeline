# =============================================================================
# Custom MLflow tracking server image.
#
# The official MLflow images don't bundle psycopg2 (Postgres driver) or boto3
# (S3 client). Since our backend store is Postgres and our artifact store is
# MinIO (S3-compatible), we need both. Rather than pip-installing them at
# container start, we bake them into the image at build time.
# =============================================================================

# LEARN: `python:3.11-slim` = Debian-based, ~45 MB. We picked Debian over
# Alpine because psycopg2-binary ships precompiled wheels for glibc (Debian)
# but NOT for musl (Alpine) — using Alpine would force a 5-minute source
# build with gcc on every image rebuild.
FROM python:3.11-slim

# LEARN: Best-practice hygiene env vars for any Python container:
#   PYTHONDONTWRITEBYTECODE — skip .pyc files (saves a few MB, plus .pyc in
#                             a container layer is dead weight).
#   PYTHONUNBUFFERED        — force stdout/stderr unbuffered so `docker logs`
#                             shows output in real time instead of in chunks.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# LEARN: Install curl for the healthcheck. ~5 MB cost, but the alternative
# (a Python one-liner) is uglier and slower to fail. apt-get && clean in a
# single RUN layer keeps the image small (intermediate apt caches don't
# survive into the final image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# LEARN: Set a working directory so subsequent COPY paths are predictable
# and we have somewhere to put the requirements file.
WORKDIR /app

# LEARN: This is the most important caching optimization in any Python
# Dockerfile. COPY requirements.txt BEFORE we copy anything else. Docker
# caches layers by content hash — as long as requirements.txt doesn't change,
# the expensive `pip install` layer is reused. If we instead copied the
# whole project first, every code change would bust the pip cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# LEARN: --no-cache-dir tells pip not to keep its download cache. We don't
# need it after install completes, and skipping it shaves ~50 MB off the image.

# LEARN: Document the port for humans reading the file. EXPOSE does NOT
# publish the port (that's the compose `ports:` block) — it's purely metadata.
EXPOSE 5000

# LEARN: No CMD here. We set the command in docker-compose.yml because the
# arguments depend on env vars (Postgres credentials, MinIO bucket) that
# belong in the compose file, not the image.
