# docker/Dockerfile
#
# Build from project root:
#   docker build -f docker/Dockerfile -t fraud-api .
#
# DEVIATION from spec (Section 9.1):
#   Spec build command: `docker build -t fraud-api .`
#   Actual command:     `docker build -f docker/Dockerfile -t fraud-api .`
#   Reason: Dockerfile lives at docker/Dockerfile per project structure — build
#   context remains the project root so all COPY paths resolve correctly.
#
# DEVIATION from spec (Section 7.5):
#   Spec artifact name: models/fraud_model_v{n}.joblib
#   Actual artifact:    models/xgb_generalized_primary.joblib
#   Reason: Name chosen during Phase 3 to reflect generalization strategy.
#   No functional impact — path is resolved via fraud_model_v1_config.json.

# ---------------------------------------------------------------------------
# Stage 1 — base
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Keeps Python output unbuffered so logs appear immediately in `docker logs`
ENV PYTHONUNBUFFERED=1 \
    # Prevents .pyc files from being written to disk inside the container
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2 — dependencies
# Install before copying application code so this layer is cached on rebuilds
# as long as requirements.txt hasn't changed.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3 — application code
# ---------------------------------------------------------------------------
COPY api/   ./api/
COPY src/   ./src/

# ---------------------------------------------------------------------------
# Stage 4 — model artifacts
#
# models/ and data/processed/ are gitignored — they must exist locally before
# running `docker build`. The container will fail at startup if they are absent.
#
# Copied explicitly rather than using COPY data/ to avoid pulling in raw CSVs
# (590k rows × 394 cols) which are also gitignored but may exist locally.
# ---------------------------------------------------------------------------
COPY models/                                    ./models/
COPY data/processed/feature_pipeline.joblib    ./data/processed/feature_pipeline.joblib
COPY data/monitoring/                           ./data/monitoring/

# Create the logs directory the JSONL inference logger writes to (Section 8.7).
# In production use, mount a host volume here so logs survive container restarts.
RUN mkdir -p /app/logs

# ---------------------------------------------------------------------------
# Stage 5 — runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

# Model is loaded once at startup via the FastAPI lifespan event in api/main.py.
# --workers 1: single worker is correct for this single-node inference target.
# --host 0.0.0.0: required to accept connections from outside the container.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
