# ── Stage 1: Build ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Geo-spatial C libs required by rasterio / GDAL / shapely
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source (backend, model repo, checkpoints, data, frontend)
COPY backend/ ./backend/
COPY changeformer_repo/ ./changeformer_repo/
COPY cdvqa_repo/ ./cdvqa_repo/
COPY checkpoints/ ./checkpoints/
COPY data/ ./data/
COPY frontend/ ./frontend/

# Outputs directory (writable at runtime)
RUN mkdir -p /app/outputs

# Expose port (Render sets $PORT)
EXPOSE 10000

# Health-check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-10000}/health')" || exit 1

# Start the FastAPI server – Render injects PORT env var
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
