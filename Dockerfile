# ============================================================================
# Data Pipeline Service - Production Dockerfile
# ============================================================================
# Handles data ingestion from Polygon.io, Perplexity AI, and Supabase
# Features: Real-time WebSocket, data validation, quality checks
# ============================================================================

FROM python:3.11-slim as builder

LABEL maintainer="RRRalgorithms Team"
LABEL description="Data Pipeline Service - Market data ingestion and processing"

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN useradd -m -u 1000 -s /bin/bash pipelineuser && \
    mkdir -p /app/logs /app/data /app/cache && \
    chown -R pipelineuser:pipelineuser /app

COPY --from=builder --chown=pipelineuser:pipelineuser /root/.local /home/pipelineuser/.local
COPY --chown=pipelineuser:pipelineuser . .

USER pipelineuser

ENV PATH=/home/pipelineuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:$PYTHONPATH

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from data_pipeline.supabase.client import SupabaseClient; print('OK')" || exit 1

CMD ["python", "src/data_pipeline/main.py"]
