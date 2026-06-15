# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — install Python dependencies into a venv
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System packages needed only for building C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed model so the production image works offline.
# BAAI/bge-small-en-v1.5 is ~35 MB (ONNX) — fast, no PyTorch required.
RUN python -c "\
from fastembed import TextEmbedding; \
list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — lean image with non-root user
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="eComBot" \
      org.opencontainers.image.description="Multi-agent e-commerce support assistant" \
      org.opencontainers.image.version="1.0.0"

# Runtime system libraries only (libpq for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r ecombot && useradd -r -g ecombot -d /app -s /sbin/nologin ecombot

WORKDIR /app

# Copy virtualenv and pre-downloaded model from builder
COPY --from=builder /opt/venv /opt/venv
# fastembed caches the model under the home directory at build time
COPY --from=builder /root/.cache /home/ecombot/.cache

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_DIR=/home/ecombot/.cache/fastembed \
    # Suppress HuggingFace progress bars in logs
    TOKENIZERS_PARALLELISM=false

# Copy application source (owned by non-root user)
COPY --chown=ecombot:ecombot . .

# Make entrypoint executable
RUN chmod +x scripts/entrypoint.sh

USER ecombot

# Expose ADK web port
EXPOSE 8080

# Health-check: ADK web serves a frontend at "/"
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/ > /dev/null || exit 1

ENTRYPOINT ["scripts/entrypoint.sh"]

