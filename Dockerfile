# ============================================================================
# Stage 1: Builder - Uses 'uv' for lightning-fast installs
# ============================================================================
ARG BASE_DIGEST=python:3.11-slim-bookworm
FROM ${BASE_DIGEST} AS builder

# Install build dependencies for orjson/uvloop
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Fix pycares / aiodns incompatibility
RUN uv pip install --no-cache --compile \
    pycares==4.4.0 \
    aiodns==3.2.0

# Install everything else from requirements.txt
RUN uv pip install --no-cache --compile -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal & Optimized
# ============================================================================
FROM ${BASE_DIGEST} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy source code
COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./

# Environment Setup
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

# Security: Run as non-root
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

USER botuser

ENTRYPOINT ["python", "-u", "wrapper.py"]
