# ============================================================================
# Stage 1: Builder - Uses 'uv' for lightning-fast installs
# ============================================================================
ARG BASE_DIGEST=python:3.11-slim-bookworm
FROM ${BASE_DIGEST} AS builder

# Install build dependencies for orjson/uvloop/numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Fix specific incompatibilities and install requirements
RUN uv pip install --no-cache --compile \
    pycares==4.4.0 \
    aiodns==3.2.0 && \
    uv pip install --no-cache --compile -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal & Optimized
# ============================================================================
FROM ${BASE_DIGEST} AS runtime

# CRITICAL: libgomp1 is required for Numba's parallel execution (prange)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# CRITICAL FIX: Create writable cache directory for Numba JIT
# This ensures Numba can save compiled functions
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

# Copy source code and config
COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./

# OPTIMIZED: Environment variables for performance
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp \
    TZ=Asia/Kolkata

# OPTIMIZED: Run as non-root user (security best practice)
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app /tmp/numba_cache && \
    chmod +x wrapper.py

USER botuser

# Set entry point
ENTRYPOINT ["python", "-u", "wrapper.py"]