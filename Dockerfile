# ============================================================================
# Stage 1: Builder - uv + cache, numpy/numba-safe
# ============================================================================
ARG BASE_DIGEST=python:3.11-slim-bookworm
FROM ${BASE_DIGEST} AS builder

# Build deps required for numpy/numba/orjson/uvloop wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements early for cache hits
COPY requirements.txt .

# Resolver-safe installs for aiodns/pycares, with uv cache
RUN --mount=type=cache,target=/root/.cache/uv,type=gha \
    uv pip install --compile --no-cache \
    pycares==4.4.0 \
    aiodns==3.1.1

# Main dependency install (numpy/numba/etc.)
RUN --mount=type=cache,target=/root/.cache/uv,type=gha \
    uv pip install --compile --no-cache -r requirements.txt

# ============================================================================
# Stage 2: Runtime - minimal but compiled-lib safe
# ============================================================================
FROM ${BASE_DIGEST} AS runtime

# Only the essentials; include libgomp + libstdc++ for numba/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy source code and config
COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./

# Environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

# Security: non-root + explicit permissions
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

USER botuser

ENTRYPOINT ["python", "-u", "wrapper.py"]
