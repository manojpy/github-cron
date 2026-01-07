# =============================================================================
# ULTRA-OPTIMIZED FOR 40s COLD START
# =============================================================================
# ✅ Preserved: UV installer, AOT, security
# ✅ Removed: Redundant verifications, debug symbols, unused labels
# ✅ Merged: RUN layers to reduce overlayfs overhead
# =============================================================================

# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.11-slim-bookworm AS uv-installer
RUN pip install --no-cache-dir uv==0.5.15

# ---------- STAGE 2: BASE DEPS ----------
FROM python:3.11-slim-bookworm AS base-deps
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /build

# ---------- STAGE 3: PYTHON DEPS ----------
FROM base-deps AS python-deps
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt && \
    python -m compileall -q /usr/local/lib/python3.11/site-packages

# ---------- STAGE 4: AOT BUILD ----------
FROM python-deps AS aot-builder
WORKDIR /build
COPY src/numba_functions_shared.py src/aot_bridge.py src/aot_build.py ./
RUN python aot_build.py --output-dir /build --module-name macd_aot_compiled && \
    cp /build/macd_aot_compiled*.so /build/macd_aot_compiled.so

# ---------- STAGE 5: FINAL ----------
FROM python:3.11-slim-bookworm
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends libtbb12 ca-certificates && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
RUN useradd --uid 1000 --no-log-init -m appuser && \
    mkdir -p /app/src && \
    chown -R appuser:appuser /app

WORKDIR /app/src
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled.so ./
COPY --chown=appuser:appuser src/*.py ./
COPY --chown=appuser:appuser config_macd.json ./

USER appuser
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MEMORY_LIMIT_BYTES=700000000 \
    TZ=Asia/Kolkata \
    AOT_LIB_PATH=/app/src

CMD ["python", "macd_unified.py"]
