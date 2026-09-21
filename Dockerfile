# =============================================================================
# MULTI-STAGE BUILD: Aggressive Caching + UV + AOT Compilation (HYBRID OPTIMIZED)
# =============================================================================

# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.11-slim-bookworm AS uv-installer

RUN pip install --no-cache-dir uv==0.6.12

# ---------- STAGE 2: DEPENDENCIES BUILDER ----------
FROM python:3.11-slim-bookworm AS deps-builder

COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /build

# Create virtual environment for clean multi-stage copying
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Enable BuildKit caching by removing --no-cache
ENV UV_CACHE_DIR='/tmp/uv_cache'
COPY requirements.txt .
RUN --mount=type=cache,target=/tmp/uv_cache \
    uv pip install -r requirements.txt && \
    python -m compileall -q -o 2 $VIRTUAL_ENV

# ---------- STAGE 3: CYTHON COMPILER ----------
FROM deps-builder AS cython-builder

WORKDIR /build

COPY setup.py ./
COPY src/cython_functions.pyx ./src/

ARG CYTHON_STRICT=0

RUN set -e; \
    echo "🔨 Starting Cython compilation..."; \
    if python setup.py build_ext --inplace; then \
        echo "✅ Cython build successful"; \
    else \
        echo "⚠️ Cython compilation failed!"; \
        if [ "$CYTHON_STRICT" = "1" ]; then \
            echo "❌ CYTHON_STRICT=1: Aborting build."; \
            exit 1; \
        else \
            echo "⚠️ CYTHON_STRICT=0: continuing — bot will use Numba JIT fallback."; \
        fi; \
    fi

# ---------- STAGE 4: FINAL RUNTIME ----------
FROM python:3.11-slim-bookworm AS final

HEALTHCHECK NONE

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libtbb12 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security - Non-root user
RUN useradd --uid 1000 --no-log-init -m appuser && \
    mkdir -p /app/src && \
    chown -R appuser:appuser /app

WORKDIR /app/src

# Copy Virtual Environment from deps-builder
COPY --from=deps-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy compiled Cython extension (filename carries the ABI tag, e.g. .cpython-311-x86_64-linux-gnu.so)
COPY --from=cython-builder --chown=appuser:appuser /build/cython_functions*.so ./

# Copy AOT / bridge files (change rarely — keep early for layer cache)
COPY --chown=appuser:appuser src/aot_meta.py ./
COPY --chown=appuser:appuser src/numba_functions_shared.py ./
COPY --chown=appuser:appuser src/aot_bridge.py ./
COPY --chown=appuser:appuser src/numeric_selftest.py ./

# Copy business logic modules (change frequently)
COPY --chown=appuser:appuser src/bot_config.py ./
COPY --chown=appuser:appuser src/state.py ./
COPY --chown=appuser:appuser src/fetcher.py ./
COPY --chown=appuser:appuser src/indicators.py ./
COPY --chown=appuser:appuser src/gates.py ./
COPY --chown=appuser:appuser src/alerts.py ./
COPY --chown=appuser:appuser src/threshold_engine.py ./
COPY --chown=appuser:appuser src/brain.py ./
COPY --chown=appuser:appuser src/brain_enhanced.py ./
COPY --chown=appuser:appuser src/brain_audit.py ./
COPY --chown=appuser:appuser src/repair_ledger.py ./
COPY --chown=appuser:appuser src/apply_config_override.py ./
COPY --chown=appuser:appuser src/outcome_storage.py ./
COPY --chown=appuser:appuser src/archive_reader.py ./
COPY --chown=appuser:appuser src/macd_unified.py ./

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_WARNINGS=0 \
    
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MEMORY_LIMIT_BYTES=850000000 \
    TZ=Asia/Kolkata

LABEL org.opencontainers.image.title="MACD Unified Bot (AOT)" \
      org.opencontainers.image.description="High-performance trading alert bot with AOT compilation" \
      org.opencontainers.image.source="https://github.com/manojpy/github-cron" \
      org.opencontainers.image.memory_limit="900MB" \
      org.opencontainers.image.platform="linux/amd64"

# Let PYTHONOPTIMIZE=2 control optimization level; do not override with -O/-OO
CMD ["python", "macd_unified.py"]
