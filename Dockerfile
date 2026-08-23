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

# ---------- STAGE 3: AOT COMPILER ----------
FROM deps-builder AS aot-builder

WORKDIR /build

# Only copy files the compiler actually imports.
# Do NOT copy aot_bridge.py or macd_unified.py here — changing business logic
# should not invalidate the AOT compilation cache.
COPY src/aot_version.py ./
COPY src/aot_function_registry.py ./
COPY src/numba_functions_shared.py ./
COPY src/aot_build.py ./

ARG AOT_STRICT=0

# Clean structured shell block to correctly handle AOT_STRICT fallbacks
RUN set -e; \
    echo "🔨 Starting AOT compilation..."; \
    if python aot_build.py --output-dir /build --module-name macd_aot_compiled --verify; then \
        echo "✅ AOT build successful"; \
        SO_FILE=$(ls -1 /build/macd_aot_compiled*.so 2>/dev/null | head -1); \
        if [ -n "$SO_FILE" ]; then \
            mv "$SO_FILE" /build/macd_aot_compiled.so; \
        fi; \
    else \
        echo "⚠️ AOT compilation failed!"; \
        if [ "$AOT_STRICT" = "1" ]; then \
            echo "❌ AOT_STRICT=1: Aborting build."; \
            exit 1; \
        else \
            echo "⚠️ AOT_STRICT=0: Creating empty stub files for JIT fallback..."; \
            touch /build/macd_aot_compiled.so /build/macd_aot_compiled.version; \
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

# Copy AOT binary + version stamp from aot-builder
COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled.so ./
COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled.version ./

# Copy AOT / bridge files (change rarely — keep early for layer cache)
COPY --chown=appuser:appuser src/aot_version.py ./
COPY --chown=appuser:appuser src/aot_function_registry.py ./
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
    TZ=Asia/Kolkata \
    AOT_LIB_PATH=/app/src

LABEL org.opencontainers.image.title="MACD Unified Bot (AOT)" \
      org.opencontainers.image.description="High-performance trading alert bot with AOT compilation" \
      org.opencontainers.image.source="https://github.com/manojpy/github-cron" \
      org.opencontainers.image.memory_limit="900MB" \
      org.opencontainers.image.platform="linux/amd64"

# Let PYTHONOPTIMIZE=2 control optimization level; do not override with -O/-OO
CMD ["python", "macd_unified.py"]
