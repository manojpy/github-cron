# =============================================================================
# MULTI-STAGE BUILD: Aggressive Caching + UV + AOT Compilation (HYBRID OPTIMIZED)
# =============================================================================
# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.11-slim-bookworm AS uv-installer
RUN pip install --no-cache-dir uv==0.5.15

# ---------- STAGE 2: DEPENDENCIES BUILDER ----------
FROM python:3.11-slim-bookworm AS deps-builder
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
COPY --from=uv-installer /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /build
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt && \
    python -m compileall -q -o 2 /usr/local/lib/python3.11/site-packages

# Prune .py files (not needed with PYTHONOPTIMIZE=2 + AOT)
RUN find /usr/local/lib/python3.11/site-packages -name "*.py" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ---------- STAGE 3: AOT COMPILER ----------
FROM deps-builder AS aot-builder
WORKDIR /build
COPY src/numba_functions_shared.py ./
COPY src/aot_bridge.py ./
COPY src/aot_build.py ./
COPY src/macd_unified.py ./

RUN ls -la *.py && \
    test -f numba_functions_shared.py || (echo "❌ Missing numba_functions_shared.py" && exit 1) && \
    test -f aot_build.py || (echo "❌ Missing aot_build.py" && exit 1)

ARG AOT_STRICT=1
RUN echo "🔨 Starting AOT compilation..." && \
    python -O aot_build.py --output-dir /build --module-name macd_aot_compiled --verify || \
    (echo "❌ AOT build script failed" && exit 1) && \
    mv /build/macd_aot_compiled*.so /build/macd_aot_compiled.so && \
    python -O -c "import importlib.util; \
    spec=importlib.util.spec_from_file_location('macd_aot_compiled','/build/macd_aot_compiled.so'); \
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
    print('✅ AOT binary verified')" || \
    ( [ "$AOT_STRICT" != "1" ] && echo "⚠️ AOT failed, continuing..." || (echo "❌ AOT STRICT mode: Compilation failed" && exit 1) )

# ---------- STAGE 4: FINAL RUNTIME ----------
FROM python:3.11-slim-bookworm AS final
HEALTHCHECK NONE

# Only essential runtime deps
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libtbb12 ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy UV
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv

# Security: non-root user with no shell
RUN useradd --uid 1000 --no-log-init --home-dir /app --shell /sbin/nologin appuser && \
    mkdir -p /app/src && chown -R appuser:appuser /app

WORKDIR /app/src

# Copy deps and AOT
COPY --from=deps-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled.so ./

# ONLY copy runtime-needed source files
COPY --chown=appuser:appuser src/macd_unified.py ./
COPY --chown=appuser:appuser src/aot_bridge.py ./

USER appuser

# Optimized env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_WARNINGS=0 \
    NUMBA_NUM_THREADS=2 \
    OMP_NUM_THREADS=1 \
    MEMORY_LIMIT_BYTES=850000000 \
    TZ=Asia/Kolkata \
    AOT_LIB_PATH=/app/src

LABEL org.opencontainers.image.title="MACD Unified Bot (AOT)" \
      org.opencontainers.image.description="High-performance trading alert bot with AOT compilation" \
      org.opencontainers.image.source="https://github.com/manojpy/github-cron" \
      org.opencontainers.image.memory_limit="900MB" \
      org.opencontainers.image.platform="linux/amd64"

CMD ["python", "macd_unified.py"]