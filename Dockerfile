# =============================================================================
# MULTI-STAGE BUILD: Aggressive Caching + UV + AOT Compilation (HARDENED)
# - Explicit cache busting via REBUILD_TS
# - Strict AOT artifact normalization and import verification
# - Runtime JIT-fallback guard (NUMBA_DISABLE_JIT check)
# =============================================================================

# ---------- GLOBAL CACHE BUST ARG ----------
ARG REBUILD_TS="unset"

# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.11-slim-bookworm AS uv-installer
ARG REBUILD_TS
ENV REBUILD_TS=${REBUILD_TS}

# Install UV in isolated stage (cached across builds)
RUN echo "Cache bust token (uv-installer): ${REBUILD_TS}" && \
    pip install --no-cache-dir uv==0.5.15

# ---------- STAGE 2: DEPENDENCIES BUILDER ----------
FROM python:3.11-slim-bookworm AS deps-builder
ARG REBUILD_TS
ENV REBUILD_TS=${REBUILD_TS}

# Copy UV from installer stage (binary + its site-packages)
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
COPY --from=uv-installer /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Minimal build dependencies
RUN echo "Cache bust token (deps-builder): ${REBUILD_TS}" && \
    apt-get update -qq && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /build

# Install dependencies with AOT in mind
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt && \
    python -m compileall -q /usr/local/lib/python3.11/site-packages

# ---------- STAGE 3: AOT COMPILER ----------
FROM deps-builder AS aot-builder
ARG REBUILD_TS
ENV REBUILD_TS=${REBUILD_TS}

WORKDIR /build

# Copy in order of change frequency (maximize cache hits)
COPY src/numba_functions_shared.py ./
COPY src/aot_bridge.py ./
COPY src/aot_build.py ./
COPY src/macd_unified.py ./

# Verify files exist before compilation
RUN ls -la *.py && \
    test -f numba_functions_shared.py || (echo "❌ Missing numba_functions_shared.py" && exit 1) && \
    test -f aot_build.py || (echo "❌ Missing aot_build.py" && exit 1)

# AOT Compilation with strict verification and normalized filename
ARG AOT_STRICT=1
RUN echo "🔨 Starting AOT compilation (token: ${REBUILD_TS})..." && \
    python aot_build.py --output-dir /build --module-name macd_aot_compiled --verify || \
      (echo "❌ AOT build script failed" && exit 1) && \
    echo "📂 Listing build outputs..." && ls -lh /build && \
    echo "🔍 Normalizing compiled filename..." && \
    sh -c 'set -e; \
      so=$(ls -1 /build/macd_aot_compiled*.so 2>/dev/null | head -n1); \
      [ -n "$so" ] || { echo "❌ No .so produced"; exit 1; }; \
      mv "$so" /build/macd_aot_compiled.so' && \
    python -c "import importlib.util; \
spec=importlib.util.spec_from_file_location('macd_aot_compiled','/build/macd_aot_compiled.so'); \
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
print('✅ AOT binary verified')" || \
    ( [ "$AOT_STRICT" != "1" ] && echo "⚠️ AOT failed, continuing..." || (echo "❌ AOT STRICT mode: Compilation failed" && exit 1) )

# ---------- STAGE 4: FINAL RUNTIME ----------
FROM python:3.11-slim-bookworm AS final
ARG REBUILD_TS
ENV REBUILD_TS=${REBUILD_TS}

# Only essential runtime dependencies
RUN echo "Cache bust token (final): ${REBUILD_TS}" && \
    apt-get update -qq && apt-get install -y --no-install-recommends \
      libtbb12 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy UV binary (lightweight)
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv

# Security - Non-root user with minimal permissions
RUN useradd --uid 1000 --no-log-init -m appuser && \
    mkdir -p /app/src && \
    chown -R appuser:appuser /app

WORKDIR /app/src

# Copy Python dependencies from deps-builder (cached layer)
COPY --from=deps-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy AOT binary from aot-builder (explicit, deterministic)
COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled.so ./

# Copy in order of change frequency (maximize cache hits)
COPY --chown=appuser:appuser src/numba_functions_shared.py ./
COPY --chown=appuser:appuser src/aot_bridge.py ./
# aot_build.py is not needed at runtime; omit to keep image minimal
COPY --chown=appuser:appuser src/macd_unified.py ./

# Config copied at runtime (kept here for current design)
COPY --chown=appuser:appuser config_macd.json ./

USER appuser

# Environment optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_WARNINGS=0 \
    PYTHONOPTIMIZE=1 \
    MEMORY_LIMIT_BYTES=850000000 \
    TZ=Asia/Kolkata \
    AOT_LIB_PATH=/app/src

# Build metadata
LABEL org.opencontainers.image.title="MACD Unified Bot (AOT)" \
      org.opencontainers.image.description="High-performance trading alert bot with AOT compilation" \
      org.opencontainers.image.source="https://github.com/manojpy/github-cron" \
      org.opencontainers.image.memory_limit="900MB" \
      org.opencontainers.image.platform="linux/amd64" \
      org.opencontainers.image.rebuild_ts="${REBUILD_TS}"

# Runtime JIT-fallback guard: import with JIT disabled to ensure AOT suffices
# This is a lightweight self-test during build; it won't run at container start.
RUN NUMBA_DISABLE_JIT=1 python -c "import importlib.util; \
spec=importlib.util.spec_from_file_location('macd_aot_compiled','/app/src/macd_aot_compiled.so'); \
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
print('✅ Runtime import check passed (no JIT)')"

CMD ["python", "macd_unified.py"]
