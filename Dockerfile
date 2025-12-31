# =============================================================================
# MULTI-STAGE BUILD: Aggressive Caching + UV + AOT Compilation
# =============================================================================

# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.12-slim-bookworm AS uv-installer

# Install UV in isolated stage (cached across builds)
RUN pip install --no-cache-dir uv==0.5.15

# ---------- STAGE 2: DEPENDENCIES BUILDER ----------
FROM python:3.12-slim-bookworm AS deps-builder

# Copy UV from installer stage
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
COPY --from=uv-installer /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Install build essentials (minimal, cached layer)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /build

# Layer 1: Install dependencies ONLY (most cacheable)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt && \
    python -m compileall -q /usr/local/lib/python3.12/site-packages

# ---------- STAGE 3: AOT COMPILER ----------
FROM deps-builder AS aot-builder


# ✅ FIXED (separate COPYs)
COPY src/aot_bridge.py ./
COPY src/aot_build.py ./
COPY src/numba_functions_shared.py ./

WORKDIR /build

# AOT Compilation with strict verification
ARG AOT_STRICT=1
RUN echo "🔨 Starting AOT compilation..." && \
    python aot_build.py && \
    ls -lh macd_aot_compiled*.so && \
    python -c "import macd_aot_compiled; print('✅ AOT binary verified')" || \
    ( [ "$AOT_STRICT" != "1" ] && echo "⚠️ AOT failed, continuing..." || (echo "❌ AOT STRICT mode: Compilation failed" && exit 1) )

# ---------- STAGE 4: FINAL RUNTIME ----------
FROM python:3.12-slim-bookworm AS final

# Runtime dependencies (minimal)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libtbb12 \
    tzdata \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy UV binary (for potential runtime use)
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv

# Security: Non-root user
RUN useradd --uid 1000 --no-log-init -m appuser && \
    mkdir -p /app/src /app/logs && \
    chown -R appuser:appuser /app

WORKDIR /app/src

# Copy Python dependencies from deps-builder (cached)
COPY --from=deps-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy AOT binary from aot-builder
COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled*.so ./

# Copy application source files (single layer for efficiency)
COPY --chown=appuser:appuser src/*.py ./

# Copy config template (will be overridden at runtime by volume mount)
COPY --chown=appuser:appuser config_macd.json ./

USER appuser

# Environment optimization

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_WARNINGS=0 \
    PYTHONOPTIMIZE=1 \
    NUMBA_DISABLE_JIT=1 \
    MEMORY_LIMIT_BYTES=850000000


# Health check to verify AOT compilation
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=2 \
  CMD python -c "import aot_bridge; assert aot_bridge.is_using_aot()" || exit 1

# Labels for metadata
LABEL org.opencontainers.image.title="MACD Unified Bot (AOT)" \
      org.opencontainers.image.description="High-performance trading alert bot with AOT compilation" \
      org.opencontainers.image.source="https://github.com/manojpy/github-cron" \
      org.opencontainers.image.memory_limit="1GB"

CMD ["python", "macd_unified.py"]
