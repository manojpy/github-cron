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

# ✅ Copy in order of change frequency:
COPY src/numba_functions_shared.py ./
COPY src/aot_build.py ./
COPY src/aot_bridge.py ./
COPY src/macd_unified.py ./

RUN ls -la *.py && \
    test -f numba_functions_shared.py || (echo "❌ Missing numba_functions_shared.py" && exit 1) && \
    test -f aot_build.py || (echo "❌ Missing aot_build.py" && exit 1)

ul

# AOT Compilation with strict verification
ARG AOT_STRICT=1
RUN echo "🔨 Starting AOT compilation..." && \
    python aot_build.py || (echo "❌ AOT build script failed" && exit 1) && \
    echo "📂 Listing build outputs..." && \
    ls -lh /build || true && \
    echo "🔍 Checking for compiled module..." && \
    find /build -maxdepth 1 -name "macd_aot_compiled*.*" -ls && \
    python -c "import importlib.util, pathlib; \
so_files=list(pathlib.Path('/build').glob('macd_aot_compiled*.so')); \
assert so_files, 'No .so file found'; \
spec=importlib.util.spec_from_file_location('macd_aot_compiled', so_files[0]); \
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
print('✅ AOT binary verified')" || \
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

# ✅ Copy in order of change frequency (maximize cache hits)
COPY --chown=appuser:appuser src/numba_functions_shared.py ./
COPY --chown=appuser:appuser src/aot_bridge.py ./
COPY --chown=appuser:appuser src/aot_build.py ./
COPY --chown=appuser:appuser src/macd_unified.py ./

COPY --chown=appuser:appuser config_macd.json ./

USER appuser

# Environment optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_WARNINGS=0 \
    PYTHONOPTIMIZE=1 \
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
