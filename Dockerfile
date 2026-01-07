# =============================================================================
# MULTI-STAGE BUILD: Aggressive Caching + UV + AOT Compilation (OPTIMIZED)
# Requires: BuildKit enabled
# =============================================================================

# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.11-slim-bookworm AS uv-installer

# Install UV via pip (binary only, no site-packages copied to final stages)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir uv==0.5.15


# ---------- STAGE 2: DEPENDENCIES BUILDER ----------
FROM python:3.11-slim-bookworm AS deps-builder

# Copy UV binary from installer stage
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv

# ✅ OPTIMIZED: Minimal build dependencies + apt cache
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /build

# ✅ OPTIMIZED: Install dependencies with cache, compile site-packages
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-cache -r requirements.txt && \
    python -m compileall -q /usr/local/lib/python3.11/site-packages


# ---------- STAGE 3: AOT COMPILER ----------
FROM deps-builder AS aot-builder

WORKDIR /build

# ✅ OPTIMIZED: Copy in order of change frequency (maximize cache hits)
COPY src/numba_functions_shared.py ./
COPY src/aot_bridge.py ./
COPY src/aot_build.py ./
COPY src/macd_unified.py ./

# ✅ OPTIMIZED: Verify files exist before compilation
RUN ls -la *.py && \
    test -f numba_functions_shared.py || (echo "❌ Missing numba_functions_shared.py" && exit 1) && \
    test -f aot_build.py || (echo "❌ Missing aot_build.py" && exit 1)

# ✅ OPTIMIZED: AOT Compilation with strict verification + strip to reduce size
ARG AOT_STRICT=1
RUN echo "🔨 Starting AOT compilation..." && \
    python aot_build.py --output-dir /build --module-name macd_aot_compiled --verify || \
      (echo "❌ AOT build script failed" && exit 1) && \
    echo "📂 Listing build outputs..." && ls -lh /build && \
    echo "🔍 Normalizing compiled filename..." && \
    python - <<'PY'
import glob, os, shutil
files = sorted(glob.glob("/build/macd_aot_compiled*.so"))
if not files:
    raise SystemExit("No compiled .so found")
# Normalize name deterministically
shutil.copy2(files[0], "/build/macd_aot_compiled.so")
PY
# Install binutils temporarily to strip the .so (smaller final image layer)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq && apt-get install -y --no-install-recommends binutils && \
    strip --strip-unneeded /build/macd_aot_compiled.so && \
    apt-get purge -y binutils && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Verify importability of the stripped AOT binary
RUN python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    'macd_aot_compiled', '/build/macd_aot_compiled.so'
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('✅ AOT binary verified')
PY


# ---------- STAGE 4: FINAL RUNTIME ----------
FROM python:3.11-slim-bookworm AS final

# ✅ OPTIMIZED: Only essential runtime dependencies (+ apt cache)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq && apt-get install -y --no-install-recommends \
      libtbb12 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ✅ OPTIMIZED: Security - Non-root user with minimal permissions
RUN useradd --uid 1000 --no-log-init -m appuser && \
    mkdir -p /app/src && \
    chown -R appuser:appuser /app

WORKDIR /app/src

# ✅ OPTIMIZED: Copy Python dependencies from deps-builder (cached layer)
COPY --from=deps-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# ✅ OPTIMIZED: Copy AOT binary from aot-builder (explicit, deterministic)
COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled.so ./

# ✅ OPTIMIZED: Copy in order of change frequency (maximize cache hits)
COPY --chown=appuser:appuser src/numba_functions_shared.py ./
COPY --chown=appuser:appuser src/aot_bridge.py ./
COPY --chown=appuser:appuser src/aot_build.py ./
COPY --chown=appuser:appuser src/macd_unified.py ./

# NOTE: Config is volume-mounted at runtime; do not bake into image
# (keeps image lean and avoids stale config)
# COPY config_macd.json ./

USER appuser

# ✅ OPTIMIZED: Environment optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_WARNINGS=0 \
    PYTHONOPTIMIZE=1 \
    MEMORY_LIMIT_BYTES=850000000 \
    TZ=Asia/Kolkata \
    AOT_LIB_PATH=/app/src

# Labels for metadata
LABEL org.opencontainers.image.title="MACD Unified Bot (AOT)" \
      org.opencontainers.image.description="High-performance trading alert bot with AOT compilation" \
      org.opencontainers.image.source="https://github.com/manojpy/github-cron" \
      org.opencontainers.image.memory_limit="900MB" \
      org.opencontainers.image.platform="linux/amd64"

CMD ["python", "macd_unified.py"]
