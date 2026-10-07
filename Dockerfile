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

ARG CYTHON_STRICT=1

RUN set -e; \
    echo "🔨 Starting Cython compilation..."; \
    if python setup.py build_ext --inplace; then \
        echo "✅ Cython build successful"; \
    else \
        echo "❌ Cython compilation failed."; \
        echo "❌ Production image requires the Cython backend."; \
        exit 1; \
    fi

# ---------- STAGE 3b: RUNTIME VENV (pruned copy of the build venv) ----------
# The build venv carries things the running bot never imports. Cutting them
# shrinks the image the runner has to pull on every 15-minute run.
FROM deps-builder AS runtime-venv

# KEEP_JIT_FALLBACK=1 keeps numba + llvmlite + tbb (about 165 MB) so the slow
# Numba JIT can take over if the compiled Cython module ever fails to load.
# The build already refuses to produce an image without Cython (CYTHON_STRICT)
# and the smoke test in build.yml requires AOT, so the default drops them.
ARG KEEP_JIT_FALLBACK=0

RUN set -e; \
    SP="$VIRTUAL_ENV/lib/python3.11/site-packages"; \
    uv pip uninstall cython setuptools wheel py-cpuinfo; \
    if [ "$KEEP_JIT_FALLBACK" != "1" ]; then \
        uv pip uninstall numba llvmlite tbb; \
    fi; \
    rm -rf "$SP"/Cython "$SP"/cython* "$SP"/pyximport "$SP"/setuptools* "$SP"/pkg_resources \
           "$SP"/_distutils_hack "$SP"/wheel* "$SP"/cpuinfo "$SP"/py_cpuinfo*; \
    if [ "$KEEP_JIT_FALLBACK" != "1" ]; then \
        rm -rf "$SP"/numba* "$SP"/llvmlite* "$SP"/tbb* "$SP"/TBB*; \
    fi; \
    find "$SP/numpy" -type d -name tests -prune -exec rm -rf {} +; \
    echo "runtime venv: $(du -sm "$VIRTUAL_ENV" | cut -f1) MB"

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
COPY --from=runtime-venv /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy compiled Cython extension (filename carries the ABI tag, e.g. .cpython-311-x86_64-linux-gnu.so)
COPY --from=cython-builder --chown=appuser:appuser /build/cython_functions*.so ./

# All application modules in ONE layer (was ~40 separate layers: every layer
# is its own registry round trip when the runner pulls the image).
COPY --chown=appuser:appuser src/*.py ./

# Pre-compile the app modules. The container runs --read-only with
# PYTHONDONTWRITEBYTECODE=1, so without this Python recompiles every module
# from source on every single run. -o 2 matches PYTHONOPTIMIZE=2 below.
RUN python -m compileall -q -o 2 /app/src

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
