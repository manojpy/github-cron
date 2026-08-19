# FILE 1: Dockerfile (CORRECTED)
# ============================================================================

FROM python:3.11-slim-bookworm AS uv-installer

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv==0.5.15


FROM python:3.11-slim-bookworm AS deps-builder

COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
COPY --from=uv-installer /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /build

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r requirements.txt && \
    python -m compileall -q /usr/local/lib/python3.11/site-packages


FROM deps-builder AS aot-builder

WORKDIR /build
COPY src/aot_version.py ./
COPY src/numba_functions_shared.py ./
COPY src/aot_bridge.py ./
COPY src/aot_build.py ./
COPY src/macd_unified.py ./

RUN ls -la *.py && \
    test -f numba_functions_shared.py || (echo "❌ Missing numba_functions_shared.py" && exit 1) && \
    test -f aot_build.py || (echo "❌ Missing aot_build.py" && exit 1)

ARG AOT_STRICT=0
RUN --mount=type=cache,target=/root/.cache/numba \
    echo "🔨 Starting AOT compilation (unoptimized build)..." && \
    python aot_build.py --output-dir /build --module-name macd_aot_compiled --verify || \
    (echo "❌ AOT build script failed" && exit 1) && \
    echo "📂 Listing build outputs..." && ls -lh /build && \
    echo "📄 Normalizing compiled filename..." && \
    mv /build/macd_aot_compiled*.so /build/macd_aot_compiled.so && \
    python -c "import importlib.util; \
spec=importlib.util.spec_from_file_location('macd_aot_compiled','/build/macd_aot_compiled.so'); \
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
print('✅ AOT binary verified')" || \
    ( [ "$AOT_STRICT" != "1" ] && echo "⚠️ AOT failed, continuing..." || (echo "❌ AOT STRICT mode: Compilation failed" && exit 1) )


FROM python:3.11-slim-bookworm AS final

HEALTHCHECK NONE

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends libtbb12 ca-certificates && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv

RUN useradd --uid 1000 --no-log-init -m appuser && \
    mkdir -p /app/src && \
    chown -R appuser:appuser /app

WORKDIR /app/src

COPY --from=deps-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled.so ./

# Copy source files
COPY --chown=appuser:appuser src/aot_version.py ./
COPY --chown=appuser:appuser src/numba_functions_shared.py ./
COPY --chown=appuser:appuser src/aot_bridge.py ./
COPY --chown=appuser:appuser src/macd_unified.py ./

# Copy minimal default config (secrets override via env vars)
COPY --chown=appuser:appuser config_macd.json ./

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    PYTHONPATH=/app/src:$PYTHONPATH \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_WARNINGS=0 \
    NUMBA_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MEMORY_LIMIT_BYTES=450000000 \
    TZ=Asia/Kolkata \
    AOT_LIB_PATH=/app/src

# Let PYTHONOPTIMIZE=2 (env, above) control optimization level; do not
# additionally pass -O/-OO here — same convention as the GitHub Actions
# build of this bot.
CMD ["python", "macd_unified.py"]