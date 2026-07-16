# =============================================================================
# MULTI-STAGE BUILD: Aggressive Caching + UV + AOT Compilation (HYBRID OPTIMIZED)
# =============================================================================

# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.11-slim-bookworm AS uv-installer

# Install UV in isolated stage (cached across builds)
RUN pip install --no-cache-dir uv==0.5.15


# ---------- STAGE 2: DEPENDENCIES BUILDER ----------
FROM python:3.11-slim-bookworm AS deps-builder

# Copy UV from installer stage
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
COPY --from=uv-installer /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# ✅ Minimal build dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /build

# ✅ OPTIMIZATION: Use BuildKit cache for UV downloads
ENV UV_CACHE_DIR='/tmp/uv_cache'
COPY requirements.txt .
RUN --mount=type=cache,target=/tmp/uv_cache \
    uv pip install --system --no-cache -r requirements.txt && \
    python -m compileall -q -o 1 /usr/local/lib/python3.11/site-packages


# ---------- STAGE 3: AOT COMPILER ----------
FROM deps-builder AS aot-builder

WORKDIR /build

# ✅ Copy in order of change frequency (maximize cache hits)
COPY src/numba_functions_shared.py ./
COPY src/aot_bridge.py ./
COPY src/aot_build.py ./
COPY src/macd_unified.py ./

# ✅ Verify files exist before compilation
RUN ls -la *.py && \
    test -f numba_functions_shared.py || (echo "❌ Missing numba_functions_shared.py" && exit 1) && \
    test -f aot_build.py || (echo "❌ Missing aot_build.py" && exit 1) && \
    test -f aot_bridge.py || (echo "❌ Missing aot_bridge.py" && exit 1) && \
    test -f macd_unified.py || (echo "❌ Missing macd_unified.py" && exit 1)

# ✅ Safe AOT Fallback handling logic (Strict vs. Graceful JIT fallback)
ARG AOT_STRICT=0

RUN echo "🔨 Starting AOT compilation..." && \
    if python aot_build.py --output-dir /build --module-name macd_aot_compiled --verify; then \
        echo "📂 Listing build outputs..." && ls -lh /build && \
        echo "🔄 Normalizing compiled filename..." && \
        SO_FILE=$(ls -1 /build/macd_aot_compiled*.so 2>/dev/null | head -1) && \
        if [ -n "$SO_FILE" ]; then \
            mv "$SO_FILE" /build/macd_aot_compiled.so && \
            echo "✅ AOT binary verified and normalized: /build/macd_aot_compiled.so" && \
            python -c "import importlib.util; \
spec=importlib.util.spec_from_file_location('macd_aot_compiled','/build/macd_aot_compiled.so'); \
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
print('✅ Dynamic load check passed')" || exit 1; \
        else \
            echo "❌ No matching AOT library found despite successful build script execution" && exit 1; \
        fi; \
    else \
        echo "⚠️ AOT Build execution failed!"; \
        if [ "$AOT_STRICT" = "1" ]; then \
            echo "❌ AOT STRICT mode active. Aborting build." && exit 1; \
        else \
            echo "⚠️ Fallback to JIT enabled. Creating dummy placeholder .so to prevent COPY errors..." && \
            touch /build/macd_aot_compiled.so; \
        fi; \
    fi


# ---------- STAGE 4: FINAL RUNTIME ----------
FROM python:3.11-slim-bookworm AS final

# ✅ Explicitly disable healthcheck to save CPU cycles
HEALTHCHECK NONE

# ✅ Only essential runtime dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libtbb12 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ✅ Copy UV binary
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv

# ✅ Security - Non-root user
RUN useradd --uid 1000 --no-log-init -m appuser && \
    mkdir -p /app/src && \
    chown -R appuser:appuser /app

WORKDIR /app/src

# ✅ Copy Python dependencies from deps-builder
COPY --from=deps-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# ✅ Copy AOT binary from aot-builder (Always exists: compiled or placeholder)
COPY --from=aot-builder --chown=appuser:appuser /build/macd_aot_compiled.so ./

# ✅ Copy source files in order of change frequency
COPY --chown=appuser:appuser src/numba_functions_shared.py ./
COPY --chown=appuser:appuser src/aot_bridge.py ./
COPY --chown=appuser:appuser src/macd_unified.py ./

USER appuser

# ✅ Environment optimization with deterministic threading (PYTHONOPTIMIZE=1 is safe for Numba JIT)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_WARNINGS=0 \
    NUMBA_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MEMORY_LIMIT_BYTES=850000000 \
    TZ=Asia/Kolkata \
    AOT_LIB_PATH=/app/src

# Labels for metadata
LABEL org.opencontainers.image.title="MACD Unified Bot (AOT)" \
      org.opencontainers.image.description="High-performance trading alert bot with AOT compilation" \
      org.opencontainers.image.source="https://github.com/manojpy/github-cron" \
      org.opencontainers.image.memory_limit="900MB" \
      org.opencontainers.image.platform="linux/amd64"

# ✅ Run bot WITH safe level-1 optimization (-O flag removes assert statement runtime cost)
CMD ["python", "-O", "macd_unified.py"]
