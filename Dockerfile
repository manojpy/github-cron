# =============================================================================
# MULTI-STAGE BUILD: Aggressive Caching + UV + AOT Compilation (OPTIMIZED)
# =============================================================================

# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.11-slim-bookworm AS uv-installer
RUN pip install --no-cache-dir uv==0.5.15

# ---------- STAGE 2: DEPENDENCIES BUILDER ----------
FROM python:3.11-slim-bookworm AS deps-builder
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
COPY --from=uv-installer /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# ---------- STAGE 3: AOT COMPILER ----------
FROM deps-builder AS aot-builder
WORKDIR /build

# Copy source files needed for compilation
COPY src/numba_functions_shared.py ./
COPY src/aot_build.py ./

# Create output directory
RUN mkdir -p /build/out

# ✅ FIX: Run AOT Build with strict error checking
ARG AOT_STRICT=1
RUN echo "🔨 Starting AOT compilation..." && \
    python aot_build.py --output-dir /build/out --module-name macd_aot_compiled --verify || \
    (if [ "$AOT_STRICT" = "1" ]; then echo "❌ AOT build failed"; exit 1; else echo "⚠️ AOT failed, continuing..."; fi)

# ---------- STAGE 4: FINAL RUNTIME ----------
FROM python:3.11-slim-bookworm
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/src && chown -R appuser:appuser /app

WORKDIR /app/src

# Copy dependencies and AOT binary
COPY --from=deps-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=aot-builder --chown=appuser:appuser /build/out/macd_aot_compiled*.so ./macd_aot_compiled.so

# Copy application source
COPY --chown=appuser:appuser src/numba_functions_shared.py ./
COPY --chown=appuser:appuser src/aot_bridge.py ./
COPY --chown=appuser:appuser src/macd_unified.py ./
COPY --chown=appuser:appuser config_macd.json ./

USER appuser
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    TZ=Asia/Kolkata \
    AOT_LIB_PATH=/app/src

ENTRYPOINT ["python", "macd_unified.py"]