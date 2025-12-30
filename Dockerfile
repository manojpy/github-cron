# =============================================================================
# MULTI-STAGE BUILD: Optimized for AOT Compilation
# =============================================================================

# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip>=25.3
RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy all source files (including aot_build.py and config_macd.json)
COPY src/ .
COPY config_macd.json . 

# ✅ AOT Build
ARG AOT_STRICT=1
RUN python aot_build.py && \
    ls -lh macd_aot_compiled*.so 2>/dev/null || \
    ( [ "$AOT_STRICT" != "1" ] || (echo "❌ AOT Compilation Failed" && exit 1) )

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 tzdata ca-certificates bc && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip>=25.3

# Setup non-root user
RUN useradd --uid 1000 -m appuser && \
    mkdir -p /app/src /app/logs && \
    chown -R appuser:appuser /app

WORKDIR /app/src

# Copy dependencies and code
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder --chown=appuser:appuser /build/ /app/src/

USER appuser

# ✅ IMPROVED VERIFICATION
# This script checks for the binary but doesn't crash if the config is missing 
# (since the config is injected by GitHub Actions at RUN time, not BUILD time).
RUN echo "Starting verification..." && \
    python -c "import macd_aot_compiled; print('✅ AOT Binary: Loaded Successfully')" && \
    if [ -f "config_macd.json" ]; then echo "✅ Config template found"; else echo "⚠️ Config template missing (Expected if using external injection)"; fi

ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PYTHONUNBUFFERED=1

CMD ["python", "macd_unified.py"]