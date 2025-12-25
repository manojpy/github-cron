# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

# ✅ Combine apt operations and add --quiet flags
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ✅ Use specific uv version for reproducibility
RUN pip install --quiet --no-cache-dir uv==0.1.18

# Copy requirements first for cache efficiency
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir --quiet -r requirements.txt

# Copy source code and config
COPY src ./src
COPY config_macd.json ./config_macd.json

WORKDIR /build/src

# ✅ Build AOT with less verbose output
ARG AOT_STRICT=1
RUN python aot_build.py 2>&1 | grep -E "(ERROR|WARNING|succeeded|failed)" || \
    if [ "$AOT_STRICT" = "1" ]; then \
        echo "AOT artifact missing" && exit 1; \
    else \
        echo "JIT fallback allowed (dev build)"; \
    fi


# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# ✅ Install runtime libraries quietly
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src

# Copy Python runtime
COPY --from=builder /usr/local /usr/local

# Copy all source code (including .so file)
COPY --from=builder /build/src /app/src
COPY --from=builder /build/config_macd.json /app/src/config_macd.json

# ✅ Add optimization flags
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=2 \
    NUMBA_DISABLE_JIT=0 \
    NUMBA_WARNINGS=0

# Run from /app/src where everything is located
CMD ["python", "-u", "-m", "macd_unified"]