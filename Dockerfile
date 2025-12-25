# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

# Install compilers and build tools
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv for faster pip installs
RUN pip install --quiet --no-cache-dir uv

# Copy requirements first for cache efficiency
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir --quiet -r requirements.txt

# Copy source code and config
COPY src ./src
COPY config_macd.json ./config_macd.json

WORKDIR /build/src

# ✅ FIXED: Build AOT without grep filtering that breaks the logic
ARG AOT_STRICT=1
RUN python aot_build.py && \
    ls -lh macd_aot_compiled*.so 2>/dev/null || \
    if [ "$AOT_STRICT" = "1" ]; then \
        echo "ERROR: AOT artifact missing" && exit 1; \
    else \
        echo "JIT fallback allowed (dev build)"; \
    fi


# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# Install runtime libraries quietly
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src

# Copy Python runtime
COPY --from=builder /usr/local /usr/local

# Copy all source code (including .so file)
COPY --from=builder /build/src /app/src
COPY --from=builder /build/config_macd.json /app/src/config_macd.json

# Runtime environment with optimizations
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=2 \
    NUMBA_WARNINGS=0

# Run from /app/src where everything is located
CMD ["python", "-u", "-m", "macd_unified"]