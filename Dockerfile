# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

# Install compilers and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv for faster pip installs
RUN pip install uv

# Copy requirements first for cache efficiency
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src ./src
COPY config_macd.json ./config_macd.json

WORKDIR /build/src

# OPTION 1: Try AOT with fallback to JIT (current)
ARG AOT_STRICT=1
RUN python aot_build.py || \
    if [ "$AOT_STRICT" = "1" ]; then \
        echo "AOT artifact missing" && exit 1; \
    else \
        echo "JIT fallback allowed (dev build)"; \
    fi

# OPTION 2: Disable AOT completely (uncomment to use)
# Just comment out the entire RUN python aot_build.py block above
# and the bot will use JIT automatically

# Diagnostic: list everything under /build/src so logs show artifact location
RUN ls -R /build/src

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# Install runtime libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python runtime and compiled artifacts
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/src /app/src
COPY --from=builder /build/config_macd.json /app/config_macd.json

# Copy .so file with platform-specific name (e.g., macd_aot_compiled.cpython-311-x86_64-linux-gnu.so)
COPY --from=builder /build/src/__pycache__/macd_aot_compiled*.so /app/src/__pycache__/

# Runtime environment
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=2

CMD ["python", "-m", "macd_unified"]