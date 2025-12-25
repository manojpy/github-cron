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

# Build AOT with detailed output
ARG AOT_STRICT=1
RUN python aot_build.py || \
    if [ "$AOT_STRICT" = "1" ]; then \
        echo "AOT artifact missing" && exit 1; \
    else \
        echo "JIT fallback allowed (dev build)"; \
    fi

# Diagnostic: Show what was created
RUN echo "=== AOT Build Results ===" && \
    ls -lh __pycache__/macd_aot_compiled*.so && \
    echo "========================"

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# Install runtime libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src

# Copy Python runtime
COPY --from=builder /usr/local /usr/local

# Copy all source code (including __pycache__ with .so files)
COPY --from=builder /build/src /app/src
COPY --from=builder /build/config_macd.json /app/src/config_macd.json

# Verify .so file is present
RUN echo "=== Verifying AOT artifact in final image ===" && \
    ls -lh /app/src/__pycache__/macd_aot_compiled*.so && \
    echo "Config file location:" && \
    ls -lh /app/src/config_macd.json && \
    echo "============================================="

# Runtime environment
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=2

# Run from /app/src where everything is located
CMD ["python", "-m", "macd_unified"]