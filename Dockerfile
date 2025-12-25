# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

# Install compilers and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv via pip (simpler than curl script)
RUN pip install uv

# Copy requirements first for cache efficiency
COPY requirements.txt .

RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code last
COPY src ./src

WORKDIR /build/src

# Hybrid AOT build step:
# - In CI/CD (AOT_STRICT=1), fail hard if artifact missing
# - In local/dev, allow JIT fallback
ARG AOT_STRICT=0
RUN python aot_build.py || \
    if [ "$AOT_STRICT" = "1" ]; then \
        echo "❌ AOT artifact missing" && exit 1; \
    else \
        echo "⚠️ JIT fallback allowed (dev build)"; \
    fi

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# Install runtime libraries only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python runtime and compiled artifacts
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/src /app/src

# Ensure Python can find src/ modules
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=2


# Default command: run macd_unified
CMD ["python", "-m", "macd_unified"]
