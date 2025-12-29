# =============================================================================
# MULTI-STAGE BUILD: Optimized Dockerfile with Bug Fixes
# =============================================================================

# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

# Install compilers and build tools
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv for faster pip installs
RUN pip install --quiet --no-cache-dir uv

# ✅ FIX: Handle different project layouts (root vs src/)
# Copy requirements with fallback logic
COPY requirements.txt* ./
COPY src/requirements.txt* ./src/ 2>/dev/null || true

RUN if [ -f requirements.txt ]; then \
        echo "Installing from root requirements.txt"; \
        uv pip install --system --no-cache-dir --quiet -r requirements.txt; \
    elif [ -f src/requirements.txt ]; then \
        echo "Installing from src/requirements.txt"; \
        uv pip install --system --no-cache-dir --quiet -r src/requirements.txt; \
    else \
        echo "ERROR: requirements.txt not found in root or src/" && exit 1; \
    fi

# Copy source code: ROOT/src -> /build/src
COPY src ./src 2>/dev/null || COPY . ./src

# Copy config: ROOT/config_macd.json -> /build/config_macd.json
COPY config_macd.json ./config_macd.json
     
WORKDIR /build/src

# Build AOT - Script is in src/ so this works
ARG AOT_STRICT=1
RUN python aot_build.py && \
    ls -lh macd_aot_compiled*.so 2>/dev/null || \
    if [ "$AOT_STRICT" = "1" ]; then \
        echo "❌ ERROR: AOT artifact missing" && exit 1; \
    else \
        echo "⚠️ JIT fallback allowed (dev build)"; \
    fi

# Verify AOT build
RUN if [ "$AOT_STRICT" = "1" ]; then \
        if [ ! -f macd_aot_compiled*.so ]; then \
            echo "❌ FATAL: AOT_STRICT=1 but no .so file found" && exit 1; \
        fi; \
        echo "✅ AOT artifact verified"; \
    fi

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# Install runtime dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 \
    tzdata \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app/src

# Copy Python runtime + artifacts from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/src /app/src

# ✅ Copy config with fallback (runtime may inject via volume)
COPY --from=builder /build/config_macd.json /app/src/config_macd.json 2>/dev/null || true

# Runtime config with optimizations
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=4 \
    NUMBA_WARNINGS=0 \
    NUMBA_OPT=3 \
    TZ=Asia/Kolkata \
    AIOHTTP_MAX_CONNS=16 \
    AIOHTTP_CONNS_PER_HOST=12

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ✅ Create non-root user with proper permissions
RUN useradd --uid 1000 -m appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app/logs

USER appuser

# Default command
CMD ["python", "-u", "macd_unified.py"]