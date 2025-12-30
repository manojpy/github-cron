# =============================================================================
# MULTI-STAGE BUILD: Optimized Dockerfile with Security Fixes
# Fixes CVE-2025-8869 (pip symbolic link vulnerability)
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

# ✅ FIX CVE-2025-8869: Upgrade pip to 25.3+ FIRST
RUN pip install --no-cache-dir --upgrade pip>=25.3

# Install uv for faster pip installs (after pip upgrade)
RUN pip install --no-cache-dir uv

# Copy requirements from ROOT
COPY requirements.txt .

# ✅ Install dependencies with uv (removed --quiet for better debugging)
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code: ROOT/src -> /build/src
COPY src ./src

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

# Verify AOT artifact exists (if strict mode)
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

# ✅ FIX CVE-2025-8869: Upgrade pip in final stage too
RUN pip install --no-cache-dir --upgrade pip>=25.3

WORKDIR /app/src

# Copy Python runtime + artifacts from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/src /app/src

# Copy config (optional - can be mounted at runtime)
COPY --from=builder /build/config_macd.json /app/src/config_macd.json 2>/dev/null || true

# Runtime configuration with optimizations
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

# Create non-root user with proper permissions
RUN useradd --uid 1000 -m appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app/logs

USER appuser

# Default command
CMD ["python", "-u", "macd_unified.py"]

# Metadata labels
LABEL maintainer="your-email@example.com" \
      version="1.8.0" \
      description="MACD Unified Trading Bot with AOT Compilation" \
      security.fixes="CVE-2025-8869"