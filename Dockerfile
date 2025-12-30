# =============================================================================
# MULTI-STAGE BUILD: Production-Ready Dockerfile with Security Fixes
# Fixes CVE-2025-8869 + All Enterprise Features
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

# ✅ FIX CVE-2025-8869: Upgrade pip to 25.3+ FIRST (before any other installs)
RUN pip install --no-cache-dir --upgrade pip>=25.3

# Install uv for faster pip installs (after pip upgrade)
RUN pip install --no-cache-dir uv

# ✅ FEATURE 1: Copy and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy everything from context (includes src/ and optional config)
COPY . .

# Move src to correct location and handle config
RUN if [ -d src ]; then \
        cp -r src/* . && rm -rf src; \
    fi && \
    if [ -f config_macd.json ]; then \
        echo "✅ Config file found"; \
    else \
        echo "⚠️  Config not found - runtime mount or env vars required"; \
    fi

WORKDIR /build

# ✅ FEATURE 2: AOT Build with Explicit Verification
ARG AOT_STRICT=1
RUN cd /build && \
    python aot_build.py && \
    ls -lh macd_aot_compiled*.so 2>/dev/null || \
    if [ "$AOT_STRICT" = "1" ]; then \
        echo "❌ ERROR: AOT artifact missing" && exit 1; \
    else \
        echo "⚠️  JIT fallback allowed (dev build)"; \
    fi

# Explicit AOT verification with clear messaging
RUN if [ "$AOT_STRICT" = "1" ]; then \
        SO_FILE=$(ls /build/macd_aot_compiled*.so 2>/dev/null | head -1); \
        if [ -z "$SO_FILE" ]; then \
            echo "❌ FATAL: AOT_STRICT=1 but no .so file found" && exit 1; \
        fi; \
        SO_SIZE=$(stat -c%s "$SO_FILE" 2>/dev/null || echo "0"); \
        if [ "$SO_SIZE" -lt 10000 ]; then \
            echo "❌ FATAL: AOT artifact too small (${SO_SIZE} bytes)" && exit 1; \
        fi; \
        echo "✅ AOT artifact verified: $(basename $SO_FILE) ($(echo "scale=1; $SO_SIZE/1024" | bc)KB)"; \
    fi

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# Install runtime dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 \
    tzdata \
    ca-certificates \
    bc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ✅ FIX CVE-2025-8869: Upgrade pip in final stage too
RUN pip install --no-cache-dir --upgrade pip>=25.3

WORKDIR /app/src

# Copy Python runtime + all build artifacts from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build /app/src

# ✅ FEATURE 3: Verify config presence
RUN if [ -f /app/src/config_macd.json ]; then \
        echo "✅ Config copied to runtime image"; \
    else \
        echo "ℹ️  No config in image - expecting volume mount or env vars"; \
    fi

# ✅ FEATURE 4: Enhanced Environment Variables
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

# ✅ FEATURE 5: Create non-root user with logs directory
RUN useradd --uid 1000 -m appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app/logs && \
    echo "📁 Logs directory created at /app/logs"

USER appuser

# Default command
CMD ["python", "-u", "macd_unified.py"]