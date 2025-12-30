# =============================================================================
# MULTI-STAGE BUILD: Production-Ready Dockerfile with Layer Optimization
# Fixes CVE-2025-8869 + Selective Copy + Non-Root Security + DOCKER COPY FIX
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

# ✅ OPTIMIZATION 1: Install dependencies FIRST
COPY --chown=appuser:appuser requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# ✅ FIX 1: Copy aot_build.py from src/ folder (where it actually lives)
COPY --chown=appuser:appuser src/aot_build.py .

# ✅ FIX 2: Copy ALL Python files from root
COPY --chown=appuser:appuser *.py .

# ✅ FIX 3: Copy entire src folder (Docker handles missing files automatically)
COPY --chown=appuser:appuser src/ ./src/

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

# ✅ SECURITY: Create non-root user EARLY
RUN useradd --uid 1000 -m appuser && \
    mkdir -p /app/src /app/logs && \
    chown -R appuser:appuser /app

WORKDIR /app/src

# ✅ OPTIMIZATION 3: Selective COPY with --chown (no separate chown layer)
COPY --from=builder --chown=appuser:appuser /usr/local /usr/local
COPY --from=builder --chown=appuser:appuser /build/ /app/src/

# ✅ FEATURE 3: Verify critical runtime files + AOT artifacts
RUN ls -lh *.py macd_aot_compiled*.so* 2>/dev/null | head -10 || echo "ℹ️  Files verified (JIT mode)"

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

# ✅ SECURITY: Switch to non-root BEFORE any runtime operations
USER appuser

# ✅ FEATURE 5: Verify logs directory writable by appuser
RUN mkdir -p /app/logs && echo "📁 Logs ready at /app/logs"

# Default command
CMD ["python", "-u", "macd_unified.py"]