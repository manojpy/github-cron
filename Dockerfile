FROM python:3.11-slim AS builder

# Install compilers and build tools
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv for faster pip installs + upgrade pip (fixes CVE)
RUN pip install --quiet --no-cache-dir uv && \
    pip install --upgrade pip==25.3 --quiet

# Copy requirements from ROOT
COPY requirements.txt .
# ✅ FIXED: Remove --quiet for uv pip compatibility
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

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 \
    tzdata \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src

# Copy Python runtime + artifacts
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/src /app/src

# Runtime config - ADD TCP limits
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=4 \
    NUMBA_WARNINGS=0 \
    TZ=Asia/Kolkata \
    AIOHTTP_MAX_CONNS=16 \
    AIOHTTP_CONNS_PER_HOST=12

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Non-root user
RUN useradd --uid 1000 -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-u", "macd_unified.py"]