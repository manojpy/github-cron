# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

# Install compilers and build tools
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv for faster pip installs
RUN pip install --quiet --no-cache-dir uv

# Copy requirements from ROOT
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir --quiet -r requirements.txt

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
        echo "ERROR: AOT artifact missing" && exit 1; \
    else \
        echo "JIT fallback allowed (dev build)"; \
    fi


# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final


# 1. Install tzdata and set the timezone
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src

# Copy Python runtime
COPY --from=builder /usr/local /usr/local

# Copy application code and compiled artifacts
COPY --from=builder /build/src /app/src
# Copy config to the same directory as scripts for easy access
COPY --from=builder /build/config_macd.json /app/src/config_macd.json

# Runtime config
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=2 \
    NUMBA_WARNINGS=0
    ENV TZ=Asia/Kolkata
    RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Execute file directly from src directory
CMD ["python", "-u", "macd_unified.py"]