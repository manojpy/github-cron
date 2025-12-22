# Stage 1: Builder
ARG BASE_DIGEST=python:3.11-slim-bookworm
FROM ${BASE_DIGEST} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN uv pip install --no-cache --compile \
    pycares==4.4.0 \
    aiodns==3.2.0 && \
    uv pip install --no-cache --compile -r requirements.txt

# Stage 2: AOT Compilation Stage
FROM ${BASE_DIGEST} AS aot-compiler

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy source code
COPY src/ ./src/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=0 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp \
    NUMBA_DISABLE_JIT=0

# 🔥 AOT COMPILATION WITH VERIFICATION
RUN set -ex && \
    mkdir -p /app/src/__pycache__ && \
    chmod -R 777 /app/src/__pycache__ && \
    echo "📦 Starting Numba AOT compilation..." && \
    python src/compile_numba_aot.py && \
    echo "" && \
    echo "🔍 Verifying cache files..." && \
    ls -lah /app/src/__pycache__/ && \
    CACHE_COUNT=$(find /app/src/__pycache__ -type f \( -name "*.nbi" -o -name "*.nbc" \) 2>/dev/null | wc -l) && \
    echo "📊 Cache file count: $CACHE_COUNT" && \
    if [ "$CACHE_COUNT" -lt 15 ]; then \
        echo "❌ ERROR: Expected at least 15 cache files, found $CACHE_COUNT"; \
        echo "Cache contents:"; \
        find /app/src/__pycache__ -type f; \
        exit 1; \
    fi && \
    echo "✅ AOT compilation verified: $CACHE_COUNT cache files created"

# Stage 3: Final Runtime
FROM ${BASE_DIGEST} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Setup directory structure with proper permissions
RUN mkdir -p /app/src/__pycache__ && chmod -R 777 /app/src/__pycache__

# Copy source files and the PRE-COMPILED cache
COPY --from=aot-compiler /app/src/*.py ./src/
COPY --from=aot-compiler /app/src/__pycache__/ ./src/__pycache__/
COPY wrapper.py config_macd.json ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp \
    NUMBA_DISABLE_JIT=0 \
    TZ=Asia/Kolkata

# Verify cache was copied successfully and set permissions
RUN set -ex && \
    echo "🔍 Verifying runtime cache..." && \
    ls -lah /app/src/__pycache__/ && \
    CACHE_COUNT=$(find /app/src/__pycache__ -type f \( -name "*.nbi" -o -name "*.nbc" \) 2>/dev/null | wc -l) && \
    echo "📁 Found $CACHE_COUNT AOT cache files in runtime" && \
    if [ "$CACHE_COUNT" -lt 15 ]; then \
        echo "⚠️  WARNING: Expected at least 15 cache files, found $CACHE_COUNT"; \
        echo "Runtime will use JIT compilation (slower startup)"; \
    else \
        echo "✅ AOT cache successfully copied to runtime"; \
    fi && \
    useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py && \
    echo "✅ Runtime environment ready"

USER botuser
ENTRYPOINT ["python", "-u", "wrapper.py"]