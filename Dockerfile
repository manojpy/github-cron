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
    NUMBA_THREADING_LAYER=omp

# 🔥 AOT COMPILATION - Pre-compile all Numba functions
RUN mkdir -p /app/src/__pycache__ && \
    chmod 755 /app/src/__pycache__ && \
    python src/compile_numba_aot.py && \
    echo "✅ AOT compilation completed" && \
    echo "🔍 Verifying cache files:" && \
    find /app/src/__pycache__ -type f \( -name "*.nbi" -o -name "*.nbc" \) && \
    CACHE_COUNT=$(find /app/src/__pycache__ -type f \( -name "*.nbi" -o -name "*.nbc" \) | wc -l) && \
    echo "📦 Total cache files: $CACHE_COUNT" && \
    if [ "$CACHE_COUNT" -lt 15 ]; then \
        echo "❌ ERROR: Expected at least 15 cache files, found only $CACHE_COUNT"; \
        exit 1; \
    fi

# Stage 3: Final Runtime
FROM ${BASE_DIGEST} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# 🔥 CRITICAL: Copy entire src directory with cache in one operation
COPY --from=aot-compiler /app/src/ ./src/

# Copy runtime files
COPY wrapper.py config_macd.json ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp \
    TZ=Asia/Kolkata

# Verify cache and set permissions
RUN echo "🔍 Verifying AOT cache in runtime stage:" && \
    ls -lah /app/src/ && \
    ls -lah /app/src/__pycache__/ && \
    CACHE_COUNT=$(find /app/src/__pycache__ -type f \( -name "*.nbi" -o -name "*.nbc" \) 2>/dev/null | wc -l) && \
    echo "📦 Found $CACHE_COUNT cache files" && \
    if [ "$CACHE_COUNT" -lt 5 ]; then \
        echo "⚠️ WARNING: Expected more cache files, found only $CACHE_COUNT"; \
    else \
        echo "✅ AOT cache verified successfully"; \
        find /app/src/__pycache__ -type f \( -name "*.nbi" -o -name "*.nbc" \) | head -5; \
    fi && \
    useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

USER botuser

ENTRYPOINT ["python", "-u", "wrapper.py"]