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
    NUMBA_CACHE_DIR=/app/numba_cache \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp

# AOT COMPILATION - Pre-compile all Numba functions
RUN mkdir -p /app/numba_cache && \
    python src/compile_numba_aot.py && \
    echo "✅ AOT compilation completed" && \
    echo "🔍 Checking cache directories after AOT:" && \
    ls -lah /app/numba_cache/ && \
    echo "🔍 Recursive file listing for *.nb* / *.npz / *.pkl:" && \
    find /app/numba_cache -type f \( -name "*.nb*" -o -name "*.npz" -o -name "*.pkl" \) | head -40

# Stage 3: Final Runtime
FROM ${BASE_DIGEST} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Create directory structure
RUN mkdir -p /app/numba_cache

# Copy Python source files
COPY --from=aot-compiler /app/src/*.py ./src/

# Copy the Numba cache directory separately
COPY --from=aot-compiler /app/numba_cache/ ./numba_cache/

# Copy runtime files
COPY wrapper.py config_macd.json ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/numba_cache \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp \
    TZ=Asia/Kolkata

# Verify cache was copied and set permissions
RUN echo "🔍 Verifying AOT cache in runtime stage:" && \
    ls -lah /app/numba_cache/ && \
    CACHE_COUNT=$(find /app/numba_cache -type f \( -name "*.nb*" -o -name "*.npz" -o -name "*.pkl" \) | wc -l) && \
    echo "📁 Found $CACHE_COUNT cache files" && \
    if [ "$CACHE_COUNT" -lt 5 ]; then \
        echo "⚠️  WARNING: Expected more cache files, found only $CACHE_COUNT"; \
    else \
        echo "✅ AOT cache verified successfully"; \
    fi && \
    useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

USER botuser

ENTRYPOINT ["python", "-u", "wrapper.py"]
