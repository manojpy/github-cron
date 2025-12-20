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

# Setup Numba cache directory
RUN mkdir -p /app/numba_cache && chmod 777 /app/numba_cache

# Copy source code for compilation
COPY src/ ./src/
COPY compile_numba_aot.py ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/numba_cache \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp

# 🔥 AOT COMPILATION - Pre-compile all Numba functions
RUN python compile_numba_aot.py && \
    echo "✅ AOT compilation completed successfully" && \
    ls -lah /app/numba_cache

# Stage 3: Final Runtime
FROM ${BASE_DIGEST} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
# 🔥 Copy pre-compiled Numba cache from AOT stage
COPY --from=aot-compiler /app/numba_cache /app/numba_cache

WORKDIR /app

# Copy source and wrapper
COPY src/ ./src/
COPY wrapper.py config_macd.json ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/numba_cache \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp \
    TZ=Asia/Kolkata

RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

USER botuser

ENTRYPOINT ["python", "-u", "wrapper.py"]