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

# 🔥 AOT COMPILATION
RUN mkdir -p /app/src/__pycache__ && \
    python src/compile_numba_aot.py && \
    echo "✅ AOT compilation completed"

# Stage 3: Final Runtime
FROM ${BASE_DIGEST} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Setup directory structure
RUN mkdir -p /app/src/__pycache__

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
    TZ=Asia/Kolkata

# Verify and set permissions
RUN CACHE_COUNT=$(find /app/src/__pycache__ -type f \( -name "*.nb*" -o -name "*.npz" -o -name "*.pkl" \) | wc -l) && \
    echo "📁 Found $CACHE_COUNT AOT cache files" && \
    useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

USER botuser
ENTRYPOINT ["python", "-u", "wrapper.py"]