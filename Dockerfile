# Stage 1: Builder
ARG BASE_DIGEST=python:3.11-slim-bookworm
FROM ${BASE_DIGEST} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/* [cite: 1]

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv [cite: 1]

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV [cite: 1]
ENV PATH="$VIRTUAL_ENV/bin:$PATH" [cite: 1]

COPY requirements.txt . [cite: 1]
RUN uv pip install --no-cache --compile \
    pycares==4.4.0 \
    aiodns==3.2.0 && \
    uv pip install --no-cache --compile -r requirements.txt [cite: 2]

# Stage 2: Runtime
FROM ${BASE_DIGEST} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/* [cite: 2]

COPY --from=builder /opt/venv /opt/venv [cite: 2]

WORKDIR /app

# FIX: Use /app/numba_cache for better file locator mapping
RUN mkdir -p /app/numba_cache && chmod 777 /app/numba_cache 

COPY src/ ./src/
COPY wrapper.py config_macd.json ./ 

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/numba_cache \
    NUMBA_NUM_THREADS=4 \
    NUMBA_THREADING_LAYER=omp \
    TZ=Asia/Kolkata [cite: 3]

RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py 

USER botuser

ENTRYPOINT ["python", "-u", "wrapper.py"]