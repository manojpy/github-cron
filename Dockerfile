# =========
# Stage 1: Build & compile true AOT .so
# =========
FROM python:3.11-slim AS builder

# System dependencies for numpy/numba and SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libssl-dev \
    libffi-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ src/
COPY config_macd.json config_macd.json

# Compile the AOT .so library using pycc
WORKDIR /app/src
RUN python aot_build.py

# =========
# Stage 2: Runtime (slim, no JIT warmup)
# =========
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python runtime from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code and compiled .so
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config_macd.json /app/config_macd.json
COPY wrapper.py wrapper.py

# Environment variables
ENV CONFIG_FILE=/app/config_macd.json
ENV NUMBA_CACHE_DIR=/app/src/__pycache__
ENV SKIP_WARMUP=true

# Entrypoint
CMD ["python", "wrapper.py"]
