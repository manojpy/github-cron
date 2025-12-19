# =========================
# Builder stage
# =========================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY src/ ./src/

# Install requirements (numba must be here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Compile AOT module inside src
WORKDIR /app/src
RUN python -u aot_build.py

# ✅ Verify that the .so was produced in /app/src
RUN ls -l /app/src/indicators_aot*.so || (echo "❌ AOT .so not found in builder"; exit 1)

# =========================
# Runtime stage
# =========================
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Copy source and wrapper
COPY src/ ./src/
COPY wrapper.py config_macd.json ./

# Copy compiled AOT module from builder
COPY --from=builder /app/src/indicators_aot.*.so /app/

# ✅ Verify that the .so is present in runtime
RUN ls -l /app/indicators_aot*.so || (echo "❌ AOT .so not found in runtime"; exit 1)

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔑 Fix: install OpenMP runtime and set safe threading layer
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV NUMBA_THREADING_LAYER=workqueue
ENV NUMBA_NUM_THREADS=12

# Environment
ENV PYTHONPATH="/app"

# Entrypoint
CMD ["python", "-u", "wrapper.py"]
