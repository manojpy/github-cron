# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

# Install compilers and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv first
RUN pip install uv

# Copy only requirements first to maximize cache hits
COPY requirements.txt .

# Use uv for dependencies
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code last
COPY src ./src

WORKDIR /build/src

# Run AOT compile step via aot_build.py and fail hard if artifact is missing
RUN python aot_build.py && \
    test -f /build/src/_macd_aot.so || (echo "❌ AOT artifact missing" && exit 1)

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# Install runtime libraries only (no compilers here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python runtime, site-packages, and compiled artifacts from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/src /app/src

WORKDIR /app

# Ensure Python can find src/ modules
ENV PYTHONPATH=/app/src

# Default command: run macd_unified directly
CMD ["python", "-m", "macd_unified"]
