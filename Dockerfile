# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

# Install build tools needed for compiling AOT artifacts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first to maximize cache hits
COPY requirements.txt .

# Use uv (Astral’s ultra-fast installer) for dependencies
# --system installs into the image’s Python environment
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code last (so dependency layer is cached unless requirements change)
COPY src ./src

# Compile AOT artifacts (Numba .so files)
WORKDIR /build/src
RUN python -m aot_bridge --compile


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
