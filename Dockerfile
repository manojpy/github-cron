# Dockerfile
# Multi-stage build using uv for fast installs, no pip cache, bytecode compilation,
# and PYTHONPATH configured for both repo root and src/.

# ============================================================================
# Stage 1: Builder - use uv to install dependencies
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Bring in uv binary from upstream image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN chmod +x /bin/uv

# Environment for deterministic builds
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

# Copy only requirements first to leverage Docker cache for deps
COPY requirements.txt requirements.txt

# Install dependencies using uv (wraps pip) into system prefix (/usr/local)
RUN /bin/uv pip install --no-warn-script-location --no-cache-dir -r requirements.txt --prefix=/usr/local

# Copy application source
COPY . /app

# Pre-compile bytecode to speed up cold starts (cron frequent runs)
RUN python -m compileall /app || true

# ============================================================================
# Stage 2: Runtime image (slim)
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    TZ=Asia/Kolkata

WORKDIR /app

# Copy installed packages & app from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Create a non-root user for security
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app

USER botuser

# Default command runs the wrapper once (CI-friendly)
CMD ["python", "-u", "wrapper.py"]
