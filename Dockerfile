# ============================================================================
# Stage 1: Builder - compile dependencies with uv (ultra-fast)
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv - the ultra-fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    UV_SYSTEM_PYTHON=1

# Copy requirements and install dependencies with uv (10-100x faster than pip)
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache -r /tmp/requirements.txt

# ============================================================================
# Stage 2: Runtime - minimal image with only runtime dependencies
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Install only runtime libraries (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set up working directory
WORKDIR /app

# Copy application files
COPY src/ ./src/
COPY gitlab_wrapper.py config_macd.json ./

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

# Pre-compile Python bytecode for faster startup
RUN python -m compileall -q /app/src /app/gitlab_wrapper.py

# Create non-root user for security
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"