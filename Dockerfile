# ============================================================================
# Stage 1: Builder - Uses 'uv' for lightning-fast installs and full dependencies
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Get the fast package installer 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install build dependencies (needed for packages like numpy/pandas/psutil if wheels fail)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements and install dependencies with uv (Forces rebuild to fix psutil error)
COPY requirements.txt .
# The cache-buster here (uv venv) ensures this step runs fresh if requirements.txt changes.
RUN uv pip install --no-cache -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal & Optimized for Speed
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Install only runtime libraries (e.g., for OpenMP used by numpy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY wrapper.py config_macd.json ./

# Environment configuration (Crucially sets PYTHONPATH to find 'src' modules)
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

# Create non-root user for security
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app

# Pre-compile python bytecode for faster startup time (removes need for PYTHONDONTWRITEBYTECODE=1)
RUN python -m compileall /app

USER botuser

# Run command
CMD ["python", "-u", "wrapper.py"]
