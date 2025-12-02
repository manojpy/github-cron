# ============================================================================
# Stage 1: Builder - compile dependencies with all build tools
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies for compiling Python packages (hiredis/uvloop)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install (this layer will be cached unless requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# Stage 2: Runtime - minimal image with only runtime dependencies
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Install only runtime libraries (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set up working directory
WORKDIR /app

# Copy application files
COPY src/ ./src/
COPY gitlab_wrapper.py config_macd.json ./

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata \
    LOG_JSON=false \
    FILE_LOGGING=true \
    PROMETHEUS_ENABLED=false \
    HEALTH_SERVER_ENABLED=false

# Create non-root user for security
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    mkdir -p /app/logs && \
    chown -R botuser:botuser /app/logs

USER botuser

# Health check - checks if script file exists and Python works
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os; assert os.path.exists('src/macd_unified.py')" || exit 1

# Default command - run the wrapper script
CMD ["python", "gitlab_wrapper.py"]