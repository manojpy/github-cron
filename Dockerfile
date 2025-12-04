# ============================================================================
# Stage 1: Builder - compile dependencies with all build tools
# ============================================================================
FROM python:3.11.11-slim-bookworm AS builder

# Install build dependencies (only if you really have packages that need compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with cache
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --cache-dir=/tmp/pip-cache -r /tmp/requirements.txt

# ============================================================================
# Stage 2: Runtime - minimal image with only runtime dependencies
# ============================================================================
FROM python:3.11.11-slim-bookworm

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean autoclean \
    && rm -rf /var/cache/apt/* /usr/share/man /usr/share/doc

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Activate venv + best practices for containerized Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

# Working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY gitlab_wrapper.py config_macd.json ./

# Non-root user (security)
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app

USER botuser

# Clean shutdown on SIGTERM (important for cron runners)
STOPSIGNAL SIGTERM

# Remove useless healthcheck (it always succeeds and serves no purpose in cron jobs)
# HEALTHCHECK NONE   # explicitly disable if Docker complains

# Critical: This is the actual command that runs your bot once and exits
CMD ["python", "src/macd_unified.py", "--once"]