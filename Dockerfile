# ============================================================================
# Stage 1: Builder - Ultra-fast dependency installation with uv
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# OPTIMIZATION: Single-layer build dependencies installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements first for better layer caching
COPY requirements.txt .

# OPTIMIZATION: Install all dependencies in one command
# Fix pycares/aiodns compatibility, then install everything else
RUN uv pip install --no-cache --compile \
    pycares==4.4.0 \
    aiodns==3.2.0 && \
    uv pip install --no-cache --compile -r requirements.txt

# OPTIMIZATION: Pre-compile Python bytecode for faster imports (~15-20% faster startup)
RUN python -m compileall -q /opt/venv/lib/python3.11/site-packages

# ============================================================================
# Stage 2: Runtime - Minimal & Blazing Fast
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

# OPTIMIZATION: Minimal runtime dependencies (removed tzdata - using TZ env var)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean

# Copy pre-compiled virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# OPTIMIZATION: Copy files in order of change frequency (least → most)
# Config changes least often (separate layer for better caching)
COPY config_macd.json ./

# Source code (changes more frequently)
COPY src/macd_unified.py ./src/
COPY wrapper.py ./

# OPTIMIZATION: Enhanced environment variables for performance
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    PYTHONHASHSEED=random \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

# Security: Run as non-root user
RUN useradd -m -u 1000 -s /bin/bash botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

USER botuser

# OPTIMIZATION: Use exec form for faster startup (no shell overhead)
CMD ["python", "-u", "wrapper.py"]