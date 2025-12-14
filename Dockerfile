# ============================================================================
# Stage 0: Get uv binary from official lightweight image
# ============================================================================
FROM ghcr.io/astral-sh/uv:latest AS uv-provider

# ============================================================================
# Stage 1: Builder - Install dependencies with uv
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install only the build dependencies needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc6-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy uv binary from the official uv image
COPY --from=uv-provider /uv /usr/local/bin/uv

# Make uv executable (just in case)
RUN chmod +x /usr/local/bin/uv

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV

# Activate venv for subsequent commands
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies (pin pycares/aiodns for compatibility)
RUN uv pip install --no-cache --compile-bytecode \
        pycares==4.4.0 \
        aiodns==3.2.0 && \
    uv pip install --no-cache --compile-bytecode -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim-bookworm

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    useradd -m -u 1000 botuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

WORKDIR /app

# Copy application code
COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./

# Set ownership and permissions
RUN chown -R botuser:botuser /app && \
    chmod +x wrapper.py && \
    chmod 644 config_macd.json

USER botuser

CMD ["python", "-u", "wrapper.py"]