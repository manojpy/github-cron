# ============================================================================
# Stage 1: Builder - Fast dependency installation with uv
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies + uv in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc6-dev \
        curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/uv && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements early for better caching
COPY requirements.txt .

# Install dependencies with uv (fast, compiled wheels where possible)
# Pin pycares/aiodns first for compatibility, then install rest
RUN uv pip install --no-cache --compile-bytecode \
        pycares==4.4.0 \
        aiodns==3.2.0 && \
    uv pip install --no-cache --compile-bytecode -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim-bookworm

# Install only essential runtime packages in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    # Create non-root user early
    useradd -m -u 1000 botuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH for venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

WORKDIR /app

# Copy application files
COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./

# Set permissions and switch to non-root user
RUN chown -R botuser:botuser /app && \
    chmod +x wrapper.py && \
    chmod 644 config_macd.json

USER botuser

CMD ["python", "-u", "wrapper.py"]