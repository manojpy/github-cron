# ============================================================================
# Stage 1: Builder - Uses 'uv' for lightning-fast installs
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# COMBINED: Combine update and install to reduce image layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal & Optimized
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata \
    LOG_JSON=false \
    FILE_LOGGING=false

# OPTIMIZED: Combined user creation and permission setting
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

# COMPILE: Keeps startup slightly faster
RUN python -m compileall /app/src

USER botuser

# REMOVED: HEALTHCHECK
# Reason: This is a batch job that runs once and exits. 
# A healthcheck adds polling overhead and is intended for long-running services (daemons).
# The wrapper.py logic already handles exit codes for success/failure.

CMD ["python", "-u", "wrapper.py"]