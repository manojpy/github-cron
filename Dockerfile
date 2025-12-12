# ============================================================================
# OPTIMIZATION: Use smaller base image for faster pulls
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# OPTIMIZATION: Combine apt commands to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .

# OPTIMIZATION: Install dependencies in one layer
RUN uv pip install --no-cache --compile \
    pycares==4.4.0 \
    aiodns==3.2.0 \
    && uv pip install --no-cache --compile -r requirements.txt

# ============================================================================
# OPTIMIZATION: Use distroless for minimal size (optional)
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

# OPTIMIZATION: Minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# OPTIMIZATION: Copy files in optimal order (least changing first)
COPY config_macd.json ./
COPY wrapper.py ./
COPY src/macd_unified.py ./src/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

# OPTIMIZATION: Create user in one layer
RUN useradd -m -u 1000 botuser \
    && chown -R botuser:botuser /app \
    && chmod +x wrapper.py

USER botuser

CMD ["python", "-u", "wrapper.py"]