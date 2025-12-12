# ============================================================================
# OPTIMIZATION: Alpine-based for smaller image size (faster pulls)
# WARNING: Only use if all deps support Alpine (musl libc)
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# OPTIMIZATION: Single-layer dependency install
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libc6-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .

# OPTIMIZATION: Install all deps in one layer (faster builds)
RUN uv pip install --no-cache --compile \
    pycares==4.4.0 aiodns==3.2.0 && \
    uv pip install --no-cache --compile -r requirements.txt && \
    # Pre-compile Python bytecode for faster startup
    python -m compileall -q /opt/venv

# ============================================================================
# OPTIMIZATION: Use distroless for minimal size (optional)
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

# OPTIMIZATION: Minimal runtime (no unnecessary packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

WORKDIR /app

# OPTIMIZATION: Copy in optimal layer order (config changes least)
COPY --from=builder /opt/venv /opt/venv
COPY config_macd.json ./
COPY src/macd_unified.py ./src/
COPY wrapper.py ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

# OPTIMIZATION: Create user in one layer
RUN useradd -m -u 1000 botuser \
    && chown -R botuser:botuser /app \
    && chmod +x wrapper.py

USER botuser

CMD ["python", "-u", "wrapper.py"]