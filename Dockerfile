# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN uv pip install --no-cache --compile \
    pycares==4.4.0 \
    aiodns==3.2.0

RUN uv pip install --no-cache --compile -r requirements.txt

# Stage 2: Runtime (smaller: no tzdata)
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_FILE=config_macd.json \
    TZ=Asia/Kolkata

RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app && \
    chmod +x wrapper.py

USER botuser

CMD ["python", "-u", "wrapper.py"]