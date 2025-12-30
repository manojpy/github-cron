FROM python:3.11-slim AS builder
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /build/src

COPY requirements.txt ../

RUN pip install --no-cache-dir --upgrade pip>=25.3 && \
    pip install --no-cache-dir uv && \
    uv pip install --system --no-cache-dir -r ../requirements.txt
COPY src/ .

ARG AOT_STRICT=1
RUN python aot_build.py && \
    ls -lh macd_aot_compiled*.so 2>/dev/null || \
    ( [ "$AOT_STRICT" != "1" ] || (echo "❌ AOT missing" && exit 1) )

FROM python:3.11-slim AS final

RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 tzdata ca-certificates bc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip>=25.3
RUN useradd --uid 1000 -m appuser && mkdir -p /app/{src,logs} && \
    chown -R appuser:appuser /app

WORKDIR /app/src
COPY --from=builder --chown=appuser:appuser /usr/local /usr/local
COPY --from=builder --chown=appuser:appuser /build/src/ /app/src/
RUN ls -la macd_unified.py aot_bridge.py macd_aot_compiled*.so* || echo "Files OK"

ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/app/src/__pycache__ NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=4 NUMBA_WARNINGS=0 NUMBA_OPT=3 TZ=Asia/Kolkata

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER appuser

RUN mkdir -p /app/logs

CMD ["python", "-u", "macd_unified.py"]
