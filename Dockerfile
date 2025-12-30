# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential git curl libtbb12 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir -r requirements.txt

# Copy source logic AND the root config file
COPY src/ .
COPY config_macd.json . 

# AOT Build
ARG AOT_STRICT=1
RUN python aot_build.py && \
    SO_FILE=$(ls macd_aot_compiled*.so 2>/dev/null | head -1) && \
    if [ -z "$SO_FILE" ] && [ "$AOT_STRICT" = "1" ]; then exit 1; fi

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 tzdata ca-certificates && rm -rf /var/lib/apt/lists/*

ENV NUMBA_THREADING_LAYER=tbb \
    NUMBA_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1

RUN useradd --uid 1000 -m appuser && mkdir -p /app/{src,logs} && \
    chown -R appuser:appuser /app

WORKDIR /app/src

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# This copies everything from builder's /build (which includes config_macd.json)
COPY --from=builder --chown=appuser:appuser /build/ /app/src/

USER appuser

ENTRYPOINT ["python", "macd_unified.py"]