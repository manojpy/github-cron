============================================================================

Stage 1: Builder - Uses 'uv' for lightning-fast installs

============================================================================

ARG BASE_DIGEST=python:3.11-slim-bookworm
FROM ${BASE_DIGEST} AS builder

Install build dependencies for orjson/uvloop/numba

RUN apt-get update && apt-get install -y --no-install-recommends 
gcc g++ libc6-dev 
&& rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

Copy requirements first for better caching

COPY requirements.txt .

Fix specific incompatibilities and install requirements

Added 'setuptools' which is required by Numba's pycc (AOT compiler)

RUN uv pip install --no-cache --compile 
setuptools 
pycares==4.4.0 
aiodns==3.2.0 && 
uv pip install --no-cache --compile -r requirements.txt

⚡ Build Numba AOT modules

COPY src/numba_aot.py /tmp/
RUN cd /tmp && 
python -m numba.pycc --python numba_aot.py && 
cp numba_compiled*.so /opt/venv/lib/python3.11/site-packages/ && 
echo "✅ Numba AOT modules compiled"

============================================================================

Stage 2: Runtime - Minimal & Optimized

============================================================================

FROM ${BASE_DIGEST} AS runtime

RUN apt-get update && 
apt-get install -y --no-install-recommends 
libgomp1 ca-certificates tzdata 
&& rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app



RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

Copy source code and config

COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./

Environment configuration

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENTRYPOINT ["python", "wrapper.py"]

