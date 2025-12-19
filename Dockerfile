============================================================================

Stage 1: Builder

============================================================================

ARG BASE_DIGEST=python:3.11-slim-bookworm
FROM ${BASE_DIGEST} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends 
gcc g++ libc6-dev 
&& rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt .

Install dependencies including the newly added setuptools

RUN uv pip install --no-cache --compile -r requirements.txt

⚡ Build Numba AOT modules

Ensure the source file is copied from the correct local path

COPY src/numba_aot.py /tmp/
RUN cd /tmp && 
python -m numba.pycc --python numba_aot.py && 
cp numba_compiled*.so /opt/venv/lib/python3.11/site-packages/ && 
echo "✅ Numba AOT modules compiled"

============================================================================

Stage 2: Runtime

============================================================================

FROM ${BASE_DIGEST} AS runtime
RUN apt-get update && 
apt-get install -y --no-install-recommends 
libgomp1 ca-certificates tzdata 
&& rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app

Ensure Numba has a writable cache directory for any remaining JIT needs

RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache
COPY src/macd_unified.py ./src/
COPY wrapper.py config_macd.json ./
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENTRYPOINT ["python", "wrapper.py"]

