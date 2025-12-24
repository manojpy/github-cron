# syntax=docker/dockerfile:1
# ----------------------------------------------------------
# builder stage – python 3.11 slim
# ----------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# install uv
ADD https://astral.sh/uv/install.sh /tmp/uv.sh
RUN sh /tmp/uv.sh && mv "$HOME/.local/bin/uv" /usr/local/bin/

WORKDIR /build
COPY requirements.txt .
RUN uv pip install --python python3.11 --system -r requirements.txt

# compile AOT
COPY src/ /build/src/
WORKDIR /build/src
RUN python3.11 aot_build.py

# ✅ check for any ABI-suffixed .so file
RUN python3.11 aot_build.py && \
    ls -l /build/src && \
    test -e /build/src/_macd_aot*.so || (echo "❌ AOT .so missing" && exit 1)

# ----------------------------------------------------------
# final stage – python 3.11 slim runtime
# ----------------------------------------------------------
FROM python:3.11-slim AS final

# runtime libraries including TBB for Numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

# copy python runtime, site-packages, and compiled artifact
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/src /app/src

WORKDIR /app

# ensure Python can find src/ modules
ENV PYTHONPATH=/app/src

# default command: run macd_unified directly
CMD ["python", "-m", "macd_unified"]
