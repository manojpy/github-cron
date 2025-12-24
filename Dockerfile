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
RUN cd /build/src && python3.11 aot_build.py

# ✅ check for any ABI-suffixed .so file
RUN cd /build/src && python3.11 aot_build.py && \
    ls -l /build/src && \
    test -e /build/src/_macd_aot*.so || (echo "❌ AOT .so missing" && exit 1)

# ----------------------------------------------------------
# final stage – ubuntu 24.04 + python 3.11 runtime
# ----------------------------------------------------------
FROM ubuntu:24.04

# runtime libraries python still needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 libexpat1 libbz2-1.0 libsqlite3-0 libncursesw6 \
    libreadline8 libtinfo6 zlib1g liblzma5 && \
    rm -rf /var/lib/apt/lists/*

# copy python 3.11 runtime + packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin/python3.11 /usr/local/bin/python3.11
COPY --from=builder /usr/local/bin/python3       /usr/local/bin/python3
RUN ln -sf python3.11 /usr/local/bin/python

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# ✅ copy any matching .so file
COPY --from=builder /build/src/_macd_aot*.so /app/src/

COPY src/ /app/src/
WORKDIR /app

ENV PYTHONPATH=/app
CMD ["python", "-m", "wrapper"]
