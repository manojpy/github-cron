# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

# install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# install uv
ADD https://astral.sh/uv/install.sh /tmp/uv.sh
RUN sh /tmp/uv.sh && mv $HOME/.cargo/bin/uv /usr/local/bin/

WORKDIR /build
COPY requirements.txt .
RUN uv pip install --python python3.11 --system -r requirements.txt

# compile AOT
COPY src/ /build/src/
RUN cd /build/src && python3.11 aot_build.py

# ----------------------------------------------------------
# final slim stage – Ubuntu 24.04 as requested
# ----------------------------------------------------------
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /build/src/_macd_aot.so /app/src/
COPY src/ /app/src/
WORKDIR /app

ENV PYTHONPATH=/app
CMD ["python3.11", "-m", "wrapper"]
