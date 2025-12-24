# syntax=docker/dockerfile:1
# builder stage – python 3.11 slim
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /tmp/uv.sh
RUN sh /tmp/uv.sh && mv $HOME/.cargo/bin/uv /usr/local/bin/

WORKDIR /build
COPY requirements.txt .
RUN uv pip install --python python3.11 --system -r requirements.txt

COPY src/ /build/src/
RUN cd /build/src && python3.11 aot_build.py

# ----------------------------------------------------------
# final stage – ubuntu 24.04 + python 3.11 from builder
# ----------------------------------------------------------
FROM ubuntu:24.04

# bring python 3.11 runtime + stdlib from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin/python3.11 /usr/local/bin/python3.11
COPY --from=builder /usr/local/bin/python3       /usr/local/bin/python3
RUN ln -sf python3.11 /usr/local/bin/python

# runtime libs that python still needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 libexpat1 libbz2-1.0 libsqlite3-0 libncursesw6 \
    libreadline8 libtinfo6 zlib1g liblzma5 && \
    rm -rf /var/lib/apt/lists/*

# copy installed packages + AOT shared object
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /build/src/_macd_aot.so /app/src/
COPY src/ /app/src/
WORKDIR /app

ENV PYTHONPATH=/app
CMD ["python", "-m", "wrapper"]
