# ---------- STAGE 1: UV INSTALLER ----------
FROM python:3.11-slim-bookworm AS uv-installer
RUN pip install --no-cache-dir uv==0.5.15

# ---------- STAGE 2: DEPENDENCIES BUILDER ----------
FROM python:3.11-slim-bookworm AS deps-builder
COPY --from=uv-installer /usr/local/bin/uv /usr/local/bin/uv
COPY --from=uv-installer /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# ---------- STAGE 3: AOT COMPILER ----------
FROM deps-builder AS aot-builder
WORKDIR /build

# Copy both required files into the workdir so they can import each other
COPY src/numba_functions_shared.py ./
COPY src/aot_build.py ./

# Create output dir and run build
RUN mkdir -p /build/out
ARG AOT_STRICT=1
RUN python aot_build.py --output-dir /build/out --module-name macd_aot_compiled --verify || \
    (if [ "$AOT_STRICT" = "1" ]; then exit 1; else echo "⚠️ AOT Build Failed, but strict mode is off"; fi)

# ---------- STAGE 4: RUNTIME ----------
FROM python:3.11-slim-bookworm
WORKDIR /app/src

COPY --from=deps-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Copy the compiled binary and normalize the name
COPY --from=aot-builder /build/out/macd_aot_compiled*.so ./macd_aot_compiled.so

# Copy app source
COPY src/ ./
COPY config_macd.json ./

ENV PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    AOT_LIB_PATH=/app/src

ENTRYPOINT ["python", "macd_unified.py"]