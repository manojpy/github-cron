# =============================================================================
# MULTI-STAGE BUILD: Optimized for AOT Compilation
# =============================================================================

# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip>=25.3
RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code directly to build root
COPY src/ .

# ✅ AOT Build
# Using the updated aot_build.py logic (with internal cores)
ARG AOT_STRICT=1
RUN python aot_build.py && \
    ls -lh macd_aot_compiled*.so 2>/dev/null || \
    ( [ "$AOT_STRICT" != "1" ] || (echo "❌ AOT Compilation Failed" && exit 1) )

# ---------- FINAL STAGE ----------
FROM python:3.11-slim AS final

# Install runtime dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends -qq \
    libtbb12 tzdata ca-certificates bc && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip>=25.3

# Setup non-root user
RUN useradd --uid 1000 -m appuser && \
    mkdir -p /app/src /app/logs && \
    chown -R appuser:appuser /app

WORKDIR /app/src

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and compiled AOT binaries
COPY --from=builder --chown=appuser:appuser /build/ /app/src/

# Switch to non-root user
USER appuser

# ✅ SAFE VERIFICATION
# We test only the compiled binary import. 
# This bypasses Pydantic/Telegram validation because we don't run macd_unified.py
RUN python -c "import macd_aot_compiled; print('✅ AOT Binary Verified and Loadable')"

# Set runtime environment
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PYTHONUNBUFFERED=1

CMD ["python", "macd_unified.py"]