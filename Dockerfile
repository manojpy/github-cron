# ------------------------------------------------------------------
# Stage-1  –  Build environment (g++, Python headers, Numba with pycc)
# ------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
ARG DEBIAN_TAG=bookworm

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_TAG} AS builder

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Python deps (incl. Numba w/ AOT support)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# source code
COPY src/ ./src/

# Produce shared object  (aot_build.py auto-discovers every @njit)
RUN python src/aot_build.py

# ------------------------------------------------------------------
# Stage-2  –  Lean runtime (no compiler, only .so + source)
# ------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_TAG} AS runtime

# runtime-only native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY src/ ./src/
RUN python -m pip install --editable ./src

# copy AOT artefact built in previous stage
COPY --from=builder /build/build/aot/aot_compiled*.so \
     /app/build/aot/

# non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -d /app -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# entry-point uses wrapper.py (AOT-aware)
CMD ["python", "-m", "src.wrapper"]
