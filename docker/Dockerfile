# The production image: FastAPI and ONNX Runtime on CPU, built for Cloud Run.
#
# PyTorch is not installed in the final image; only the `serve` extra is. The
# embedder is exported to ONNX in a separate build stage, which also checks the
# export against PyTorch, so a bad export fails the build.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# --- dependencies -----------------------------------------------------------
FROM base AS deps
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[serve]"

# --- model ------------------------------------------------------------------
# Exports the embedder to ONNX and verifies parity. Only the graph is copied
# forward. CPU-only torch keeps the CUDA libraries out of the build.
FROM base AS model
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/export_onnx.py scripts/
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir ".[export]" sentence-transformers \
    && python scripts/export_onnx.py --skip-benchmark

# --- runtime ----------------------------------------------------------------
FROM base AS runtime

RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=model --chown=user:user /app/artifacts/embedder-onnx /app/artifacts/embedder-onnx
COPY --chown=user:user src/ src/
COPY --chown=user:user pyproject.toml README.md ./

USER user

# Configuration only. MYSCHOLAR_DATABASE__URL, MYSCHOLAR_LLM__API_KEY and
# MYSCHOLAR_API__SHARED_SECRET are injected by the host at run time.
#
# The hosted model must support schema-constrained output (classification, the
# premise check and the groundedness check are structured calls) and accept a
# ~10k-token request. SEED is empty because this endpoint rejects unknown parameters.
ENV HOME=/home/user \
    PYTHONPATH=/app/src \
    MYSCHOLAR_EMBEDDING__PROVIDER=onnx \
    MYSCHOLAR_EMBEDDING__ONNX_PATH=/app/artifacts/embedder-onnx \
    MYSCHOLAR_LLM__PROVIDER=openai_compatible \
    MYSCHOLAR_LLM__BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
    MYSCHOLAR_LLM__GENERATOR_MODEL=gemini-3.5-flash-lite \
    MYSCHOLAR_LLM__UTILITY_MODEL=gemini-3.5-flash-lite \
    MYSCHOLAR_LLM__SEED= \
    MYSCHOLAR_DATABASE__MIGRATE_ON_STARTUP=false

# Cloud Run sets PORT. One worker: a second would load another copy of the model
# onto the same two cores, and concurrency comes from asyncio.
ENV PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ['PORT']}/healthz\")"

# `sh -c` expands $PORT; `exec` makes uvicorn PID 1 so it receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn myscholar.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
