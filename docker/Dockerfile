# The production image: FastAPI, ONNX Runtime, no PyTorch, no CUDA.
#
# Built for Google Cloud Run (CPU-only, scale to zero), and the same image runs
# anywhere that takes a container. Two constraints shape everything here:
#
#   1. **PyTorch must not be installed.** Not "not used" — not installed. It is
#      2GB, and the whole point of the ONNX export is that serving does not need
#      it. Only the `serve` extra is installed, so an accidental `import torch`
#      fails loudly at build time rather than quietly bloating the image.
#
#   2. **The model is built here, not fetched.** 416MB of weights do not belong
#      in a git repository, and depending on a model host to be up is a
#      dependency the deployment does not need. The export runs in a build
#      stage that has PyTorch, and only the resulting graph is copied forward —
#      so the parity check that proves the export matches PyTorch runs on every
#      build, and a bad export fails the build instead of the answers.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# --- dependencies -----------------------------------------------------------
# Copied before the source so that editing a Python file does not reinstall
# 200MB of wheels. This layer changes only when pyproject.toml does.
FROM base AS deps
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[serve]"

# --- model ------------------------------------------------------------------
# A throwaway stage. It installs PyTorch, exports the embedder to ONNX and
# verifies the result against PyTorch itself — then everything except the graph
# is discarded. The 2GB of framework never reaches the final image, which is the
# entire point of exporting in the first place.
#
# CPU-only torch, from the index that publishes it: the GPU build is several
# gigabytes of CUDA that a build machine has no use for.
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

# An unprivileged user, created explicitly rather than inherited. A container
# that runs as root works fine until it meets a host that forbids it, and then
# fails on permissions rather than on anything informative.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=model --chown=user:user /app/artifacts/embedder-onnx /app/artifacts/embedder-onnx
COPY --chown=user:user src/ src/
COPY --chown=user:user pyproject.toml README.md ./

USER user

# Configuration, not secrets. The hosted model is chosen here because two
# requirements eliminate most free tiers, and both were measured rather than
# assumed:
#
#   * The pipeline needs schema-constrained output. Classification, the premise
#     check and the groundedness gate are all `structured` calls, and a provider
#     that answers with prose disables them *silently* — every caller treats a
#     validation error as an outage and degrades quietly.
#   * One generation request carries eight context blocks, about 10,000 tokens.
#     A free tier that caps a whole minute at 8,000 can never answer once.
#
# Google AI Studio satisfies both: a million-token context, reliable JSON, and
# 3.3s per question against 11.6s for the local pair. `flash-lite` is pinned
# rather than `-latest` so the behaviour the tests pin cannot drift.
#
# SEED is empty on purpose: a fixed seed keeps evaluation runs comparable, but
# this endpoint rejects unknown parameters instead of ignoring them.
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

# Everything above is configuration, not secrets. MYSCHOLAR_DATABASE__URL,
# MYSCHOLAR_LLM__API_KEY and MYSCHOLAR_API__SHARED_SECRET are injected by the
# host at run time and must never appear in a layer.

# One worker deliberately. Two cores are shared with ONNX Runtime's own thread
# pool, and a second worker would load a second copy of the model to contend
# for the same cores. Concurrency here comes from asyncio, which is right for a
# service whose latency is dominated by waiting on a hosted LLM.
#
# The port is read from the environment because Cloud Run assigns it and will
# not negotiate: a container listening on a hardcoded port fails its health
# check with no useful message. 8080 is the default it injects, and the shell
# form of CMD is required for the variable to be expanded at all.
ENV PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ['PORT']}/healthz\")"

# JSON form so the container receives signals directly rather than through a
# shell that would swallow SIGTERM — a container that ignores SIGTERM is killed
# after the grace period instead of shutting down cleanly. `sh -c` is still
# needed to expand $PORT, and `exec` hands PID 1 to uvicorn.
CMD ["sh", "-c", "exec uvicorn myscholar.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
