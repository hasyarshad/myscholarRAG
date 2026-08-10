# myScholar-rag

An adaptive, citation-grounded retrieval-augmented generation system over a
personal corpus of 20 research papers and a CV, built to be embedded in a
portfolio website.

> **Status: Phase 0 of 8.** The skeleton is in place; ingestion and retrieval
> come next. See [the build plan](#build-phases) below.

## What makes it different from a naive RAG

Most RAG demos retrieve some passages, paste them into a prompt, and print
whatever comes back. That fails in three specific ways, and this system targets
each one:

| Failure | Mitigation |
|---|---|
| Answers claims the sources don't support | Every factual sentence must carry a citation, and citations are **verified deterministically** against the retrieved context before the answer is returned |
| Answers confidently when it should say "I don't know" | Refusal is a first-class outcome with its own state, not an error path — and the refusal names what the corpus *does* cover |
| Retrieves the wrong passages, then can't recover | Retrieved passages are graded for relevance; if they're insufficient the query is rewritten using the grader's stated reasons and retrieval is retried |

## Design notes

Written as the project is built, in [`docs/learning/`](docs/learning/):

- [00 — Architecture, ports, and why abstraction is the deployment story](docs/learning/00-architecture-and-ports.md)

## Quick start

```bash
make setup          # install dependencies (uv)
make test           # unit tests — no GPU, no network, no database
uv run myscholar config          # show resolved configuration
uv run myscholar ask "hello"     # walking skeleton, fake components
```

Copy `.env.example` to `.env` to change providers. The default configuration
uses fakes and needs no models.

## Architecture

```
Next.js portfolio (Vercel)
  └─ route handler proxy
        └─ FastAPI
              ├─ ONNX embedder + reranker   (CPU, ~110MB / ~280MB)
              ├─ Postgres + pgvector
              └─ ChatModel → Ollama (dev) | Groq (prod)
```

Document embeddings are computed once on local GPUs and pushed to hosted
Postgres, so production never needs a GPU. Swapping between local and hosted
inference is a single environment variable, because application code depends on
interfaces in [`ports/`](src/myscholar/ports/) rather than on any specific
backend — [`container.py`](src/myscholar/container.py) is the only module that
knows which implementation is in use.

## Build phases

| Phase | Scope | State |
|---|---|---|
| 0 | Foundations: config, ports, fakes, CLI | in progress |
| 1 | Ingestion: PDF parsing, sectioning, chunking, embedding | pending |
| 2 | Retrieval: hybrid search, RRF, reranking, parent expansion | pending |
| 3 | Query translation and routing | pending |
| 4 | Adaptive loop: grading, corrective retry, citation verification | pending |
| 5 | Evaluation: golden set, hallucination and refusal metrics | pending |
| 6 | API and chat widget | pending |
| 7 | Deployment: local containers, then free cloud | pending |

## Development

```bash
make dev        # install with GPU and serving extras
make check      # lint, typecheck, test — what CI runs
make db-up      # Postgres with pgvector on :5433
```

## Licence

MIT. The papers under `papers/` are not part of this repository and are not
redistributed.
