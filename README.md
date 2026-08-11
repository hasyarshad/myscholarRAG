# myScholar-rag

An adaptive, citation-grounded retrieval-augmented generation system over a
personal corpus of 20 research papers and a CV, built to be embedded in a
portfolio website.

> **Status: Phase 1 of 8.** The corpus is indexed. Retrieval is next.
> See [the build plan](#build-phases) below.

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
- [00b — Running services without root](docs/learning/00b-running-services-without-root.md)
- [01 — Chunking and embeddings](docs/learning/01-chunking-and-embeddings.md)
- [01b — What extraction actually produced](docs/learning/01b-extraction-findings.md)

## Quick start

Nothing below needs root, Docker, or a GPU.

```bash
make setup          # install dependencies (uv)
make test           # unit tests — no GPU, no network, no database
uv run myscholar config          # show resolved configuration
uv run myscholar ask "hello"     # walking skeleton, fake components
```

The default configuration uses fakes, so this works on any machine. Copy
`.env.example` to `.env` to switch to real models.

### Local services

Postgres and Ollama both run as ordinary user processes — see
[00b](docs/learning/00b-running-services-without-root.md) for why, and how.

```bash
make db-setup       # once: create the cluster, enable pgvector (conda-provided)
make ollama-setup   # once: install Ollama to ~/.local, pull the models
make up             # start both
make down           # stop both
```

### Building the index

```bash
cp .env.example .env             # then set providers to ollama / torch
uv run myscholar ingest run      # extract, section, chunk, embed, index
uv run myscholar papers list     # what was indexed
uv run myscholar papers sections # how sections were classified
```

Ingestion writes `data/metadata.yaml` with the bibliographic details it
extracted. **Review it and set `verified: true`** — bibliographic questions
("how many papers in 2023?") are answered by SQL over verified rows rather
than by asking a model to recall them, which is where naive RAG invents most
confidently.

Extraction is cached by content hash, so re-running after a chunker change
does not re-parse unchanged PDFs.

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
| 0 | Foundations: config, ports, fakes, CLI | done |
| 1 | Ingestion: PDF parsing, sectioning, chunking, embedding | done |
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
make up         # start Postgres and Ollama
```

## Licence

MIT. The papers under `papers/` are not part of this repository and are not
redistributed.
