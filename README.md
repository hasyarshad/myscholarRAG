# myScholar

A retrieval-augmented assistant over my research papers, CV and project
write-ups. Every factual sentence in an answer carries a citation, and each
citation is checked in code against the passage it cites before the answer is
returned.

Live demo: [haseeb-arshad.me/myscholar](https://haseeb-arshad.me/myscholar). The
corpus behind the demo is a fixed snapshot.

```
$ myscholar ask "How are weighting factors tuned in predictive torque control?" --trace

  retrieving    6 passages from 2 papers          0.5s
  generating    541 chars                         6.1s
  verifying     citations valid                   8.2s
  verifying     grounded                          9.1s

  Heuristic optimisation replaces manual tuning: NSGA-II searches the weighting
  factor space and TOPSIS selects one solution from the Pareto front [C2]. The
  cost function weights torque error against flux error through λ1 and λ2 [C3].

  [C2] Weighting Factors Optimization of MPTC · VI. STATOR FLUX · p.8-9
  [C3] Weighting Factors Optimization of MPTC · IV. PROPOSED METHOD · p.5
```

## What it does

- **Answers are traceable.** Each factual sentence must carry a citation marker.
  Before the answer is returned, every marker has to resolve to a passage that
  was actually retrieved, and any quoted words have to appear in that passage.
  Sentences that fail are removed.
- **It declines when the corpus doesn't cover a question.** Retrieval always
  returns something, even for an off-topic question, so refusal is an explicit
  outcome with a reason and a list of what the corpus does cover.
- **Questions about the collection are answered from metadata.** "How many papers
  were published in 2023?" isn't answered by any single paper. It compiles to SQL
  over Crossref-verified metadata and never reaches a model.

## How it works

### Ingestion

Docling parses the PDFs, including two-column IEEE layouts. Headings are
classified into section types, reference lists are dropped, and text is chunked
at about 350 tokens with overlap, never across a section boundary.

Each chunk is embedded with a header naming its paper and section:

```
Paper: Weighting Factors Optimization of MPTC (IEEE Access, 2023). Section: III. PROPOSED METHOD.

<chunk text>
```

The header lets a question like "what has he achieved on torque ripple?" find a
chunk whose text only says "the proposed method reduces ripple by 18%".

DOIs are found by regex in the page text, and each paper's metadata is replaced
by its Crossref record, so titles, venues and years never come from a model.

### Retrieval

Two branches, fused by reciprocal rank fusion:

- **Dense:** pgvector cosine similarity over an HNSW index.
- **Lexical:** `tsvector` with `ts_rank_cd`. The corpus is dense in identifiers
  (`FCS-MPCC`, `NSGA-II`, `T-Type VSI`) that reward exact matching, so the
  lexical branch has the same weight as the dense one.

Each hit is expanded to a contiguous span within its section. For questions that
span several papers, document selection runs first and picks the relevant papers
using a separately embedded summary of each one.

### The pipeline

```
question → classify ─┬→ out of scope    → canned reply
                     ├→ ambiguous       → ask which paper
                     ├→ bibliographic   → SQL over verified metadata
                     └→ content         → retrieve
                                             ↓
                                         check the question's premise
                                             ↓
                                         generate
                                             ↓
                                         verify citations ──── drop what fails
                                             ↓
                                         check groundedness ── drop what fails
                                             ↓
                                         answer, or refuse if nothing survived
```

Citation verification is a set lookup, not a model call. A sentence whose marker
doesn't resolve, or a factual sentence with no marker, is dropped and the rest of
the answer stands. Regenerating the answer instead was measured and made answers
worse (see Results). Every refusal carries its reason, so "the corpus doesn't
cover this" and "the time budget ran out" are different responses. The loop is
bounded by step count, retrieval rounds and wall-clock time, and the bound is
property-tested.

### Serving

FastAPI with server-sent events, so each stage is reported as it happens.
Conversation history is kept on the server. The embedder runs as ONNX on CPU
(cosine similarity 1.00000 against PyTorch), and generation goes to a hosted model
through the OpenAI chat-completions API, so the service needs no GPU.

## Results

Each component was measured by ablation: nine configurations over the same 37
questions, each adding one mechanism to the previous one, with the index,
embedder and generator held fixed. The six on the path to the shipped pipeline:

| Configuration | unsupported claims | claims/question | correct refusal | over-refusal | citation validity |
|---|---|---|---|---|---|
| naive: dense retrieval, one call | 51.4% | 5.7 | 75.0% | 17.6% | n/a |
| + hybrid retrieval and span expansion | 56.5% | 5.0 | 65.0% | 11.8% | n/a |
| **+ citation-marker prompt** | **17.3%** | 2.2 | 95.0% | 11.8% | n/a |
| + routing | 12.3% | 2.2 | 90.0% | **0.0%** | n/a |
| + citation verification | 11.8% | 2.1 | 90.0% | 0.0% | **100%** |
| **+ groundedness check** | **8.3%** | 1.6 | 90.0% | 0.0% | **100%** |

Run again on its own, the shipped configuration measured 7.8% unsupported claims,
1.7 claims per question, 80% correct refusal, 0% over-refusal and 100% citation
validity.

**Most of the improvement came from the prompt.** Better retrieval made
groundedness worse (56.5% against 51.4%): given more relevant material, the model
wrote more claims the passages didn't support. Instructing it that every sentence
must carry a citation marker took wrong claims from 2.84 per question to 0.38,
which is 87% of the total improvement. Every other component together accounts
for the remaining 0.35.

Three components that RAG pipelines commonly include were measured and removed:

| removed | measured effect |
|---|---|
| cross-encoder reranker | hit@1 fell from 0.81 to 0.65 and MRR from 0.865 to 0.771; end to end it was no better (8.3% against 8.5% unsupported) and 2 s slower per answer |
| relevance grading | against the shipped configuration: +0.05 wrong claims per question, −10 points correct refusal, and no change on 15 questions where several papers cover the subject |
| regeneration after a failed check | against the configuration it was added to: +0.08 wrong claims per question, −5 points correct refusal, −14 points correct attribution on those 15 questions |

Other measurements:

| | |
|---|---|
| Retrieval | MRR 0.860, hit@1 0.81 |
| Document selection | 90% recall of the papers an answer needs |
| Attribution where several papers cover a subject | 87% correct paper cited, 0% citing a wrong one |
| Query classification | 93.5% overall, 100% on bibliographic and out-of-scope questions |
| Latency | 11.6 s per answer with local models, 3.3 s with the hosted model |
| Tests | 701: 657 run offline, 44 need Postgres and Ollama |

Citation validity is 100% by construction: every marker is checked against the
retrieved set in code.

Caveats:

- The groundedness check and the faithfulness judge that scores these runs use
  the same small model, so the shipped configuration's advantage is partly
  self-marked.
- The 37-question set has informed enough decisions that it is a development
  set, not a held-out one.
- All figures were measured with the local models, `qwen3:14b` and
  `qwen3:4b-instruct`, because a free hosted tier's rate limits would dominate an
  evaluation run. `tests/integration/test_prompt_contract.py` checks that the
  prompt contract also holds on the hosted model.

`results/README.md` lists which file backs each figure.

## Running it

```bash
make setup                         # install dependencies with uv
make test                          # 657 offline tests: no GPU, network or database
uv run myscholar ask "hello"       # runs on fake components
```

With no `.env`, every component is a fake, so this runs anywhere. Copy
`.env.example` to `.env` to use real models.

### With the corpus

Postgres and Ollama run as user processes, so this needs no root or Docker. The
default local configuration embeds and generates on a GPU.

```bash
make db-setup && make ollama-setup     # once
make up                                # start Postgres and Ollama
make ingest                            # extract, chunk, embed, index

uv run myscholar ask "How are weighting factors tuned?" --trace
uv run myscholar route "How many papers in 2023?"
uv run myscholar search "NSGA-II" --mode dense
uv run myscholar serve                 # API on :8000
```

### Reproducing the numbers

```bash
make eval          # every evaluation (~30 min)
make eval-fast     # retrieval and routing only, deterministic

uv run python scripts/eval_generation.py    # the ablation
uv run python scripts/eval_retrieval.py     # dense vs hybrid vs routed
uv run python scripts/eval_routing.py       # classification accuracy
uv run python scripts/eval_selection.py     # which papers a question is about
uv run python scripts/eval_confusable.py    # attribution across similar papers
uv run myscholar papers check               # corpus invariants
```

Results are written to `results/` as Markdown and JSON. The generation
evaluation takes about eight minutes per configuration.

### Deploying

The container is CPU-only and scales to zero. Generation uses Google AI Studio
(`gemini-3.5-flash-lite`) through its OpenAI-compatible endpoint. Any provider
works if it supports schema-constrained output (classification, the premise check
and the groundedness check are structured calls) and accepts a request of about
10,000 tokens, which is what eight context blocks come to. Switching provider
means changing two environment variables.

```bash
make docker-build      # builds the image; exports the embedder and checks parity
make supabase-setup    # schema and search indexes on Supabase
psql "$SUPABASE_DB_URL" -f scripts/supabase_harden.sql   # close Supabase's public REST access
make push-index        # upload the local index
make deploy            # Cloud Build and Cloud Run
```

Supabase exposes every `public` table through its REST API by default.
`supabase_harden.sql` removes those grants and enables row-level security with no
policies; the app connects as the table owner, so it is unaffected.

`web/` is a minimal Next.js page with the chat widget and a server-side proxy
that adds the API's shared secret.

## Stack

| | | |
|---|---|---|
| Generation | `qwen3:14b` locally, `gemini-3.5-flash-lite` hosted | selected by one environment variable |
| Classification, premise checks | `qwen3:4b-instruct` | many small calls per question |
| Embeddings | `bge-base-en-v1.5`, ONNX for serving | the same weights at index and query time, checked at startup |
| Store | PostgreSQL + pgvector | dense, lexical, relational data and summaries in one database |
| Orchestration | no framework | the pipeline is a small state machine in `src/myscholar/reasoning/` |

## Repository layout

| path | contents |
|---|---|
| `src/myscholar/api/` | FastAPI app, request limits, SSE |
| `src/myscholar/query/` | classification, routing, follow-up resolution, bibliographic SQL |
| `src/myscholar/retrieval/` | hybrid retrieval, rank fusion, span expansion, document selection |
| `src/myscholar/generation/` | answer generation and citation verification |
| `src/myscholar/reasoning/` | the pipeline state machine |
| `src/myscholar/ingest/` | sectioning, chunking, metadata, summaries |
| `src/myscholar/ports/`, `adapters/` | interfaces and their implementations: Ollama and OpenAI-compatible LLMs, ONNX and PyTorch embedders, Docling, Crossref, Postgres, and fakes for tests |
| `src/myscholar/evaluation/` | baselines, ablation configurations, the faithfulness judge |
| `scripts/` | evaluation harness, ONNX export, database and deployment scripts |
| `tests/` | unit and integration tests; `tests/eval/` holds the evaluation sets |
| `results/` | evaluation output behind every figure above |

## Limitations

- **A broadly phrased question about a topic can be answered from one paper.**
  "His work on X" is classified as a single-paper question in 3 of 20 probe
  phrasings, which turns off document selection and the per-document cap, so a
  subject covered by two papers may be described through one of them. Fixing it
  needs a signal the classifier doesn't produce. The behaviour is pinned by
  `tests/integration/test_routing_contract.py` against
  `tests/eval/work_on_probes.yaml`.
- Answers cite fewer sources than they are given: 1.1 documents on average,
  against 1.7 to 4.3 retrieved.
- A false premise is corrected when the corpus states the true fact ("it reached
  99.64%"), but a premise asserting a number the corpus contradicts is still
  sometimes accepted.
- Lexical ranking has no IDF, so a rare term can lose to several common ones.
- Document selection can't express negation: "published outside power
  electronics" retrieves power electronics.
- Hybrid retrieval trades context precision for recall on identifier queries:
  0.705 against 0.817 for dense only.
- Faithfulness is judged by a 4B model, so the unsupported-claim rate is a lower
  bound.
- Equations are extracted but not embedded, so the assistant discusses methods,
  not formulae.
- Every measurement uses fewer than 50 examples.

## Licence

MIT, see `LICENSE`. The research papers are not included in this repository and
remain the property of their publishers.
