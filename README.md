# myScholar

A retrieval-augmented assistant over a personal research corpus, a
CV, portfolio pages and project write-ups. It answers questions about the work,
and every factual sentence it returns carries a citation that has been verified
against the passage it came from.

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

---

## What it does

**Answers are traceable.** Every factual sentence must carry a citation marker,
and the markers are checked in code before the answer is returned: the label has
to resolve to a passage that was actually retrieved, and any quoted words have to
appear in it. Sentences that fail are removed rather than shown.

**It declines when the corpus does not cover something.** Retrieval always
returns *something* — ask about a topic that is not there and you get the eight
least-irrelevant passages. Refusal is therefore a first-class outcome with its
own reason, naming what the corpus does cover instead of ending in a dead end.

**Questions about the collection are answered from the collection.** "How many
papers did he publish in 2023?" is not a fact inside any paper. It compiles to
SQL over verified metadata rather than being passed to a model.

---

## Components

### Ingestion

PDFs are parsed with Docling, which handles two-column IEEE layout. Headings are
classified into section kinds, references are dropped, and text is chunked at
~350 tokens with overlap, never crossing a section boundary.

Each chunk is embedded with a contextual header prepended:

```
Paper: Weighting Factors Optimization of MPTC (IEEE Access, 2023). Section: III. PROPOSED METHOD.

<chunk text>
```

That header is what lets "what has he achieved on torque ripple?" find a chunk
whose text reads only "the proposed method reduces ripple by 18%".

Bibliographic metadata is resolved from DOIs found by regex in the literal page
text, then replaced wholesale by the Crossref record — so titles, venues and
years are never model output.

### Retrieval

Two branches, fused by Reciprocal Rank Fusion:

- **Dense** — pgvector cosine over an HNSW index.
- **Lexical** — `tsvector` with `ts_rank_cd`. A corpus this dense in identifiers
  (`FCS-MPCC`, `NSGA-II`, `T-Type VSI`) rewards exact term matching, so the
  lexical branch is weighted equal to the dense one.

**Parent expansion** merges each hit into a contiguous span within its section,
keyed on `section_id` so a span can never cross documents. **Document selection**
runs first for cross-paper questions, choosing which papers a question is about
from separately embedded per-document summaries.

### The adaptive loop

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

Citation verification is deterministic i.e. a set lookup, not a model asked whether
it hallucinated. A marker that resolves to nothing, or a factual sentence with no
marker at all, costs that sentence and no more: the answer is trimmed rather than
regenerated, because regenerating was measured to produce a worse one. Refusals
carry their reason, so "I don't have anything on that" and "I ran out of time"
are different answers. Termination is bounded by steps, retrieval rounds and a
wall clock, and property-tested.

There is no relevance grading step and no retry. Both cost accuracy when measured
— see Results.

### Serving

FastAPI with server-sent events, so the stages above appear as they happen rather
than after twenty seconds of spinner. Conversation history is server-side. The
embedder runs as ONNX on CPU, verified at 1.00000 cosine parity with PyTorch;
generation goes to a hosted API. The whole thing runs without a GPU.

---

## Results

Every component was measured by ablation: nine configurations over the same 37
questions, each adding exactly one mechanism to the one below it, same index,
embedder and generator throughout. The six on the path to what ships are below;
the other three are in the table after it.

| Configuration | unsupported claims | claims/question | correct refusal | over-refusal | citation validity |
|---|---|---|---|---|---|
| naive — dense retrieval, one call | 51.4% | 5.7 | 75.0% | 17.6% | — |
| + hybrid retrieval and parent expansion | 56.5% | 5.0 | 65.0% | 11.8% | — |
| **+ the grounding prompt** | **17.3%** | 2.2 | 95.0% | 11.8% | — |
| + routing | 12.3% | 2.2 | 90.0% | **0.0%** | — |
| + citation verification | 11.8% | 2.1 | 90.0% | 0.0% | **100%** |
| **+ groundedness checking — shipped** | **7.8%** | 1.7 | **80.0%** | **0.0%** | **100%** |

**The finding worth taking from this project is that most of the machinery did
not earn its place.** Better retrieval made groundedness *worse* — more relevant
material gives the model more to be confidently wrong about. What fixed it was
the prompt: instructing the model that every sentence must carry a citation
marker took wrong claims from 2.84 per question to 0.38, which is 87% of the
total improvement. Every remaining component together accounts for 0.35.

Three mechanisms a RAG pipeline is usually assumed to need are **absent**, each
because measuring it showed it cost accuracy as well as time:

| absent | why |
|---|---|
| cross-encoder reranker | −13 points hit@1 on near-identical papers |
| relevance grading | +0.05 wrong claims/question, −10 points correct refusal, and **zero** movement on 15 questions built to give it its best case |
| regeneration after failed verification | +0.08 wrong claims/question, −20 points correct refusal, −14 points attribution precision |

Supporting measurements:

| | |
|---|---|
| Retrieval | MRR 0.860, hit@1 0.81 |
| Document selection | 90% recall of the papers an answer needs |
| Attribution on confusable questions | 87% correct paper cited, **0% contamination** |
| Query classification | 93.5%, and 100% on the classes where a misroute fabricates |
| Latency | 11.6s per answer |
| Tests | 694 |

Citation validity is 100% by construction rather than by sampling: every marker
is checked against the retrieved set in code.

Two caveats stated rather than buried. The groundedness check and the
faithfulness judge that scores these runs are the same small model, so the last
row's advantage is partly self-marked. And the 37-question set has informed
enough decisions to be a development set rather than a held-out one.

---

## Running it

Nothing here needs root, Docker, or a GPU.

```bash
make setup                         # install dependencies (uv)
make test                          # 650 tests, no GPU, network or database
uv run myscholar ask "hello"       # runs with fake components
```

The default configuration uses fakes, so it runs anywhere. Copy `.env.example` to
`.env` for real models.

### With the corpus

```bash
make db-setup && make ollama-setup    # once
make up                                # start Postgres and Ollama
make ingest                            # extract, chunk, embed, index

uv run myscholar ask "How are weighting factors tuned?" --trace
uv run myscholar route "How many papers in 2023?"
uv run myscholar search "NSGA-II" --mode dense
uv run myscholar serve                 # API on :8000
```

### Reproducing the numbers

Every figure in this README comes from a script in the repository, run against
the same index.

```bash
make eval          # everything, writes results/report.md
make eval-fast     # retrieval and routing only, deterministic

uv run python scripts/eval_generation.py    # the ablation, rung by rung
uv run python scripts/eval_retrieval.py     # dense vs hybrid vs routed
uv run python scripts/eval_routing.py       # classification accuracy
uv run python scripts/eval_selection.py     # which papers a question is about
uv run python scripts/eval_confusable.py    # attribution across similar papers
uv run myscholar papers check               # corpus invariants
```

Results are written to `results/` as markdown and JSON; `results/README.md` says
which file backs which claim. The generation evaluation takes roughly eight
minutes per configuration, so the ablation is not quick.

### Serving it somewhere

The container is CPU-only: the embedder is exported to ONNX and generation moves
to a hosted API, so the image needs no GPU and scales to zero.

```bash
make docker-build
uv run python scripts/export_onnx.py    # float32, asserted at 1.00000 parity
./scripts/supabase_setup.sh             # hosted Postgres, pgvector, RLS
./scripts/push_index.sh                 # upload the built index
```

---

## Stack

| | | |
|---|---|---|
| Generation | `qwen3:14b` local / `llama-3.3-70b` hosted | one env var swaps them |
| Classification, premise checks | `qwen3:4b-instruct` | many small calls per question |
| Embeddings | `bge-base-en-v1.5`, ONNX for serving | same weights at index and query time, asserted at startup |
| Store | PostgreSQL + pgvector | dense, lexical, relational and summaries in one system |
| Framework | none | the loop is a small state machine |

---

## Limitations

Measured, and stated because a number without its boundaries is not a
measurement.

- **A broadly phrased question about a topic can be answered from one paper.**
  "His work on X" is classified as a single-paper question in 3 of 20 probe
  phrasings, which switches off document selection and the per-document cap, so a
  subject covered by two papers may be described through one of them. Classifying
  it correctly instead needs a signal the classifier does not currently produce.
  The behaviour is pinned by `tests/integration/test_routing_contract.py` against
  the probe set in `tests/eval/work_on_probes.yaml`.
- Answers lean on fewer sources than they are given: 1.1 documents cited on
  average against 1.7–4.3 retrieved.
- Correcting a false premise by asserting the truth — "it reached 99.64%", not
  "that is wrong" — is handled, but a premise asserting a *number* the corpus
  contradicts is still sometimes accepted and explained. Semantic contradiction
  detection is the open problem.
- Lexical ranking has no IDF, so a rare term can lose to several common ones.
- Semantic selection cannot express negation — "published outside power
  electronics" retrieves power electronics.
- Hybrid retrieval trades context precision for recall on identifier queries:
  0.705 against dense-only's 0.817.
- Faithfulness is judged by a 4B model, so the unsupported-claim rate is a floor
  rather than a count.
- Equations are extracted but excluded from embeddings; the assistant discusses
  approaches, not formulae.
- Every measurement uses fewer than 50 examples.

## Licence

MIT — see `LICENSE`. The research papers under `papers/` are not part of this
repository, are not redistributed, and remain the property of their publishers.
