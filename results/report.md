# Evaluation

Generated 2026-08-10 12:01 UTC by `make eval`.

## The comparison

| Configuration | unsupported claims | correct refusal | over-refusal |
|---|---|---|---|
| naive | 65.4% | 52.4% | 18.8% |
| retrieved | 65.0% | 42.9% | 12.5% |
| adaptive | 12.8% | 71.4% | 12.5% |

`naive` is the pipeline this project replaces, reimplemented against the same index, the same embedder and the same generator — so the difference is pipeline design and nothing else. `retrieved` keeps the one-shot generation and swaps in hybrid retrieval, reranking and parent expansion. `adaptive` is the full system.

Basis: naive 179 claims over 23 answers, retrieved 197 claims over 26 answers, adaptive 39 claims over 20 answers.

The denominators differ because the configurations write differently. That does not mechanically move the *rate*, which is per sentence — but it does mean the adaptive figure rests on far fewer judgements, so each one is worth about two and a half points. Read the direction, not the decimal.

Part of what separates the rows is style rather than retrieval: the adaptive system's prompt requires every factual sentence to carry a citation, which shapes what it is willing to write. That is a design choice being measured, not a confound — but it is why the middle row matters. Better retrieval alone moved the rate by 0.4 points. The verification loop moved it by 52.

## Stages

| Stage | measures | status | took |
|---|---|---|---|
| [judge](./judge.md) | can the faithfulness judge be trusted | ok | 8s |
| [retrieval](./retrieval.md) | is the right document found | ok | 415s |
| [routing](./routing.md) | is the right strategy chosen | ok | 12s |
| [generation](./generation.md) | is the answer supported, and does it decline what it cannot answer | skipped | 0s |

## Judge calibration

The faithfulness numbers above are only as good as the model producing them, so it is measured first, on cases where the right answer is not in doubt.

| Measure | Value |
|---|---|
| Recognises a supported claim | 100% |
| Catches an unsupported claim | 100% |

## Retrieval

| Configuration | hit@1 | hit@3 | MRR | n |
|---|---|---|---|---|
| dense | 0.81 | 0.92 | 0.865 | 37 |
| hybrid | 0.84 | 0.95 | 0.887 | 37 |
| full | 0.65 | 0.89 | 0.764 | 37 |
| routed | 0.62 | 0.89 | 0.744 | 37 |

## Reading these

Correct refusal and over-refusal are only meaningful together. A system that declines everything scores 100% on the first and is useless; one that answers everything scores 0% on the second and is dangerous. The pair measures discrimination.

Unsupported claims are counted per *sentence*, not per answer, and only over answers that were actually given. A refusal contains no claims, so declining more cannot improve the number.

Full detail for each stage is in the linked file.
