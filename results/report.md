# Evaluation

Generated 2026-08-10 13:49 UTC by `make eval`.

## The comparison

| Configuration | unsupported claims | correct refusal | over-refusal |
|---|---|---|---|
| naive | 57.3% | 47.6% | 12.5% |
| retrieved | 56.2% | 33.3% | 12.5% |
| adaptive | 11.4% | 61.9% | 6.2% |

`naive` is the pipeline this project replaces, reimplemented against the same index, the same embedder and the same generator — so the difference is pipeline design and nothing else. `retrieved` keeps the one-shot generation and swaps in hybrid retrieval, reranking and parent expansion. `adaptive` is the full system.

Basis: naive 211 claims over 25 answers, retrieved 242 claims over 28 answers, adaptive 44 claims over 23 answers.

The denominators differ because the configurations write differently. That does not mechanically move the *rate*, which is per sentence — but it does mean the adaptive figure rests on far fewer judgements, so each one is worth about two and a half points. Read the direction, not the decimal.

Part of what separates the rows is style rather than retrieval: the adaptive system's prompt requires every factual sentence to carry a citation, which shapes what it is willing to write. That is a design choice being measured, not a confound — but it is why the middle row matters.

Retrieval alone moved the unsupported-claim rate by 1.1 points. The verification loop moved it by 44.8. Whatever else the retrieval work bought, it was not groundedness — and without the middle row the loop's share would have been credited to it.

`retrieved` keeps the cross-encoder, which the shipped configuration no longer uses — it measured worse on this corpus (see [retrieval](./retrieval.md)). The row is an ablation of retrieval quality, not a description of what runs.

## Stages

| Stage | measures | status | took |
|---|---|---|---|
| [judge](./judge.md) | can the faithfulness judge be trusted | ok | 8s |
| [retrieval](./retrieval.md) | is the right document found | skipped | 0s |
| [routing](./routing.md) | is the right strategy chosen | skipped | 0s |
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
| hybrid | 0.81 | 0.95 | 0.869 | 37 |
| full | 0.65 | 0.89 | 0.773 | 37 |
| routed | 0.78 | 0.89 | 0.836 | 37 |

## Reading these

Correct refusal and over-refusal are only meaningful together. A system that declines everything scores 100% on the first and is useless; one that answers everything scores 0% on the second and is dangerous. The pair measures discrimination.

Unsupported claims are counted per *sentence*, not per answer, and only over answers that were actually given. A refusal contains no claims, so declining more cannot improve the number.

Full detail for each stage is in the linked file.
