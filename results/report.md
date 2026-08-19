# Evaluation

Generated 2026-08-10 16:40 UTC by `make eval`.

## The comparison

| Configuration | unsupported claims | correct refusal | over-refusal |
|---|---|---|---|
| naive | 53.6% | 60.0% | 5.9% |
| retrieved | 52.8% | 55.0% | 5.9% |
| adaptive | 8.7% | 75.0% | 0.0% |

`naive` is the pipeline this project replaces, reimplemented against the same index, the same embedder and the same generator — so the difference is pipeline design and nothing else. `retrieved` keeps the one-shot generation and swaps in hybrid retrieval, reranking and parent expansion. `adaptive` is the full system.

Basis: naive 250 claims over 28 answers, retrieved 250 claims over 29 answers, adaptive 46 claims over 25 answers.

The denominators differ because the configurations write differently. That does not mechanically move the *rate*, which is per sentence — but it does mean the adaptive figure rests on far fewer judgements, so each one is worth about two and a half points. Read the direction, not the decimal.

Part of what separates the rows is style rather than retrieval: the adaptive system's prompt requires every factual sentence to carry a citation, which shapes what it is willing to write. That is a design choice being measured, not a confound — but it is why the middle row matters.

Retrieval alone moved the unsupported-claim rate by 0.8 points. The verification loop moved it by 44.1. Whatever else the retrieval work bought, it was not groundedness — and without the middle row the loop's share would have been credited to it.

`retrieved` keeps the cross-encoder, which the shipped configuration no longer uses — it measured worse on this corpus (see [retrieval](./retrieval.md)). The row is an ablation of retrieval quality, not a description of what runs.

## Stages

| Stage | measures | status | took |
|---|---|---|---|
| [judge](./judge.md) | can the faithfulness judge be trusted | ok | 9s |
| [selection](./selection.md) | are the right papers brought into scope | ok | 6s |
| [retrieval](./retrieval.md) | is the right document found | ok | 803s |
| [routing](./routing.md) | is the right strategy chosen | ok | 13s |
| [generation](./generation.md) | is the answer supported, and does it decline what it cannot answer | ok | 1863s |

## Judge calibration

The faithfulness numbers above are only as good as the model producing them, so it is measured first, on cases where the right answer is not in doubt.

| Measure | Value |
|---|---|
| Recognises a supported claim | 100% |
| Catches an unsupported claim | 100% |

## Does the cross-encoder earn its place?

| adaptive | unsupported claims | correct refusal | latency | claims judged |
|---|---|---|---|---|
| without reranking | 8.7% | 75.0% | 30.4s | 46 |
| with reranking | 8.3% | 76.2% | 21.8s | 48 |

Nothing separates them end to end — one claim in forty-seven, and identical refusal behaviour — while reranking costs about two seconds a question.

That sits against a retrieval measurement where the cross-encoder costs 0.09 MRR and 17 points of hit@1. Both are true, and the reconciliation is the interesting part: **the loop does not care much where in the top ten the right passage sits.** Grading filters what came back, parent expansion hands the generator generous context, and an answer is built from several blocks. Rank 1 against rank 3 barely moves an answer.

So hit@1 and MRR, the numbers this field optimises, are weak proxies for answer quality in a system with a verification loop behind them. Worth knowing before spending a week on a reranker.

The decision is still to leave it off: it is measurably worse at ranking, indistinguishable at answering, and slower. But the honest caveat is that 37 questions and 47 judged claims cannot detect a small difference, so this rules out a large effect rather than proving no effect.

## Document selection

Which papers a cross-paper question is about, decided before any passage is fetched. The retrieval set below cannot measure this — it is 27 single-paper questions and 9 methodological ones, with no synthesis at all.

| Measure | Value |
|---|---|
| Recall of needed documents | 76% |
| Questions with nothing missing | 50% |

## Retrieval

| Configuration | hit@1 | hit@3 | MRR | n |
|---|---|---|---|---|
| dense | 0.81 | 0.92 | 0.865 | 37 |
| hybrid | 0.81 | 0.95 | 0.869 | 37 |
| full | 0.65 | 0.89 | 0.773 | 37 |
| routed | 0.78 | 0.95 | 0.856 | 37 |
| routed+rerank | 0.65 | 0.89 | 0.769 | 37 |

## Reading these

Correct refusal and over-refusal are only meaningful together. A system that declines everything scores 100% on the first and is useless; one that answers everything scores 0% on the second and is dangerous. The pair measures discrimination.

Unsupported claims are counted per *sentence*, not per answer, and only over answers that were actually given. A refusal contains no claims, so declining more cannot improve the number.

Full detail for each stage is in the linked file.
