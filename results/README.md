# Results

Every figure in the top-level README comes from one of these files. Each is
written by the matching script in `scripts/`.

| file | what it measures | figures it backs |
|---|---|---|
| `generation_phase1.{json,md}` | the ablation: nine configurations over the same 37 questions | naive 51.4%, +hybrid 56.5%, +prompt 17.3%, +routing 12.3%, +citations 11.8%, +grounding 8.3% unsupported claims |
| `generation_phase5.{json,md}` | the shipped configuration, run on its own | 7.8% unsupported claims, 80% correct refusal, 100% citation validity |
| `retrieval.{json,md}` | dense vs hybrid vs routed retrieval | MRR 0.860, hit@1 0.81 |
| `selection.{json,md}` | which papers a question is about | 90% recall of the papers an answer needs |
| `routing.{json,md}` | query classification | 93.5% overall, 100% on bibliographic and out-of-scope |
| `confusable.md` | attribution where several papers cover a subject | 87% correct paper cited, 0% contamination |
| `probe_routing.json` | broadly phrased "work on X" questions | the first entry under Limitations |
| `judge.{json,md}` | the faithfulness judge on cases with a known answer | the self-marking caveat under Results |

`make eval` also writes a combined `report.md`. It is gitignored: the files above
are the record.

## Configuration names

In both generation files, `adaptive` is the full engine as it stood when the run
was made. In `generation_phase1` that engine still included relevance grading and
regeneration after a failed citation check. Both were removed afterwards, so in
`generation_phase5` `adaptive` is the pipeline that ships.

## The cross-encoder reranker

The reranker was removed before these files were last written, so its numbers
are recorded here.

| | hit@1 | MRR |
|---|---|---|
| routed retrieval | 0.81 | 0.865 |
| routed retrieval + reranker | 0.65 | 0.771 |

End to end, two otherwise identical runs of the full engine: 8.5% unsupported
claims without the reranker and 8.3% with it, 76.2% correct refusal in both, and
19.7 s against 21.8 s per answer. With 37 questions and 47 judged claims that
rules out a large effect, not a small one. The reranker ranked worse, answered no
better and added two seconds, so it was removed.

## Variance runs

`variance/` holds repeated runs of the shipped configuration, written by
`scripts/variance_run.sh` to measure run-to-run spread. `run-1.json` is also a
test fixture: the hand labels in `tests/eval/labels.yaml` were judged against it,
and `tests/unit/test_rubric_agreement.py` reads it to check that the automated
metric agrees with them. Those tests skip if it is missing.

## Caveats

Every measurement uses 37 or fewer examples, and the 37-question set has informed
enough decisions that it is a development set, not a held-out one.
