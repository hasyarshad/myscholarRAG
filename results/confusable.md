# Attribution precision on confusable questions

15 questions where several papers cover the subject and one is right.

`retrieved` — the expected paper reached the generator. `cited` — it appears
in the answer. `clean` — it is cited and no decoy is. `clean` is the headline;
the gap between `cited` and `clean` is contamination.

| Configuration | retrieved | cited | **clean** | mean latency |
|---|---|---|---|---|
| cited | 87% | 87% | **87%** | 13.1s |
| grounded | 87% | 87% | **87%** | 13.0s |
| graded | 87% | 87% | **87%** | 14.5s |
| graded-strict | 87% | 87% | **87%** | 14.4s |
| adaptive | 87% | 73% | **73%** | 17.5s |

Fifteen questions detect a large effect and nothing else. The rule fixed
before running: keep the grader only on a **20 point or larger** move in
`clean`.

## Questions no configuration got clean

- Which of his converter studies combined H-infinity robust control with predictive control?
- Which of his projects runs entirely in the browser with no server behind it?
