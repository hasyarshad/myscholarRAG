# Query classification

30 questions. Accuracy **96.7%**, critical classes **100.0%**.

Critical classes are bibliographic and out-of-scope: a misroute there causes
confident fabrication or wasted generation, rather than merely a worse search.

| Expected class | n | correct | most common error |
|---|---|---|---|
| `ambiguous` | 2 | 100% | — |
| `bibliographic` | 7 | 100% | — |
| `biographical` | 2 | 100% | — |
| `cross_paper_synthesis` | 3 | 100% | — |
| `methodological` | 4 | 75% | `single_paper_factual` x1 |
| `out_of_scope` | 8 | 100% | — |
| `single_paper_factual` | 4 | 100% | — |

## Misclassified

- `methodological` → `single_paper_factual` — How are weighting factors tuned in predictive torque control?
