# Query classification

46 questions. Accuracy **93.5%**, critical classes **100.0%**.

Critical classes are bibliographic and out-of-scope: a misroute there causes
confident fabrication or wasted generation, rather than merely a worse search.

| Expected class | n | correct | most common error |
|---|---|---|---|
| `ambiguous` | 3 | 100% | — |
| `bibliographic` | 11 | 100% | — |
| `biographical` | 3 | 100% | — |
| `content` | 1 | 0% | `bibliographic` x1 |
| `cross_paper_synthesis` | 5 | 100% | — |
| `methodological` | 6 | 67% | `single_paper_factual` x1 |
| `out_of_scope` | 11 | 100% | — |
| `single_paper_factual` | 6 | 100% | — |

## Misclassified

- `content` → `bibliographic` — What is the capital of France?
- `methodological` → `single_paper_factual` — How are weighting factors tuned in predictive torque control?
- `methodological` → `cross_paper_synthesis` — What validation strategies appear across the machine learning papers?
