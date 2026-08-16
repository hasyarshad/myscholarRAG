# Query classification

46 questions. Accuracy **89.1%**, critical classes **95.5%**.

Critical classes are bibliographic and out-of-scope: a misroute there causes
confident fabrication or wasted generation, rather than merely a worse search.

| Expected class | n | correct | most common error |
|---|---|---|---|
| `ambiguous` | 3 | 100% | — |
| `bibliographic` | 11 | 91% | `biographical` x1 |
| `biographical` | 3 | 100% | — |
| `content` | 1 | 0% | `ambiguous` x1 |
| `cross_paper_synthesis` | 5 | 100% | — |
| `methodological` | 6 | 50% | `cross_paper_synthesis` x2 |
| `out_of_scope` | 11 | 100% | — |
| `single_paper_factual` | 6 | 100% | — |

## Misclassified

- `content` → `ambiguous` — What is the capital of France?
- `methodological` → `single_paper_factual` — How are weighting factors tuned in predictive torque control?
- `bibliographic` → `biographical` — Give me a count of his 2021 output
- `methodological` → `cross_paper_synthesis` — How is switching loss reduced in these converters?
- `methodological` → `cross_paper_synthesis` — What validation strategies appear across the machine learning papers?
