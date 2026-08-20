# Query classification

46 questions. Accuracy **80.4%**, critical classes **95.5%**.

Critical classes are bibliographic and out-of-scope: a misroute there causes
confident fabrication or wasted generation, rather than merely a worse search.

| Expected class | n | correct | most common error |
|---|---|---|---|
| `ambiguous` | 3 | 100% | — |
| `bibliographic` | 11 | 91% | `cross_paper_synthesis` x1 |
| `biographical` | 3 | 100% | — |
| `content` | 1 | 100% | — |
| `cross_paper_synthesis` | 5 | 100% | — |
| `methodological` | 6 | 33% | `cross_paper_synthesis` x4 |
| `out_of_scope` | 11 | 100% | — |
| `single_paper_factual` | 6 | 33% | `cross_paper_synthesis` x4 |

## Misclassified

- `single_paper_factual` → `cross_paper_synthesis` — What filter is used in the grid-tied inverter paper?
- `single_paper_factual` → `cross_paper_synthesis` — Which machine does the meta-learner stacking paper diagnose?
- `methodological` → `cross_paper_synthesis` — What approach is taken to sensorless control?
- `methodological` → `cross_paper_synthesis` — How is chaos analysed quantitatively?
- `bibliographic` → `cross_paper_synthesis` — Give me a count of his 2021 output
- `single_paper_factual` → `cross_paper_synthesis` — What sampling frequency does the LSPMSM fault study use?
- `single_paper_factual` → `cross_paper_synthesis` — In the UAV paper, which estimator is applied?
- `methodological` → `cross_paper_synthesis` — How is switching loss reduced in these converters?
- `methodological` → `cross_paper_synthesis` — What validation strategies appear across the machine learning papers?
