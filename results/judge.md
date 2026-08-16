# Faithfulness judge calibration

16 cases where the right answer is not in doubt.

| Measure | Value | Basis |
|---|---|---|
| Recognises a supported claim | **100%** | 8 cases |
| Catches an unsupported claim | **100%** | 8 cases |

Reported separately because they fail in opposite directions. A judge biased toward *unsupported* makes every configuration look bad while roughly preserving the comparison; one biased toward *supported* makes everything look clean and destroys it. A single accuracy figure hides which is happening.

## By case shape

| Shape | correct | n |
|---|---|---|
| `added_cause` | 1/1 | 1 |
| `added_detail` | 1/1 | 1 |
| `invented_number` | 2/2 | 2 |
| `inverted_finding` | 3/3 | 3 |
| `paraphrase` | 3/3 | 3 |
| `synthesis` | 2/2 | 2 |
| `verbatim` | 3/3 | 3 |
| `wrong_attribution` | 1/1 | 1 |
