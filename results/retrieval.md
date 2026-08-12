# Retrieval evaluation

37 questions, document-level ranking.
No language model involved: these numbers are deterministic.

## Overall

| Configuration | hit@1 | hit@3 | hit@5 | MRR | context precision | found |
|---|---|---|---|---|---|---|
| dense | 0.86 | 0.92 | 0.92 | 0.892 | 0.889 | 34/37 |
| hybrid | 0.86 | 0.95 | 0.95 | 0.901 | 0.676 | 35/37 |
| full | 0.84 | 0.92 | 0.95 | 0.885 | 0.706 | 35/37 |

## hit@3 by question kind

| Configuration | ambiguousdomain | confusable | crossdomain | deepcontent | identifier | paraphrase | topical |
|---|---|---|---|---|---|---|---|
| dense | 0.75 | 1.00 | 1.00 | 0.83 | 0.89 | 1.00 | 1.00 |
| hybrid | 0.75 | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 |
| full | 0.75 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 |

## Questions no configuration answered

- Which control paper avoids needing a mechanical position measurement?

## Questions the reranker changed

- 1 → 2  (identifier) Which paper concerns LSPMSM stator winding inter-turn faults?
- 1 → —  (topical) Comparing field oriented control against model predictive control
- 2 → 1  (identifier) Which work applies artificial bee colony optimisation?
- 1 → 4  (confusable) Which multilevel converter paper appeared in an IET journal?
- — → 1  (deepcontent) Which paper reduces electromagnetic interference by varying the switching frequency?
- 3 → 2  (ambiguousdomain) Which paper compares two established drive control strategies head to head?
