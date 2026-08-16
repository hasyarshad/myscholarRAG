# Retrieval evaluation

37 questions, document-level ranking over the
expanded context blocks the generator would receive.

## Overall

| Configuration | hit@1 | hit@3 | hit@5 | MRR | context precision | found | bypassed |
|---|---|---|---|---|---|---|---|
| dense | 0.81 | 0.89 | 0.89 | 0.851 | 0.838 | 33/37 | 0 |
| hybrid | 0.84 | 0.89 | 0.92 | 0.872 | 0.632 | 34/37 | 0 |
| full | 0.81 | 0.92 | 0.95 | 0.867 | 0.665 | 35/37 | 0 |
| routed | 0.81 | 0.92 | 0.94 | 0.863 | 0.658 | 34/36 | 1 |

## hit@3 by question kind

| Configuration | ambiguousdomain | confusable | crossdomain | deepcontent | identifier | paraphrase | topical |
|---|---|---|---|---|---|---|---|
| dense | 0.75 | 1.00 | 1.00 | 0.67 | 0.89 | 1.00 | 1.00 |
| hybrid | 0.50 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| full | 0.75 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 |
| routed | 0.75 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 |

## Questions no configuration answered

- Which control paper avoids needing a mechanical position measurement?

## Questions routing changed

- 1 -> —  (deepcontent) Which work uses Lyapunov exponents and bifurcation diagrams as evidence?

## Questions the reranker changed

- 1 → 3  (identifier) Which paper concerns LSPMSM stator winding inter-turn faults?
- 2 → —  (topical) Comparing field oriented control against model predictive control
- 1 → 2  (topical) Data-driven predictive control that is computationally efficient
- 2 → 1  (topical) Sliding mode control combined with differential flatness
- 1 → 2  (confusable) Which converter control work targets reducing computational burden specifically?
- 1 → 4  (confusable) Which multilevel converter paper appeared in an IET journal?
- — → 1  (deepcontent) Which paper reduces electromagnetic interference by varying the switching frequency?
- 4 → 2  (ambiguousdomain) Which paper compares two established drive control strategies head to head?
- — → 1  (deepcontent) Which work models an attacker and a defender improving against each other?
