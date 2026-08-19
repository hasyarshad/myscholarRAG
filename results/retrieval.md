# Retrieval evaluation

37 questions, document-level ranking over the
expanded context blocks the generator would receive.

## Overall

| Configuration | hit@1 | hit@3 | hit@5 | MRR | context precision | found | bypassed |
|---|---|---|---|---|---|---|---|
| dense | 0.81 | 0.92 | 0.92 | 0.865 | 0.840 | 34/37 | 0 |
| hybrid | 0.81 | 0.95 | 0.95 | 0.869 | 0.718 | 35/37 | 0 |
| full | 0.65 | 0.89 | 0.95 | 0.773 | 0.577 | 36/37 | 0 |
| routed | 0.78 | 0.95 | 0.95 | 0.856 | 0.746 | 35/37 | 0 |
| routed+rerank | 0.65 | 0.89 | 0.95 | 0.769 | 0.599 | 36/37 | 0 |

## hit@3 by question kind

| Configuration | ambiguousdomain | confusable | crossdomain | deepcontent | identifier | paraphrase | topical |
|---|---|---|---|---|---|---|---|
| dense | 0.75 | 1.00 | 1.00 | 0.83 | 1.00 | 0.88 | 1.00 |
| hybrid | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 |
| full | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 0.75 | 0.80 |
| routed | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 |
| routed+rerank | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 0.88 | 0.60 |

## Questions no configuration answered

- How is unbalanced rotation in rotating machinery detected automatically?

## Questions routing changed

- 2 -> 1  (identifier) Which paper uses NSGA-II with TOPSIS for weighting factor selection?
- 3 -> 1  (identifier) Which paper concerns LSPMSM stator winding inter-turn faults?
- 4 -> 2  (paraphrase) How is torque ripple reduced without using a speed sensor?
- 6 -> 2  (topical) Comparing field oriented control against model predictive control
- 2 -> 1  (crossdomain) Which paper validates a three-factor measurement scale?
- 2 -> 1  (identifier) Which work applies artificial bee colony optimisation?
- 5 -> 1  (confusable) Which multilevel converter paper appeared in an IET journal?
- 1 -> 3  (deepcontent) Which paper reduces electromagnetic interference by varying the switching frequency?
- 3 -> 1  (ambiguousdomain) Which control paper avoids needing a mechanical position measurement?
- 3 -> —  (ambiguousdomain) Which inverter paper deals with the DC-link capacitor voltages staying balanced?
- 2 -> 3  (ambiguousdomain) Which paper compares two established drive control strategies head to head?

## Questions the reranker changed

- 1 → 2  (identifier) Which paper uses NSGA-II with TOPSIS for weighting factor selection?
- 1 → 3  (identifier) Which paper concerns LSPMSM stator winding inter-turn faults?
- 2 → 4  (paraphrase) How is torque ripple reduced without using a speed sensor?
- 2 → 6  (topical) Comparing field oriented control against model predictive control
- 1 → 2  (topical) Data-driven predictive control that is computationally efficient
- 1 → 2  (crossdomain) Which paper validates a three-factor measurement scale?
- 1 → 2  (identifier) Which work applies artificial bee colony optimisation?
- 1 → 5  (confusable) Which multilevel converter paper appeared in an IET journal?
- 3 → 1  (deepcontent) Which paper reduces electromagnetic interference by varying the switching frequency?
- 1 → 3  (ambiguousdomain) Which control paper avoids needing a mechanical position measurement?
- — → 3  (ambiguousdomain) Which inverter paper deals with the DC-link capacitor voltages staying balanced?
- 3 → 2  (ambiguousdomain) Which paper compares two established drive control strategies head to head?
