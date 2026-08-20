# Retrieval evaluation

37 questions, document-level ranking over the
expanded context blocks the generator would receive.

## Overall

| Configuration | hit@1 | hit@3 | hit@5 | MRR | context precision | found | bypassed |
|---|---|---|---|---|---|---|---|
| dense | 0.78 | 0.89 | 0.89 | 0.838 | 0.817 | 33/37 | 0 |
| hybrid | 0.81 | 0.92 | 0.92 | 0.860 | 0.705 | 34/37 | 0 |
| routed | 0.81 | 0.92 | 0.92 | 0.860 | 0.694 | 34/37 | 0 |

## hit@3 by question kind

| Configuration | ambiguousdomain | confusable | crossdomain | deepcontent | identifier | paraphrase | topical |
|---|---|---|---|---|---|---|---|
| dense | 0.50 | 1.00 | 1.00 | 0.83 | 1.00 | 0.88 | 1.00 |
| hybrid | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 |
| routed | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 |

## Questions no configuration answered

- How is unbalanced rotation in rotating machinery detected automatically?
- Which inverter paper deals with the DC-link capacitor voltages staying balanced?
- Which paper compares two established drive control strategies head to head?

## Questions routing changed

