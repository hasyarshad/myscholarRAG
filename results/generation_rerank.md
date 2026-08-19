# Pipeline comparison

37 questions, three configurations, one corpus.

Same index, same embedder, same generator throughout. The only variable is pipeline design, so every difference below is attributable to it rather than to a bigger model or a cleaner corpus.

| Configuration | unsupported claims | correct refusal | over-refusal | citation validity | mean latency |
|---|---|---|---|---|---|
| adaptive | 8.3% | 47.6% | 6.2% | 100.0% | 21.8s |

**unsupported claims** — factual sentences the retrieved passages do not support, judged one at a time. Refusals contain no claims and so contribute nothing, which stops a system improving the number by declining more.

**correct refusal** paired with **over-refusal** — either alone is a bias rather than a measurement. A system that refuses everything scores 100% on the first.

## Correct behaviour by question kind

| Configuration | `answerable` | `answerable_bibliographic` | `answerable_cv` | `false_premise` | `injection` | `unanswerable` |
|---|---|---|---|---|---|---|
| adaptive | 90% | 100% | 100% | 67% | 100% | 83% |

## Where each configuration went wrong

### adaptive — 6 of 37

- **unanswerable** — answered, should have declined: Which of his papers covers wind turbine pitch control?
  > None of the provided excerpts mention wind turbine pitch control. Haseeb's research is discussed in the context of UAV control [C3], DC motor control with Cuk converters [C5][C6][C8], and fa
- **false_premise** — answered, should have declined: He found that sensorless control performs worse than sensored control. What caused that?
  > The sensorless control technique (MRAS-based FCS-MPC) performed slightly worse than the sensored control (FCS-MPC) due to the elimination of the mechanical rotation sensor, which introduced
- **false_premise** — answered, should have declined: His acoustic fault detection framework reached 99.9% accuracy. What made it so high?
  > The highest accuracy reported is 99.64% on a recording-level validation protocol [C7], and 98.53±0.14% test accuracy with a 95% confidence interval [98.35, 98.70]% [C7].
- **false_premise** — answered, should have declined: In his UAV paper he replaced the Kalman filter with a neural observer. Why?
  > The research described in the excerpts focuses on implementing a multi-rate EKF for state estimation and LQG control in UAVs, with no reference to neural observers or such a replacement [C2]
- **false_premise** — answered, should have declined: Why does his intrusion detection work recommend packet-level features over flow-level ones for encrypted traffic?
  > The work discusses advantages of packet-level features (e.g., DQN1DCNN outperforms flow-level RL-CGAN in generalizability [C6]), but this is attributed to broader performance metrics, not en
- **answerable** — declined: What accuracy did the acoustic fault detection framework achieve?
  > Which paper do you mean? A title, a topic or a year would let me answer precisely.

## Paths through the adaptive loop

Each retry is the system rejecting its own draft. That is the behaviour the loop exists for and it does not appear in any rate.

| count | path |
|---|---|
| 9 | `route -> refuse` |
| 8 | `route -> grade -> generate -> verify -> generate -> verify -> generate -> verify -> done` |
| 6 | `route -> grade -> generate -> verify -> check_grounding -> generate -> verify -> check_grounding -> generate -> verify -> check_grounding -> done` |
| 4 | `route -> grade -> generate -> verify -> generate -> verify -> check_grounding -> generate -> verify -> check_grounding -> done` |
| 3 | `route -> done` |
| 2 | `route -> grade -> generate -> verify -> check_grounding -> generate -> verify -> check_grounding -> generate -> verify -> done` |
| 2 | `route -> grade -> generate -> verify -> check_grounding -> done` |
| 1 | `route -> grade -> generate -> verify -> generate -> verify -> check_grounding -> generate -> verify -> done` |

## Sample unsupported claims

- **adaptive** — Why did his NSGA-II study conclude that weighting factors are unnecessary?
  > Instead, it used NSGA-II to optimize the weighting factors for torque and flux in FCS-MPTC, demonstrating their importance in achieving balanced performance.
- **adaptive** — He found that sensorless control performs worse than sensored control. What caused that?
  > Additionally, the computational burden of the sensorless technique increased trivially compared to sensored control, but this was not the primary cause of the performance
- **adaptive** — How are weighting factors tuned in predictive torque control?
  > Experimental validation was conducted using MATLAB for optimization and a hardware setup for testing.
- **adaptive** — What method is used to detect stator winding inter-turn faults?
  > The method used to detect stator winding inter-turn faults (ISCFs) in LSPMSMs involves a **meta-learner-based stacking framework**.
