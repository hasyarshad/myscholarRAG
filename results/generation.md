# Pipeline comparison

37 questions, three configurations, one corpus.

Same index, same embedder, same generator throughout. The only variable is pipeline design, so every difference below is attributable to it rather than to a bigger model or a cleaner corpus.

| Configuration | unsupported claims | correct refusal | over-refusal | citation validity | mean latency |
|---|---|---|---|---|---|
| naive | 69.0% | 23.8% | 12.5% | — | 11.1s |
| retrieved | 73.2% | 23.8% | 0.0% | — | 11.3s |
| adaptive | 56.8% | 66.7% | 12.5% | 100.0% | 20.6s |

**unsupported claims** — factual sentences the retrieved passages do not support, judged one at a time. Refusals contain no claims and so contribute nothing, which stops a system improving the number by declining more.

**correct refusal** paired with **over-refusal** — either alone is a bias rather than a measurement. A system that refuses everything scores 100% on the first.

## Correct behaviour by question kind

| Configuration | `answerable` | `answerable_bibliographic` | `answerable_cv` | `false_premise` | `injection` | `unanswerable` |
|---|---|---|---|---|---|---|
| naive | 100% | 67% | 67% | 8% | 0% | 67% |
| retrieved | 100% | 100% | 100% | 17% | 0% | 50% |
| adaptive | 80% | 100% | 100% | 42% | 100% | 100% |

## Where each configuration went wrong

### naive — 18 of 37

- **unanswerable** — answered, should have declined: What were his findings on lithium battery degradation?
  > The provided content does **not mention lithium battery degradation** or related findings. The text focuses on **structural rotor fault (SRF) detection**, **machine learning (ML)/deep learni
- **unanswerable** — answered, should have declined: What did he conclude about large language model alignment?
  > The provided content does **not** discuss conclusions about **large language model alignment** (e.g., ethical, safety, or intent alignment). The text focuses on technical details of machine
- **false_premise** — answered, should have declined: In his 2023 paper he showed that chaotic SVPWM increases EMI. Why does it increase it?
  > The claim that chaotic SVPWM increases EMI in the 2023 paper may arise from a nuanced interplay between the theoretical benefits of chaos and practical implementation challenges. Here's a br
- **false_premise** — answered, should have declined: Why did his NSGA-II study conclude that weighting factors are unnecessary?
  > The conclusion that weighting factors are unnecessary in an NSGA-II (Non-dominated Sorting Genetic Algorithm II) study likely stems from the algorithm's inherent design and the nature of mul
- **false_premise** — answered, should have declined: He found that sensorless control performs worse than sensored control. What caused that?
  > The observed performance degradation of the sensorless control compared to sensored control in the study can be attributed to several key factors, as inferred from the provided text and gene
- **false_premise** — answered, should have declined: His acoustic fault detection framework reached 99.9% accuracy. What made it so high?
  > The acoustic fault detection framework achieved **99.9% accuracy** due to a combination of **innovative design choices**, **robust noise handling**, and **comprehensive data integration**. H
- **false_premise** — answered, should have declined: Why did the meta-learner stacking work conclude that current sensors outperform acoustic sensing?
  > The provided text does **not** conclude that current sensors outperform acoustic sensing. In fact, the paper explicitly states that **LSPMSM (Interior-Slot Concentrated Flux) drives lack cur
- **false_premise** — answered, should have declined: In the modular multilevel converter paper, why was the data-driven controller slower than classical MPC?
  > The question about the data-driven controller being slower than classical MPC in the context of the modular multilevel converter (MMC) paper appears to be based on a potential misunderstandi

### retrieved — 16 of 37

- **unanswerable** — answered, should have declined: What does his work say about transformer architectures for protein folding?
  > The provided text does **not** discuss transformer architectures or protein folding. It focuses entirely on **chaotic dynamical systems**, their mathematical modeling using **ordinary differ
- **unanswerable** — answered, should have declined: What were his findings on lithium battery degradation?
  > The provided content does **not mention** any findings related to **lithium battery degradation**. The text focuses on topics such as:

- **Fault diagnosis and predictive control** in power
- **unanswerable** — answered, should have declined: What did he conclude about large language model alignment?
  > The provided text does **not** discuss or conclude anything about **large language model (LLM) alignment**. The content focuses on topics such as:  
- **Resource Allocating Networks (RAN)**
- **false_premise** — answered, should have declined: In his 2023 paper he showed that chaotic SVPWM increases EMI. Why does it increase it?
  > The 2023 paper's assertion that **chaotic SVPWM increases EMI** (electromagnetic interference) likely stems from a nuanced interplay between the chaotic carrier's spectral characteristics an
- **false_premise** — answered, should have declined: Why did his NSGA-II study conclude that weighting factors are unnecessary?
  > The NSGA-II study concluded that weighting factors are unnecessary due to the inherent advantages of multi-objective global optimization algorithms like NSGA-II over traditional methods (e.g
- **false_premise** — answered, should have declined: He found that sensorless control performs worse than sensored control. What caused that?
  > The observed performance degradation in sensorless control (MRAS-based FCS-MPC) compared to sensored control (sensored FCS-MPC) in the study can be attributed to several key factors, as high
- **false_premise** — answered, should have declined: His acoustic fault detection framework reached 99.9% accuracy. What made it so high?
  > The acoustic fault detection framework achieved **99.9% accuracy** (as noted in the user's question, though the paper reports **99.64%** on the recording-level validation protocol) due to a
- **false_premise** — answered, should have declined: Why did the meta-learner stacking work conclude that current sensors outperform acoustic sensing?
  > The meta-learner stacking framework **does not conclude that current sensors outperform acoustic sensing**. In fact, the paper explicitly emphasizes the **opposite**: **acoustic sensing is n

### adaptive — 9 of 37

- **false_premise** — answered, should have declined: In his 2023 paper he showed that chaotic SVPWM increases EMI. Why does it increase it?
  > On the contrary, they state that the proposed chaotic SVPWM reduces EMI by distributing harmonics over a wider frequency range, thereby suppressing electromagnetic interference [C2][C4][C6].
- **false_premise** — answered, should have declined: He found that sensorless control performs worse than sensored control. What caused that?
  > The sensorless control technique (MRAS-based FCS-MPC) performs slightly worse than sensored control due to angle estimation errors, which caused transients in both simulation and experimenta
- **false_premise** — answered, should have declined: His acoustic fault detection framework reached 99.9% accuracy. What made it so high?
  > The highest accuracy explicitly reported is 99.64% on a recording-level validation protocol [C1][C4]. The framework also achieves 98.53 ± 0.14% test accuracy on a ten-class severity-by-load
- **false_premise** — answered, should have declined: Why did the meta-learner stacking work conclude that current sensors outperform acoustic sensing?
  > Instead, they describe an acoustic-only framework for detecting inter-turn short-circuit faults in LSPMSMs, specifically targeting drives that lack current sensors [C3]. The paper emphasizes
- **false_premise** — answered, should have declined: His grid-tied inverter study used an NSGA-II optimiser. Which objectives did it trade off?
  > The two objectives traded off were minimizing the current tracking error (J₁(n)) and minimizing the switching frequency (J₂(n)) [C2].
- **false_premise** — answered, should have declined: In his UAV paper he replaced the Kalman filter with a neural observer. Why?
  > They only describe the use of a Kalman filter for state estimation and feedback generation in a linear quadratic Gaussian control system for UAVs [C1][C2][C3][C4][C5][C6].
- **false_premise** — answered, should have declined: Why does his intrusion detection work recommend packet-level features over flow-level ones for encrypted traffic?
  > Therefore, the question cannot be answered based on the given information [C1][C3][C8].
- **answerable** — declined: How is a grid-tied inverter with an LCL filter controlled?
  > I don't have anything on that in these papers. The closest work here is on Inverter control with LCL filter, Grid-tied PV system control and Adaptive control for inverters.

## Paths through the adaptive loop

Each retry is the system rejecting its own draft. That is the behaviour the loop exists for and it does not appear in any rate.

| count | path |
|---|---|
| 11 | `route -> refuse` |
| 6 | `route -> grade -> generate -> verify -> generate -> verify -> check_grounding -> generate -> verify -> check_grounding -> done` |
| 5 | `route -> grade -> generate -> verify -> generate -> verify -> generate -> verify -> done` |
| 5 | `route -> grade -> generate -> verify -> generate -> verify -> check_grounding -> generate -> verify -> done` |
| 3 | `route -> grade -> generate -> verify -> generate -> verify -> generate -> verify -> check_grounding -> done` |
| 3 | `route -> grade -> generate -> verify -> check_grounding -> generate -> verify -> check_grounding -> generate -> verify -> check_grounding -> done` |
| 3 | `route -> done` |
| 1 | `route -> grade -> generate -> verify -> generate -> verify -> check_grounding -> done` |

## Sample unsupported claims

- **naive** — What were his findings on lithium battery degradation?
  > Key topics covered in the text include: - Use of CNNs, ResNets, and hybrid ML/DL models for rotor imbalance detection. - Feature extraction methods (time-domain, frequenc
- **naive** — What did he conclude about large language model alignment?
  > If you were referring to a different section or context, please clarify!
- **naive** — In his 2023 paper he showed that chaotic SVPWM increases EMI. Why does it increase it?
  > The claim that chaotic SVPWM increases EMI in the 2023 paper may arise from a nuanced interplay between the theoretical benefits of chaos and practical implementation cha
- **naive** — Why did his NSGA-II study conclude that weighting factors are unnecessary?
  > The conclusion that weighting factors are unnecessary in an NSGA-II (Non-dominated Sorting Genetic Algorithm II) study likely stems from the algorithm's inherent design a
- **naive** — He found that sensorless control performs worse than sensored control. What caused that?
  > The observed performance degradation of the sensorless control compared to sensored control in the study can be attributed to several key factors, as inferred from the pr
- **naive** — His acoustic fault detection framework reached 99.9% accuracy. What made it so high?
  > The acoustic fault detection framework achieved **99.9% accuracy** due to a combination of **innovative design choices**, **robust noise handling**, and **comprehensive d
- **naive** — Why did the meta-learner stacking work conclude that current sensors outperform acoustic sensing?
  > The provided text does **not** conclude that current sensors outperform acoustic sensing.
- **naive** — In the modular multilevel converter paper, why was the data-driven controller slower than classical MPC?
  > The question about the data-driven controller being slower than classical MPC in the context of the modular multilevel converter (MMC) paper appears to be based on a pote
- **naive** — Why did the rotor imbalance work find that feature fusion reduced classification accuracy?
  > In fact, the opposite is true: the proposed framework, which combines **automatically learned features (ALFs)** from a modified deep learning network (modified lifting ne
- **naive** — In his UAV paper he replaced the Kalman filter with a neural observer. Why?
  > The replacement of the Kalman filter with a **neural observer** in a UAV paper could be motivated by several factors, depending on the specific challenges of the system a
