# Pipeline comparison

37 questions, three configurations, one corpus.

Same index, same embedder, same generator throughout. The only variable is pipeline design, so every difference below is attributable to it rather than to a bigger model or a cleaner corpus.

| Configuration | unsupported claims | claims/question | correct refusal | over-refusal | citation validity | mean latency |
|---|---|---|---|---|---|---|
| adaptive | 7.8% | 1.7 | 80.0% | 0.0% | 100.0% | 11.6s |

**unsupported claims** — factual sentences the retrieved passages do not support, judged one at a time. Refusals contain no claims and so contribute nothing, which stops a system improving the number by declining more.

**correct refusal** paired with **over-refusal** — either alone is a bias rather than a measurement. A system that refuses everything scores 100% on the first.

## Correct behaviour by question kind

| Configuration | `answerable` | `answerable_bibliographic` | `answerable_cv` | `false_premise` | `injection` | `unanswerable` |
|---|---|---|---|---|---|---|
| adaptive | 100% | 100% | 100% | 64% | 100% | 100% |

## Coverage — did the answer use what it was given

Distinct source documents cited, against the number the generator held
after grading. Reported per question kind because breadth is only a
virtue where the corpus has several relevant sources: a question about
one measurement should cite one paper, and a question about the span of
someone's work should not.

| Configuration | kind | n | mean available | mean cited | coverage |
|---|---|---|---|---|---|
| adaptive | answerable | 11 | 1.9 | 1.1 | 57% |
| adaptive | answerable_cv | 3 | 4.3 | 1.3 | 31% |
| adaptive | false_premise | 5 | 2.0 | 1.2 | 60% |

Answers citing one source while holding four or more:

- (answerable_cv) What awards has he received? — cited 1 of 5
- (answerable_cv) What does he teach? — cited 1 of 5

## Where each configuration went wrong

### adaptive — 4 of 37

- **false_premise** — answered, should have declined: In his 2023 paper he showed that chaotic SVPWM increases EMI. Why does it increase it?
  > In the simulation results, the power concentrated in the 10-150 kHz EMI frequency band is less in C-SVPWM compared to conventional SVPWM, confirming improved EMI performance [C2].
- **false_premise** — answered, should have declined: His acoustic fault detection framework reached 99.9% accuracy. What made it so high?
  > The framework's high accuracy stems from its hybrid stacking architecture combining a learnable filter-bank LSTM, an adaptive focal loss 1D-CNN, and a Random Forest, fused via a Gradient Boo
- **false_premise** — answered, should have declined: Why did the meta-learner stacking work conclude that current sensors outperform acoustic sensing?
  > The meta-learner stacking framework is designed to work with acoustic signals due to the lack of current sensors in these systems [C2].
- **false_premise** — answered, should have declined: In the modular multilevel converter paper, why was the data-driven controller slower than classical MPC?
  > The excerpts describe the proposed data-driven model predictive control (MPC) as reducing computational complexity and improving performance metrics like total harmonic distortion (THD) and

## Paths through the adaptive loop

Each retry is the system rejecting its own draft. That is the behaviour the loop exists for and it does not appear in any rate.

| count | path |
|---|---|
| 15 | `route -> screen -> generate -> verify -> check_grounding -> done` |
| 14 | `route -> screen -> generate -> verify -> check_grounding -> drop_ungrounded -> done` |
| 5 | `route -> refuse` |
| 3 | `route -> done` |

## Sample unsupported claims

- **adaptive** — How are weighting factors tuned in predictive torque control?
  > This approach reduces torque and flux ripple while improving switching frequency losses compared to conventional methods.
- **adaptive** — What method is used to detect stator winding inter-turn faults?
  > The method used to detect stator winding inter-turn faults is a meta-learner-based stacking framework.
- **adaptive** — How is chaos analysed in the jerk equation work?
  > The Jacobian matrix of the system is calculated, and its eigenvalues are determined by solving the characteristic equation $\lambda^3 + \alpha \lambda^2 + \beta \lambda -
- **adaptive** — What awards has he received?
  > He held the Alberta Innovates Graduate Student Scholarship from 2023 to 2025.
