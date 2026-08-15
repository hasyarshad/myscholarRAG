# Pipeline comparison

3 questions, three configurations, one corpus.

Same index, same embedder, same generator throughout. The only variable is pipeline design, so every difference below is attributable to it rather than to a bigger model or a cleaner corpus.

| Configuration | unsupported claims | correct refusal | over-refusal | citation validity | mean latency |
|---|---|---|---|---|---|
| naive | 50.0% | 66.7% | — | — | 5.8s |
| retrieved | 25.0% | 66.7% | — | — | 5.8s |
| adaptive | — | 100.0% | — | 100.0% | 9.1s |

**unsupported claims** — factual sentences the retrieved passages do not support, judged one at a time. Refusals contain no claims and so contribute nothing, which stops a system improving the number by declining more.

**correct refusal** paired with **over-refusal** — either alone is a bias rather than a measurement. A system that refuses everything scores 100% on the first.

## Correct behaviour by question kind

| Configuration | `unanswerable` |
|---|---|
| naive | 67% |
| retrieved | 67% |
| adaptive | 100% |

## Where each configuration went wrong

### naive — 1 of 3

- **unanswerable** — answered, should have declined: What were his findings on lithium battery degradation?
  > The provided content does **not mention lithium battery degradation** or related findings. The text focuses on **structural rotor fault (SRF) detection**, **machine learning (ML)/deep learni

### retrieved — 1 of 3

- **unanswerable** — answered, should have declined: What were his findings on lithium battery degradation?
  > The provided content does **not mention** any findings related to **lithium battery degradation**. The text focuses on topics such as:

- **Fault diagnosis and predictive control** in power

### adaptive — 0 of 3

Nothing.

## Sample unsupported claims

- **naive** — What were his findings on lithium battery degradation?
  > Key topics covered in the text include: - Use of CNNs, ResNets, and hybrid ML/DL models for rotor imbalance detection. - Feature extraction methods (time-domain, frequenc
- **retrieved** — What were his findings on lithium battery degradation?
  > If the question pertains to a different document or section, additional context would be needed.
