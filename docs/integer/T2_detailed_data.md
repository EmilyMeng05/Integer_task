# T2 Detailed Data

## Setup

| Field | Value |
|---|---|
| Training tasks | `decimal_digit_sum`, `greatest_common_divisor` |
| Architectures | Causal Transformer, Causal MLP |
| Seeds | `17`, `42`, `314159` |
| Training budget | 20,000 steps per model |
| Training lengths | 1–5 decimal digits |
| IID evaluation | 1,000 examples per task per length; 10,000 per model |
| OOD evaluation | 1,000 examples per task per length; 10,000 per model |
| OOD lengths | 6–10 decimal digits |
| Number of models | 6 |

## Training-time validation

| Architecture | Seed | Task | Validation loss | Token accuracy | Sequence accuracy |
|---|---:|---|---:|---:|---:|
| Transformer | 17 | Digit sum | 0.000504 | 100.00% | 100.00% |
| Transformer | 17 | GCD | 0.198868 | 94.58% | 92.50% |
| Transformer | 42 | Digit sum | 0.000580 | 100.00% | 100.00% |
| Transformer | 42 | GCD | 0.202975 | 95.18% | 92.50% |
| Transformer | 314159 | Digit sum | 0.000212 | 100.00% | 100.00% |
| Transformer | 314159 | GCD | 0.234522 | 94.58% | 92.50% |
| Causal MLP | 17 | Digit sum | 0.000347 | 100.00% | 100.00% |
| Causal MLP | 17 | GCD | 0.268087 | 94.58% | 92.50% |
| Causal MLP | 42 | Digit sum | 0.000266 | 100.00% | 100.00% |
| Causal MLP | 42 | GCD | 0.279411 | 94.58% | 92.50% |
| Causal MLP | 314159 | Digit sum | 0.000296 | 100.00% | 100.00% |
| Causal MLP | 314159 | GCD | 0.292365 | 95.18% | 92.50% |

## Full IID results by model and task

| Architecture | Seed | Task | Examples | Exact accuracy | Token accuracy | Well-formed | Teacher-forced loss |
|---|---:|---|---:|---:|---:|---:|---:|
| Transformer | 17 | Digit sum | 5,000 | 99.96% | 99.98% | 100.00% | 0.001106 |
| Transformer | 17 | GCD | 5,000 | 95.06% | 96.83% | 100.00% | 0.132111 |
| Transformer | 42 | Digit sum | 5,000 | 99.98% | 99.99% | 100.00% | 0.000981 |
| Transformer | 42 | GCD | 5,000 | 95.06% | 96.83% | 100.00% | 0.129774 |
| Transformer | 314159 | Digit sum | 5,000 | 100.00% | 100.00% | 100.00% | 0.000394 |
| Transformer | 314159 | GCD | 5,000 | 95.08% | 96.84% | 100.00% | 0.132240 |
| Causal MLP | 17 | Digit sum | 5,000 | 99.96% | 99.98% | 100.00% | 0.000880 |
| Causal MLP | 17 | GCD | 5,000 | 92.88% | 95.74% | 100.00% | 0.171994 |
| Causal MLP | 42 | Digit sum | 5,000 | 100.00% | 100.00% | 100.00% | 0.000415 |
| Causal MLP | 42 | GCD | 5,000 | 93.78% | 96.19% | 100.00% | 0.162335 |
| Causal MLP | 314159 | Digit sum | 5,000 | 99.98% | 99.99% | 100.00% | 0.000638 |
| Causal MLP | 314159 | GCD | 5,000 | 93.72% | 96.16% | 100.00% | 0.162290 |

## Full OOD results by model and task

| Architecture | Seed | Task | Examples | Exact accuracy | Token accuracy | Well-formed | Teacher-forced loss |
|---|---:|---|---:|---:|---:|---:|---:|
| Transformer | 17 | Digit sum | 5,000 | 1.98% | 32.16% | 62.34% | 6.585570 |
| Transformer | 17 | GCD | 5,000 | 35.90% | 46.37% | 59.40% | 3.653800 |
| Transformer | 42 | Digit sum | 5,000 | 1.38% | 50.69% | 99.98% | 7.068010 |
| Transformer | 42 | GCD | 5,000 | 35.02% | 46.01% | 59.82% | 4.392556 |
| Transformer | 314159 | Digit sum | 5,000 | 2.76% | 20.08% | 37.40% | 6.747498 |
| Transformer | 314159 | GCD | 5,000 | 37.74% | 49.00% | 59.72% | 3.283002 |
| Causal MLP | 17 | Digit sum | 5,000 | 19.26% | 38.74% | 58.72% | 10.062516 |
| Causal MLP | 17 | GCD | 5,000 | 16.86% | 18.26% | 20.14% | 7.401312 |
| Causal MLP | 42 | Digit sum | 5,000 | 19.30% | 19.65% | 20.00% | 8.545457 |
| Causal MLP | 42 | GCD | 5,000 | 16.76% | 12.12% | 20.00% | 5.345671 |
| Causal MLP | 314159 | Digit sum | 5,000 | 19.46% | 30.91% | 50.46% | 7.956323 |
| Causal MLP | 314159 | GCD | 5,000 | 22.92% | 35.91% | 60.00% | 5.703467 |

## IID exact accuracy by decimal length

| Architecture | Seed | Task | 1 digit | 2 digits | 3 digits | 4 digits | 5 digits |
|---|---:|---|---:|---:|---:|---:|---:|
| Transformer | 17 | Digit sum | 100.00% | 100.00% | 100.00% | 100.00% | 99.80% |
| Transformer | 17 | GCD | 100.00% | 98.00% | 91.60% | 92.80% | 92.90% |
| Transformer | 42 | Digit sum | 100.00% | 100.00% | 100.00% | 100.00% | 99.90% |
| Transformer | 42 | GCD | 100.00% | 97.80% | 91.50% | 92.80% | 93.20% |
| Transformer | 314159 | Digit sum | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Transformer | 314159 | GCD | 100.00% | 98.70% | 91.30% | 92.60% | 92.80% |
| Causal MLP | 17 | Digit sum | 100.00% | 100.00% | 100.00% | 100.00% | 99.80% |
| Causal MLP | 17 | GCD | 100.00% | 99.50% | 89.00% | 91.20% | 84.70% |
| Causal MLP | 42 | Digit sum | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Causal MLP | 42 | GCD | 100.00% | 99.50% | 91.80% | 92.90% | 84.70% |
| Causal MLP | 314159 | Digit sum | 100.00% | 100.00% | 100.00% | 100.00% | 99.90% |
| Causal MLP | 314159 | GCD | 100.00% | 99.50% | 91.60% | 92.80% | 84.70% |

## OOD exact accuracy by decimal length

| Architecture | Seed | Task | 6 digits | 7 digits | 8 digits | 9 digits | 10 digits |
|---|---:|---|---:|---:|---:|---:|---:|
| Transformer | 17 | Digit sum | 9.90% | 0.00% | 0.00% | 0.00% | 0.00% |
| Transformer | 17 | GCD | 85.70% | 0.00% | 0.00% | 46.40% | 47.40% |
| Transformer | 42 | Digit sum | 6.90% | 0.00% | 0.00% | 0.00% | 0.00% |
| Transformer | 42 | GCD | 86.70% | 0.00% | 0.00% | 46.20% | 42.20% |
| Transformer | 314159 | Digit sum | 13.80% | 0.00% | 0.00% | 0.00% | 0.00% |
| Transformer | 314159 | GCD | 87.20% | 0.00% | 0.00% | 52.00% | 49.50% |
| Causal MLP | 17 | Digit sum | 96.30% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 17 | GCD | 83.80% | 0.40% | 0.10% | 0.00% | 0.00% |
| Causal MLP | 42 | Digit sum | 96.50% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 42 | GCD | 83.80% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 314159 | Digit sum | 97.30% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 314159 | GCD | 83.80% | 0.00% | 0.00% | 15.30% | 15.50% |

## Three-seed summary by task

Values are mean ± sample standard deviation.

| Architecture | Task | IID exact accuracy | OOD exact accuracy |
|---|---|---:|---:|
| Transformer | Digit sum | 99.98% ± 0.02% | 2.04% ± 0.69% |
| Transformer | GCD | 95.07% ± 0.01% | 36.22% ± 1.39% |
| Causal MLP | Digit sum | 99.98% ± 0.02% | 19.34% ± 0.11% |
| Causal MLP | GCD | 93.46% ± 0.50% | 18.85% ± 3.53% |

## Three-seed OOD length summary

| Architecture | Task | 6 digits | 7 digits | 8 digits | 9 digits | 10 digits |
|---|---|---:|---:|---:|---:|---:|
| Transformer | Digit sum | 10.20% ± 3.45% | 0.00% | 0.00% | 0.00% | 0.00% |
| Transformer | GCD | 86.53% ± 0.76% | 0.00% | 0.00% | 48.20% ± 3.29% | 46.37% ± 3.76% |
| Causal MLP | Digit sum | 96.70% ± 0.53% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | GCD | 83.80% ± 0.00% | 0.13% ± 0.23% | 0.03% ± 0.06% | 5.10% ± 8.83% | 5.17% ± 8.95% |

## GCD common-answer baseline

| OOD length | Target `GCD = 1` frequency |
|---:|---:|
| 6 digits | 60.0% |
| 7 digits | 62.7% |
| 8 digits | 61.1% |
| 9 digits | 61.1% |
| 10 digits | 60.3% |

## T1 versus T2 digit-sum comparison

| Architecture | T1 IID | T2 IID | T1 OOD | T2 OOD | OOD change |
|---|---:|---:|---:|---:|---:|
| Transformer | 99.97% | 99.98% | 11.19% | 2.04% | −9.15 points |
| Causal MLP | 100.00% | 99.98% | 19.16% | 19.34% | +0.18 points |

## Metric definitions

| Metric | Meaning |
|---|---|
| Exact accuracy | Entire generated numerical answer is correct |
| Token accuracy | Proportion of generated answer tokens that are correct |
| Well-formed rate | Generated answer follows the required output grammar |
| Teacher-forced loss | Answer-token loss when previous correct answer tokens are supplied |

