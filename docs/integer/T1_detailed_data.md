# T1 Detailed Data

## Setup

| Field | Value |
|---|---|
| Training tasks | `decimal_digit_sum` |
| Architectures | Causal Transformer, Causal MLP |
| Seeds | `17`, `42`, `314159` |
| Training budget | 20,000 steps per model |
| Training lengths | 1–5 decimal digits |
| IID evaluation | 1,000 examples per length; 5,000 per model |
| OOD evaluation | 1,000 examples per length; 5,000 per model |
| OOD lengths | 6–10 decimal digits |
| Number of models | 6 |

## Training-time validation

| Architecture | Seed | Task | Validation loss | Token accuracy | Sequence accuracy |
|---|---:|---|---:|---:|---:|
| Transformer | 17 | Digit sum | 0.001990 | 100.00% | 100.00% |
| Transformer | 42 | Digit sum | 0.001068 | 100.00% | 100.00% |
| Transformer | 314159 | Digit sum | 0.001989 | 100.00% | 100.00% |
| Causal MLP | 17 | Digit sum | 0.00000462 | 100.00% | 100.00% |
| Causal MLP | 42 | Digit sum | 0.00000279 | 100.00% | 100.00% |
| Causal MLP | 314159 | Digit sum | 0.00000299 | 100.00% | 100.00% |

## Full IID results

| Architecture | Seed | Examples | Exact accuracy | Token accuracy | Well-formed rate | Teacher-forced loss |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 17 | 5,000 | 99.98% | 99.99% | 100.00% | 0.002949 |
| Transformer | 42 | 5,000 | 99.94% | 99.97% | 100.00% | 0.001995 |
| Transformer | 314159 | 5,000 | 99.98% | 99.99% | 100.00% | 0.002889 |
| Causal MLP | 17 | 5,000 | 100.00% | 100.00% | 100.00% | 0.00000766 |
| Causal MLP | 42 | 5,000 | 100.00% | 100.00% | 100.00% | 0.00000525 |
| Causal MLP | 314159 | 5,000 | 100.00% | 100.00% | 100.00% | 0.00000745 |

## Full OOD results

| Architecture | Seed | Examples | Exact accuracy | Token accuracy | Well-formed rate | Teacher-forced loss |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 17 | 5,000 | 10.14% | 15.58% | 47.34% | 10.294818 |
| Transformer | 42 | 5,000 | 11.24% | 25.28% | 52.38% | 9.889846 |
| Transformer | 314159 | 5,000 | 12.20% | 19.42% | 41.68% | 9.481757 |
| Causal MLP | 17 | 5,000 | 18.52% | 36.58% | 56.62% | 12.148837 |
| Causal MLP | 42 | 5,000 | 19.64% | 10.24% | 31.44% | 12.487578 |
| Causal MLP | 314159 | 5,000 | 19.32% | 24.60% | 39.20% | 13.537917 |

## IID exact accuracy by decimal length

| Architecture | Seed | 1 digit | 2 digits | 3 digits | 4 digits | 5 digits |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 17 | 100.00% | 100.00% | 100.00% | 100.00% | 99.90% |
| Transformer | 42 | 100.00% | 100.00% | 100.00% | 100.00% | 99.70% |
| Transformer | 314159 | 100.00% | 100.00% | 100.00% | 100.00% | 99.90% |
| Causal MLP | 17 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Causal MLP | 42 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Causal MLP | 314159 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

## OOD exact accuracy by decimal length

| Architecture | Seed | 6 digits | 7 digits | 8 digits | 9 digits | 10 digits |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 17 | 50.70% | 0.00% | 0.00% | 0.00% | 0.00% |
| Transformer | 42 | 56.20% | 0.00% | 0.00% | 0.00% | 0.00% |
| Transformer | 314159 | 61.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 17 | 92.50% | 0.00% | 0.00% | 0.10% | 0.00% |
| Causal MLP | 42 | 98.10% | 0.00% | 0.00% | 0.10% | 0.00% |
| Causal MLP | 314159 | 96.40% | 0.10% | 0.10% | 0.00% | 0.00% |

## Three-seed summary

Values are mean ± sample standard deviation.

| Architecture | IID exact accuracy | OOD exact accuracy | OOD token accuracy | OOD well-formed rate |
|---|---:|---:|---:|---:|
| Transformer | 99.97% ± 0.02% | 11.19% ± 1.03% | 20.09% ± 4.88% | 47.13% ± 5.35% |
| Causal MLP | 100.00% ± 0.00% | 19.16% ± 0.58% | 23.81% ± 13.18% | 42.42% ± 12.90% |

## Three-seed OOD length summary

| Architecture | 6 digits | 7 digits | 8 digits | 9 digits | 10 digits |
|---|---:|---:|---:|---:|---:|
| Transformer | 55.97% ± 5.15% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 95.67% ± 2.87% | 0.03% ± 0.06% | 0.03% ± 0.06% | 0.07% ± 0.06% | 0.00% |

## Metric definitions

| Metric | Meaning |
|---|---|
| Exact accuracy | Entire generated numerical answer is correct |
| Token accuracy | Proportion of generated answer tokens that are correct |
| Well-formed rate | Generated answer follows the required output grammar |
| Teacher-forced loss | Answer-token loss when previous correct answer tokens are supplied |

