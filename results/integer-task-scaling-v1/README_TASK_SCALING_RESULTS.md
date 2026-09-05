# Integer Task-Scaling Results: T1 to T16

## Purpose

This report follows each shared task after it enters the nested T1, T2, T4, T8, and T16 curriculum. Values are autoregressive exact-sequence accuracy, reported as mean ± sample standard deviation across seeds 17, 42, and 314159.

## Evaluation protocol audit

- Exact paired examples across all stages: **no**
- Interpretation: **distribution-level comparison only; evaluation data identities differ**

> **Important:** The legacy T1–T8 results and T16 results record different evaluation-data identities. The tables are useful as a distribution-level comparison because they use the same decimal-length regimes and sample counts, but they are not a strictly paired comparison on identical examples. Do not attribute differences solely to task count without stating this limitation.

T16 has a smaller amount of data [800,000 vs 160,000] seen for each task. 

## IID exact accuracy

### Transformer

| Task | T1 | T2 | T4 | T8 | T16 |
|---|---:|---:|---:|---:|---:|
| decimal_digit_sum | 99.97 ± 0.02 | 99.98 ± 0.02 | 99.95 ± 0.04 | 99.77 ± 0.06 | 87.03 ± 3.95 |
| greatest_common_divisor | — | 95.07 ± 0.01 | 94.06 ± 0.81 | 89.21 ± 0.10 | 88.95 ± 0.11 |
| multiplication | — | — | 26.01 ± 0.76 | 23.11 ± 0.18 | 21.34 ± 0.12 |
| greater_than | — | — | 99.58 ± 0.04 | 99.37 ± 0.27 | 98.90 ± 0.31 |
| integer_list_sum | — | — | — | 17.91 ± 3.41 | 12.48 ± 0.70 |
| modulo | — | — | — | 67.85 ± 1.50 | 64.72 ± 1.33 |
| addition | — | — | — | 81.47 ± 6.68 | 36.60 ± 4.90 |
| successor | — | — | — | 99.71 ± 0.08 | 99.41 ± 0.09 |

### Mlp

| Task | T1 | T2 | T4 | T8 | T16 |
|---|---:|---:|---:|---:|---:|
| decimal_digit_sum | 100.00 ± 0.00 | 99.98 ± 0.02 | 99.89 ± 0.06 | 94.49 ± 2.98 | 83.37 ± 0.98 |
| greatest_common_divisor | — | 93.46 ± 0.50 | 90.11 ± 0.07 | 88.37 ± 0.56 | 87.27 ± 0.47 |
| multiplication | — | — | 24.39 ± 0.41 | 21.39 ± 0.36 | 20.13 ± 0.04 |
| greater_than | — | — | 98.89 ± 0.07 | 97.39 ± 0.42 | 94.63 ± 3.06 |
| integer_list_sum | — | — | — | 10.74 ± 1.00 | 5.65 ± 0.94 |
| modulo | — | — | — | 64.39 ± 0.99 | 44.70 ± 11.24 |
| addition | — | — | — | 38.80 ± 1.32 | 25.37 ± 0.70 |
| successor | — | — | — | 99.17 ± 0.03 | 96.54 ± 1.05 |

## Six-digit magnitude-OOD exact accuracy

### Transformer

| Task | T1 | T2 | T4 | T8 | T16 |
|---|---:|---:|---:|---:|---:|
| decimal_digit_sum | 55.97 ± 5.15 | 10.20 ± 3.46 | 30.87 ± 25.02 | 76.73 ± 3.48 | 12.87 ± 4.80 |
| greatest_common_divisor | — | 86.53 ± 0.76 | 85.17 ± 5.87 | 83.30 ± 0.61 | 83.87 ± 0.32 |
| multiplication | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| greater_than | — | — | 99.23 ± 0.06 | 98.53 ± 0.81 | 95.80 ± 1.74 |
| integer_list_sum | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| modulo | — | — | — | 1.90 ± 1.95 | 0.40 ± 0.61 |
| addition | — | — | — | 3.47 ± 3.26 | 0.07 ± 0.06 |
| successor | — | — | — | 90.13 ± 4.99 | 71.07 ± 15.90 |

### Mlp

| Task | T1 | T2 | T4 | T8 | T16 |
|---|---:|---:|---:|---:|---:|
| decimal_digit_sum | 95.67 ± 2.87 | 96.70 ± 0.53 | 39.23 ± 35.00 | 33.53 ± 22.51 | 12.20 ± 2.26 |
| greatest_common_divisor | — | 83.80 ± 0.00 | 69.43 ± 3.53 | 50.93 ± 5.01 | 53.87 ± 5.44 |
| multiplication | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| greater_than | — | — | 88.97 ± 1.97 | 62.07 ± 8.32 | 64.10 ± 23.21 |
| integer_list_sum | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| modulo | — | — | — | 5.40 ± 6.42 | 4.63 ± 8.03 |
| addition | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| successor | — | — | — | 5.33 ± 4.02 | 1.37 ± 2.03 |

## Seven-to-ten-digit token-length-OOD exact accuracy

### Transformer

| Task | T1 | T2 | T4 | T8 | T16 |
|---|---:|---:|---:|---:|---:|
| decimal_digit_sum | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.01 ± 0.01 |
| greatest_common_divisor | — | 23.64 ± 1.65 | 27.93 ± 19.96 | 1.38 ± 1.19 | 13.18 ± 7.42 |
| multiplication | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| greater_than | — | — | 11.60 ± 11.59 | 11.46 ± 7.41 | 37.08 ± 13.80 |
| integer_list_sum | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| modulo | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| addition | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| successor | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |

### Mlp

| Task | T1 | T2 | T4 | T8 | T16 |
|---|---:|---:|---:|---:|---:|
| decimal_digit_sum | 0.03 ± 0.01 | 0.00 ± 0.00 | 0.04 ± 0.07 | 0.00 ± 0.00 | 0.03 ± 0.06 |
| greatest_common_divisor | — | 2.61 ± 4.41 | 1.68 ± 2.22 | 0.00 ± 0.00 | 0.02 ± 0.03 |
| multiplication | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| greater_than | — | — | 0.22 ± 0.20 | 3.44 ± 4.75 | 23.36 ± 1.55 |
| integer_list_sum | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| modulo | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| addition | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |
| successor | — | — | — | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Change from task introduction to T16

Positive values indicate higher accuracy at T16; negative values indicate lower accuracy. These are percentage-point changes in the three-seed means.

| Architecture | Task | Starting stage | IID Δ | Six-digit Δ | 7–10-digit Δ |
|---|---|---:|---:|---:|---:|
| transformer | decimal_digit_sum | T1 | -12.93 | -43.10 | +0.01 |
| transformer | greatest_common_divisor | T2 | -6.12 | -2.67 | -10.46 |
| transformer | multiplication | T4 | -4.67 | +0.00 | +0.00 |
| transformer | greater_than | T4 | -0.68 | -3.43 | +25.48 |
| transformer | integer_list_sum | T8 | -5.43 | +0.00 | +0.00 |
| transformer | modulo | T8 | -3.13 | -1.50 | +0.00 |
| transformer | addition | T8 | -44.87 | -3.40 | +0.00 |
| transformer | successor | T8 | -0.30 | -19.07 | +0.00 |
| mlp | decimal_digit_sum | T1 | -16.63 | -83.47 | +0.00 |
| mlp | greatest_common_divisor | T2 | -6.19 | -29.93 | -2.59 |
| mlp | multiplication | T4 | -4.26 | +0.00 | +0.00 |
| mlp | greater_than | T4 | -4.27 | -24.87 | +23.14 |
| mlp | integer_list_sum | T8 | -5.09 | +0.00 | +0.00 |
| mlp | modulo | T8 | -19.69 | -0.77 | +0.00 |
| mlp | addition | T8 | -13.43 | +0.00 | +0.00 |
| mlp | successor | T8 | -2.63 | -3.97 | +0.00 |

## Interpretation rules

- Compare a task only across stages in which that task was trained.
- Do not compare changing macro averages across different task compositions.
- Compare GCD results with the most-common-target baseline because `1` is frequent.
- Treat six-digit magnitude OOD separately from genuine 7–10-digit token-length OOD.
- Under the fixed 20,000-step budget, later stages provide fewer examples per task; task diversity and per-task exposure are therefore not separately identified.
