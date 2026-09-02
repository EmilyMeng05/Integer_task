# T4: Four-Task Integer Multitask Results

## Tasks

T4 trains each model jointly on four nested tasks:

1. `decimal_digit_sum`
2. `greatest_common_divisor`
3. `multiplication`
4. `greater_than`

T4 retains both T2 tasks and adds multiplication and greater-than comparison. The purpose is to test how a fixed model and training budget behave as task diversity increases.

## Experimental setup

| Field | Value |
|---|---|
| Architectures | Causal Transformer and Causal MLP |
| Seeds | 17, 42, 314159 |
| Models | 6 |
| Training budget | 20,000 total optimizer steps per model |
| Approximate examples seen per task | 320,000 |
| Training range | 1-5 decimal digits |
| IID evaluation | 1,000 examples per task per length; 20,000 per model |
| OOD evaluation | 1,000 examples per task per length; 20,000 per model |
| Result files | Format version 2; all sample-count checks passed |

The task sampler was balanced by examples. Multiplication nevertheless contributed many more supervised answer tokens because its outputs are longer.

## Training-time validation summary

Values are means across three seeds. The exact-sequence column includes the sample standard deviation when nonzero.

| Architecture | Task | Validation loss | Token accuracy | Exact-sequence accuracy |
|---|---|---:|---:|---:|
| Transformer | Digit sum | 0.0023 | 100.00% | 100.00% |
| Transformer | Greater than | 0.0004 | 100.00% | 100.00% |
| Transformer | GCD | 0.2517 | 94.58% | 92.08% ± 0.72% |
| Transformer | Multiplication | 1.4688 | 60.94% | 25.00% ± 1.25% |
| Causal MLP | Digit sum | 0.0030 | 100.00% | 100.00% |
| Causal MLP | Greater than | 0.0064 | 100.00% | 100.00% |
| Causal MLP | GCD | 0.3546 | 92.97% | 89.17% ± 0.72% |
| Causal MLP | Multiplication | 1.5915 | 58.04% | 24.17% ± 1.44% |

Multiplication was consistently difficult across architectures and seeds. Its weak performance is not attributable to a single unlucky initialization.

## Complete per-seed exact results

`Six-digit` is magnitude OOD with a familiar three-token base-100 width. `7-10 digit` is true token-length OOD.

| Architecture | Seed | Task | IID exact | All 6-10 OOD | Six-digit exact | 7-10-digit exact | 7-10-digit well-formed |
|---|---:|---|---:|---:|---:|---:|---:|
| Transformer | 17 | Digit sum | 99.96% | 5.06% | 25.30% | 0.00% | 76.72% |
| Transformer | 17 | Greater than | 99.58% | 38.38% | 99.20% | 23.18% | 52.48% |
| Transformer | 17 | GCD | 94.68% | 53.02% | 91.50% | 43.40% | 90.40% |
| Transformer | 17 | Multiplication | 25.28% | 0.00% | 0.00% | 0.00% | 3.75% |
| Transformer | 42 | Digit sum | 99.90% | 1.82% | 9.10% | 0.00% | 22.88% |
| Transformer | 42 | Greater than | 99.62% | 29.14% | 99.20% | 11.62% | 49.05% |
| Transformer | 42 | GCD | 94.36% | 43.96% | 79.90% | 34.98% | 78.42% |
| Transformer | 42 | Multiplication | 25.94% | 0.00% | 0.00% | 0.00% | 31.13% |
| Transformer | 314159 | Digit sum | 99.98% | 11.66% | 58.20% | 0.03% | 8.15% |
| Transformer | 314159 | Greater than | 99.54% | 19.86% | 99.30% | 0.00% | 0.00% |
| Transformer | 314159 | GCD | 93.14% | 21.14% | 84.10% | 5.40% | 11.43% |
| Transformer | 314159 | Multiplication | 26.80% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 17 | Digit sum | 99.84% | 3.32% | 16.10% | 0.12% | 15.40% |
| Causal MLP | 17 | Greater than | 98.82% | 17.66% | 86.70% | 0.40% | 11.12% |
| Causal MLP | 17 | GCD | 90.04% | 16.90% | 67.70% | 4.20% | 49.35% |
| Causal MLP | 17 | Multiplication | 23.96% | 0.00% | 0.00% | 0.00% | 7.70% |
| Causal MLP | 42 | Digit sum | 99.96% | 15.90% | 79.50% | 0.00% | 2.08% |
| Causal MLP | 42 | Greater than | 98.90% | 18.26% | 90.30% | 0.25% | 2.80% |
| Causal MLP | 42 | GCD | 90.10% | 14.70% | 73.50% | 0.00% | 0.00% |
| Causal MLP | 42 | Multiplication | 24.78% | 0.00% | 0.00% | 0.00% | 5.25% |
| Causal MLP | 314159 | Digit sum | 99.88% | 4.42% | 22.10% | 0.00% | 0.00% |
| Causal MLP | 314159 | Greater than | 98.96% | 17.98% | 89.90% | 0.00% | 0.45% |
| Causal MLP | 314159 | GCD | 90.18% | 14.10% | 67.10% | 0.85% | 11.00% |
| Causal MLP | 314159 | Multiplication | 24.44% | 0.00% | 0.00% | 0.00% | 25.95% |

## Three-seed aggregate IID results

Values are mean ± sample standard deviation.

| Architecture | Task | Exact accuracy | Token accuracy | Well-formed | Teacher-forced loss |
|---|---|---:|---:|---:|---:|
| Transformer | Digit sum | 99.95% ± 0.04% | 99.97% ± 0.02% | 100.00% | 0.0025 ± 0.0017 |
| Transformer | Greater than | 99.58% ± 0.04% | 99.79% ± 0.02% | 100.00% | 0.0052 ± 0.0000 |
| Transformer | GCD | 94.06% ± 0.81% | 96.33% ± 0.40% | 100.00% | 0.1502 ± 0.0070 |
| Transformer | Multiplication | 26.01% ± 0.76% | 62.28% ± 0.95% | 100.00% | 1.4539 ± 0.0211 |
| Causal MLP | Digit sum | 99.89% ± 0.06% | 99.95% ± 0.03% | 100.00% | 0.0050 ± 0.0019 |
| Causal MLP | Greater than | 98.89% ± 0.07% | 99.45% ± 0.04% | 100.00% | 0.0149 ± 0.0015 |
| Causal MLP | GCD | 90.11% ± 0.07% | 94.36% ± 0.03% | 100.00% | 0.2259 ± 0.0020 |
| Causal MLP | Multiplication | 24.39% ± 0.41% | 59.00% ± 0.27% | 100.00% | 1.5743 ± 0.0122 |

## Three-seed OOD diagnostic

| Architecture | Task | Six-digit exact | 7-10-digit exact | 7-10-digit well-formed | 7-10 baseline |
|---|---|---:|---:|---:|---:|
| Transformer | Digit sum | 30.87% ± 25.02% | 0.01% ± 0.01% | 35.92% ± 36.10% | 4.78% |
| Transformer | Greater than | 99.23% ± 0.06% | 11.60% ± 11.59% | 33.84% ± 29.36% | 50.00% |
| Transformer | GCD | 85.17% ± 5.87% | 27.93% ± 19.96% | 60.08% ± 42.56% | 61.30% |
| Transformer | Multiplication | 0.00% | 0.00% | 11.63% ± 16.99% | 0.03% |
| Causal MLP | Digit sum | 39.23% ± 35.00% | 0.04% ± 0.07% | 5.83% ± 8.36% | 4.78% |
| Causal MLP | Greater than | 88.97% ± 1.97% | 0.22% ± 0.20% | 4.79% ± 5.61% | 50.00% |
| Causal MLP | GCD | 69.43% ± 3.54% | 1.68% ± 2.22% | 20.12% ± 25.91% | 61.30% |
| Causal MLP | Multiplication | 0.00% | 0.00% | 12.97% ± 11.31% | 0.03% |

No T4 task exceeds its most-common-target baseline reliably in the true seven-to-ten-digit token-length regime.

## GCD shortcut diagnostic

| Architecture | Scope | Accuracy when GCD = 1 | Accuracy when GCD > 1 |
|---|---|---:|---:|
| Transformer | All 6-10-digit OOD | 53.17% ± 23.80% | 17.76% ± 4.96% |
| Transformer | True 7-10-digit OOD | 42.46% ± 30.02% | 4.91% ± 4.02% |
| Causal MLP | All 6-10-digit OOD | 18.65% ± 1.73% | 9.87% ± 1.13% |
| Causal MLP | True 7-10-digit OOD | 1.90% ± 2.47% | 1.34% ± 1.83% |

The Transformer remains much more accurate when the target is the common answer `1`. True-length accuracy for non-one GCD targets is only 4.91% on average.

## T1, T2, and T4 comparison

### Digit sum

| Architecture | Condition | Six-digit exact | 7-10-digit exact |
|---|---|---:|---:|
| Transformer | T1 | 55.97% | 0.00% |
| Transformer | T2 | 10.20% | 0.00% |
| Transformer | T4 | 30.87% | 0.01% |
| Causal MLP | T1 | 95.67% | 0.03% |
| Causal MLP | T2 | 96.70% | 0.00% |
| Causal MLP | T4 | 39.23% | 0.04% |

### GCD

| Architecture | Condition | Six-digit exact | 7-10-digit exact | 7-10 baseline |
|---|---|---:|---:|---:|
| Transformer | T2 | 86.53% | 23.64% | 61.30% |
| Transformer | T4 | 85.17% | 27.93% | 61.30% |
| Causal MLP | T2 | 83.80% | 2.61% | 61.30% |
| Causal MLP | T4 | 69.43% | 1.68% | 61.30% |

The effect of adding tasks is not monotonic. T4 does not consistently outperform T1 or T2, and the large seed variance means single-checkpoint comparisons are unreliable.

## Main findings

1. **IID learning is task dependent.** Digit sum and greater-than comparison are nearly perfect, GCD is strong, and multiplication remains underfit at roughly 24-26% exact accuracy.
2. **Six-digit success is not true length generalization.** Several tasks perform well at six digits because six-digit inputs retain a familiar base-100 token width.
3. **True token-length generalization remains absent.** Every T4 task is below its constant-answer baseline on seven-to-ten-digit inputs.
4. **GCD accuracy is inflated by the common answer `1`.** Non-one GCD targets expose much weaker computation.
5. **Seed variability matters.** Transformer true-length GCD ranges from 5.40% to 43.40%, supporting the use of three seeds and error bars.
6. **More tasks do not automatically improve generalization.** The relationship between task diversity, architecture, and transfer must be measured rather than assumed.

## Next step

Train the six T8 base models using all eight tasks, then evaluate them with the same diagnostic protocol. After T8, compare T1/T2/T4/T8 and begin few-shot adaptation on the four held-out tasks.
