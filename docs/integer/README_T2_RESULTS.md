# T2: Decimal Digit Sum and Greatest Common Divisor

## Tasks

T2 trains each model jointly on:

1. `decimal_digit_sum`
2. `greatest_common_divisor`

Examples:

```text
digit_sum(137) = 11
gcd(84, 30) = 6
```

T2 retains the T1 task and adds GCD, allowing us to test whether increased task diversity changes either task's generalization.

## Experimental setup

| Field | Value |
|---|---|
| Architectures | Causal Transformer, Causal MLP |
| Seeds | 17, 42, 314159 |
| Models | 6 |
| Training budget | 20,000 total steps per model |
| Training range | 1-5 decimal digits |
| IID evaluation | 5,000 examples per task per model |
| OOD evaluation | 5,000 examples per task per model |

Loss and accuracy are reported separately for digit sum and GCD. An overall average is insufficient because it can hide task-specific failure.

## Training-time validation

| Architecture | Seed | Task | Loss | Token accuracy | Sequence accuracy |
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

## Full IID results

| Architecture | Seed | Task | Exact | Token | Well-formed | TF loss |
|---|---:|---|---:|---:|---:|---:|
| Transformer | 17 | Digit sum | 99.96% | 99.98% | 100.00% | 0.001106 |
| Transformer | 17 | GCD | 95.06% | 96.83% | 100.00% | 0.132111 |
| Transformer | 42 | Digit sum | 99.98% | 99.99% | 100.00% | 0.000981 |
| Transformer | 42 | GCD | 95.06% | 96.83% | 100.00% | 0.129774 |
| Transformer | 314159 | Digit sum | 100.00% | 100.00% | 100.00% | 0.000394 |
| Transformer | 314159 | GCD | 95.08% | 96.84% | 100.00% | 0.132240 |
| Causal MLP | 17 | Digit sum | 99.96% | 99.98% | 100.00% | 0.000880 |
| Causal MLP | 17 | GCD | 92.88% | 95.74% | 100.00% | 0.171994 |
| Causal MLP | 42 | Digit sum | 100.00% | 100.00% | 100.00% | 0.000415 |
| Causal MLP | 42 | GCD | 93.78% | 96.19% | 100.00% | 0.162335 |
| Causal MLP | 314159 | Digit sum | 99.98% | 99.99% | 100.00% | 0.000638 |
| Causal MLP | 314159 | GCD | 93.72% | 96.16% | 100.00% | 0.162290 |

## Full OOD results

These aggregate values combine lengths 6 through 10.

| Architecture | Seed | Task | Exact | Token | Well-formed | TF loss |
|---|---:|---|---:|---:|---:|---:|
| Transformer | 17 | Digit sum | 1.98% | 32.16% | 62.34% | 6.585570 |
| Transformer | 17 | GCD | 35.90% | 46.37% | 59.40% | 3.653800 |
| Transformer | 42 | Digit sum | 1.38% | 50.69% | 99.98% | 7.068010 |
| Transformer | 42 | GCD | 35.02% | 46.01% | 59.82% | 4.392556 |
| Transformer | 314159 | Digit sum | 2.76% | 20.08% | 37.40% | 6.747498 |
| Transformer | 314159 | GCD | 37.74% | 49.00% | 59.72% | 3.283002 |
| Causal MLP | 17 | Digit sum | 19.26% | 38.74% | 58.72% | 10.062516 |
| Causal MLP | 17 | GCD | 16.86% | 18.26% | 20.14% | 7.401312 |
| Causal MLP | 42 | Digit sum | 19.30% | 19.65% | 20.00% | 8.545457 |
| Causal MLP | 42 | GCD | 16.76% | 12.12% | 20.00% | 5.345671 |
| Causal MLP | 314159 | Digit sum | 19.46% | 30.91% | 50.46% | 7.956323 |
| Causal MLP | 314159 | GCD | 22.92% | 35.91% | 60.00% | 5.703467 |

## Exact accuracy by OOD length

| Architecture | Seed | Task | 6d | 7d | 8d | 9d | 10d |
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

The isolated 9-10-digit GCD accuracies are misleading without a baseline because `GCD = 1` is extremely common.

## Corrected OOD diagnostic

| Architecture | Task | Six-digit exact | 7-10-digit exact | 7-10-digit well-formed | 7-10-digit baseline |
|---|---|---:|---:|---:|---:|
| Transformer | Digit sum | 10.20% ± 3.46% | 0.00% | 58.22% ± 39.38% | 4.78% |
| Causal MLP | Digit sum | 96.70% ± 0.53% | 0.00% | 28.83% ± 25.49% | 4.78% |
| Transformer | GCD | 86.53% ± 0.76% | 23.64% ± 1.65% | 49.56% ± 0.27% | 61.30% |
| Causal MLP | GCD | 83.80% ± 0.00% | 2.61% ± 4.41% | 16.73% ± 28.82% | 61.30% |

For seven-to-ten-digit GCD examples, `1` is the correct answer in 61.30% of cases. A constant model that always outputs `1` therefore achieves 61.30%, substantially above both trained architectures.

### GCD target strata

| Architecture | Scope | Accuracy when GCD = 1 | Accuracy when GCD > 1 |
|---|---|---:|---:|
| Transformer | All 6-10-digit OOD | 48.63% ± 2.09% | 16.77% ± 0.31% |
| Transformer | True 7-10-digit token OOD | 37.33% ± 2.48% | 1.96% ± 0.33% |
| Causal MLP | All 6-10-digit OOD | 21.88% ± 3.70% | 14.10% ± 3.26% |
| Causal MLP | True 7-10-digit token OOD | 2.76% ± 4.60% | 2.37% ± 4.10% |

The Transformer's apparent GCD performance is driven heavily by common `GCD = 1` targets. Its 1.96% accuracy on non-one targets in the true token-length regime provides little evidence of robust GCD computation.

## T1 versus T2 digit sum

| Architecture | T1 aggregate OOD | T2 aggregate OOD | Change |
|---|---:|---:|---:|
| Transformer | 11.19% | 2.04% | -9.15 percentage points |
| Causal MLP | 19.16% | 19.34% | +0.18 percentage points |

Adding GCD sharply reduced Transformer digit-sum OOD accuracy but left the MLP's aggregate digit-sum OOD accuracy approximately unchanged. This is evidence of architecture-dependent task interaction, not evidence that adding a second task uniformly improves generalization.

## Finding

T2 achieved strong IID results on both tasks. Both architectures also performed well on six-digit inputs with a familiar base-100 token width. Neither showed reliable seven-to-ten-digit token-length generalization. The GCD baseline and target-stratified results demonstrate why raw accuracy alone can substantially overstate arithmetic generalization.
