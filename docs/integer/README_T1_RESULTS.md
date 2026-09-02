# T1: Decimal Digit Sum

## Task

T1 trains each model on one task: `decimal_digit_sum`.

Example:

```text
Input integer: 137
Target: 1 + 3 + 7 = 11
Model encoding of 137: 01 37
```

The operation is defined over ordinary decimal digits even though the number is encoded using base-100 tokens.

## Experimental setup

| Field | Value |
|---|---|
| Training tasks | `decimal_digit_sum` |
| Architectures | Causal Transformer, Causal MLP |
| Seeds | 17, 42, 314159 |
| Models | 6 |
| Training budget | 20,000 steps per model |
| Training range | 1-5 decimal digits |
| IID evaluation | 1,000 examples per length, 5,000 per model |
| OOD evaluation | 1,000 examples per length, 5,000 per model |

## Training-time validation

| Architecture | Seed | Validation loss | Token accuracy | Sequence accuracy |
|---|---:|---:|---:|---:|
| Transformer | 17 | 0.001990 | 100.00% | 100.00% |
| Transformer | 42 | 0.001068 | 100.00% | 100.00% |
| Transformer | 314159 | 0.001989 | 100.00% | 100.00% |
| Causal MLP | 17 | 0.00000462 | 100.00% | 100.00% |
| Causal MLP | 42 | 0.00000279 | 100.00% | 100.00% |
| Causal MLP | 314159 | 0.00000299 | 100.00% | 100.00% |

## Full IID results

| Architecture | Seed | Exact accuracy | Token accuracy | Well-formed | Teacher-forced loss |
|---|---:|---:|---:|---:|---:|
| Transformer | 17 | 99.98% | 99.99% | 100.00% | 0.002949 |
| Transformer | 42 | 99.94% | 99.97% | 100.00% | 0.001995 |
| Transformer | 314159 | 99.98% | 99.99% | 100.00% | 0.002889 |
| Causal MLP | 17 | 100.00% | 100.00% | 100.00% | 0.00000766 |
| Causal MLP | 42 | 100.00% | 100.00% | 100.00% | 0.00000525 |
| Causal MLP | 314159 | 100.00% | 100.00% | 100.00% | 0.00000745 |

## Full OOD results

These aggregate values combine decimal lengths 6 through 10.

| Architecture | Seed | Exact accuracy | Token accuracy | Well-formed | Teacher-forced loss |
|---|---:|---:|---:|---:|---:|
| Transformer | 17 | 10.14% | 15.58% | 47.34% | 10.294818 |
| Transformer | 42 | 11.24% | 25.28% | 52.38% | 9.889846 |
| Transformer | 314159 | 12.20% | 19.42% | 41.68% | 9.481757 |
| Causal MLP | 17 | 18.52% | 36.58% | 56.62% | 12.148837 |
| Causal MLP | 42 | 19.64% | 10.24% | 31.44% | 12.487578 |
| Causal MLP | 314159 | 19.32% | 24.60% | 39.20% | 13.537917 |

## Exact accuracy by decimal length

### IID

| Architecture | Seed | 1 digit | 2 digits | 3 digits | 4 digits | 5 digits |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 17 | 100.00% | 100.00% | 100.00% | 100.00% | 99.90% |
| Transformer | 42 | 100.00% | 100.00% | 100.00% | 100.00% | 99.70% |
| Transformer | 314159 | 100.00% | 100.00% | 100.00% | 100.00% | 99.90% |
| Causal MLP | 17 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Causal MLP | 42 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Causal MLP | 314159 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

### OOD

| Architecture | Seed | 6 digits | 7 digits | 8 digits | 9 digits | 10 digits |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 17 | 50.70% | 0.00% | 0.00% | 0.00% | 0.00% |
| Transformer | 42 | 56.20% | 0.00% | 0.00% | 0.00% | 0.00% |
| Transformer | 314159 | 61.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 17 | 92.50% | 0.00% | 0.00% | 0.10% | 0.00% |
| Causal MLP | 42 | 98.10% | 0.00% | 0.00% | 0.10% | 0.00% |
| Causal MLP | 314159 | 96.40% | 0.10% | 0.10% | 0.00% | 0.00% |

## Corrected OOD diagnostic

Six digits use the same maximum three-token base-100 width seen in five-digit training examples. Seven-to-ten digits require unseen four- or five-token widths.

| Architecture | Regime | Exact accuracy, mean ± SD | Well-formed, mean ± SD | Most-common-target baseline |
|---|---|---:|---:|---:|
| Transformer | Six-digit magnitude OOD | 55.97% ± 5.15% | 99.97% ± 0.06% | 7.10% |
| Transformer | 7-10-digit token-length OOD | 0.00% ± 0.00% | 33.93% ± 6.69% | 4.78% |
| Causal MLP | Six-digit magnitude OOD | 95.67% ± 2.87% | 100.00% ± 0.00% | 7.10% |
| Causal MLP | 7-10-digit token-length OOD | 0.03% ± 0.01% | 28.03% ± 16.12% | 4.78% |

For six-digit digit sum, always predicting the most common target, `30`, gives 7.10% accuracy. For seven-to-ten digits, always predicting `39` gives 4.78%.

## Finding

Both architectures learned the task nearly perfectly within the training range. They also handled a larger six-digit magnitude when the base-100 token width remained familiar. However, neither architecture demonstrated meaningful token-length generalization: seven-to-ten-digit accuracy was effectively zero and below the constant-answer baseline.
