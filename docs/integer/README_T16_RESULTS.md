# T16 Integer Multitask Results

## Status

The T16 experiment is complete. Six models were trained and evaluated:

- Architectures: Causal Transformer and Causal MLP
- Seeds: `17`, `42`, and `314159`
- Training tasks: 16
- Training budget: 20,000 optimization steps per model
- Encoding: base 100
- Full evaluation size: 1,000 examples per task per decimal length
- Total evaluated: 900,726 examples across all models and splits

Each model was evaluated on 75,101 IID examples and 75,020 OOD examples. Factorial is treated separately because its finite training domain is `0–100`: all 101 values are used for its seen-domain evaluation, and `101–120` are used for value extrapolation.

## Tasks

1. Decimal digit sum
2. Greatest common divisor
3. Multiplication
4. Greater-than comparison
5. Integer-list sum
6. Modulo
7. Addition
8. Successor
9. Subtraction
10. Integer division
11. Number of decimal digits
12. Reverse decimal digits
13. Decimal digit occurrence count
14. Even/odd classification
15. Divisibility classification
16. Factorial

## Evaluation regimes

- **IID:** inputs with 1–5 decimal digits, matching the training range.
- **Six-digit magnitude OOD:** six-digit inputs are numerically outside training, but under base-100 encoding can still occupy a familiar number of integer tokens.
- **True token-length OOD:** 7–10-digit inputs require longer base-100 token sequences than the model encountered during training.
- **Factorial value OOD:** inputs `101–120`, outside the factorial training domain of `0–100`.
- **Most-common-answer baseline:** accuracy obtained by predicting the most frequent target value for every example in that task's evaluation set.

All reported values below are the mean ± sample standard deviation across three seeds, in percentage points. The primary metric is autoregressive exact-sequence accuracy.

## Overall results

| Architecture | IID macro exact | Overall OOD macro exact | Six-digit macro exact | 7–10-digit macro exact | Factorial value OOD |
|---|---:|---:|---:|---:|---:|
| Causal Transformer | 75.88 ± 1.11 | 16.78 ± 0.68 | 37.87 ± 0.41 | 12.91 ± 0.81 | 0.00 ± 0.00 |
| Causal MLP | 67.44 ± 1.29 | 6.83 ± 0.43 | 25.20 ± 1.67 | 2.81 ± 0.97 | 0.00 ± 0.00 |

The Transformer exceeds the MLP by 8.44 percentage points IID and 9.95 points on overall OOD accuracy. However, both architectures show a large decline between six-digit magnitude OOD and genuine token-length OOD.

## Per-task exact accuracy

| Task | Transformer IID | MLP IID | Transformer 6-digit | MLP 6-digit | Transformer 7–10-digit | MLP 7–10-digit | OOD baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Addition | 36.60 ± 4.90 | 25.37 ± 0.70 | 0.07 ± 0.06 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 |
| Decimal digit occurrence count | 84.95 ± 1.33 | 75.08 ± 1.00 | 55.87 ± 1.26 | 39.37 ± 2.28 | 27.47 ± 8.21 | 4.08 ± 7.07 | 33.40 |
| Decimal digit sum | 87.03 ± 3.95 | 83.37 ± 0.98 | 12.87 ± 4.80 | 12.20 ± 2.26 | 0.01 ± 0.01 | 0.03 ± 0.06 | 4.26 |
| Divisibility | 88.58 ± 1.43 | 84.57 ± 0.38 | 77.87 ± 1.29 | 71.60 ± 1.28 | 45.33 ± 6.09 | 5.73 ± 9.93 | 50.00 |
| Even/odd | 100.00 ± 0.00 | 99.24 ± 0.73 | 100.00 ± 0.00 | 85.17 ± 8.87 | 54.62 ± 3.08 | 1.21 ± 2.09 | 50.00 |
| Factorial | 95.38 ± 3.75 | 87.46 ± 3.02 | — | — | 0.00 ± 0.00* | 0.00 ± 0.00* | 5.00 |
| Greater than | 98.90 ± 0.31 | 94.63 ± 3.06 | 95.80 ± 1.74 | 64.10 ± 23.21 | 37.08 ± 13.80 | 23.36 ± 1.55 | 50.00 |
| Greatest common divisor | 88.95 ± 0.11 | 87.27 ± 0.47 | 83.87 ± 0.32 | 53.87 ± 5.44 | 13.18 ± 7.42 | 0.02 ± 0.03 | 62.28 |
| Integer division | 89.70 ± 1.53 | 79.21 ± 0.83 | 69.60 ± 15.37 | 45.73 ± 8.79 | 15.32 ± 5.90 | 7.57 ± 2.12 | 25.00 |
| Integer-list sum | 12.48 ± 0.70 | 5.65 ± 0.94 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 |
| Modulo | 64.72 ± 1.33 | 44.70 ± 11.24 | 0.40 ± 0.61 | 4.63 ± 8.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.04 |
| Multiplication | 21.34 ± 0.12 | 20.13 ± 0.04 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 |
| Number of decimal digits | 100.00 ± 0.00 | 99.81 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.63 ± 0.36 | 0.18 ± 0.32 | 20.00 |
| Reverse decimal digits | 95.31 ± 0.34 | 63.42 ± 1.79 | 0.07 ± 0.06 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.04 |
| Subtraction | 50.75 ± 7.90 | 32.57 ± 0.19 | 0.57 ± 0.40 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.04 |
| Successor | 99.41 ± 0.09 | 96.54 ± 1.05 | 71.07 ± 15.90 | 1.37 ± 2.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 |

\*For factorial, this column reports value OOD on inputs `101–120`, not 7–10-digit input OOD.

## Output well-formedness

| Task | Transformer IID | MLP IID | Transformer OOD | MLP OOD |
|---|---:|---:|---:|---:|
| Addition | 99.99 ± 0.01 | 99.99 ± 0.02 | 53.66 ± 15.79 | 56.53 ± 34.47 |
| Decimal digit occurrence count | 100.00 ± 0.00 | 100.00 ± 0.00 | 87.07 ± 16.87 | 40.53 ± 26.43 |
| Decimal digit sum | 100.00 ± 0.00 | 100.00 ± 0.00 | 68.53 ± 14.50 | 23.53 ± 6.13 |
| Divisibility | 100.00 ± 0.00 | 100.00 ± 0.00 | 93.60 ± 10.60 | 39.13 ± 26.25 |
| Even/odd | 100.00 ± 0.00 | 100.00 ± 0.00 | 93.56 ± 7.00 | 22.17 ± 3.76 |
| Factorial | 100.00 ± 0.00 | 98.02 ± 0.99 | 88.33 ± 10.41 | 26.67 ± 46.19 |
| Greater than | 100.00 ± 0.00 | 100.00 ± 0.00 | 64.36 ± 13.54 | 73.29 ± 19.62 |
| Greatest common divisor | 100.00 ± 0.00 | 100.00 ± 0.00 | 49.29 ± 6.34 | 20.33 ± 0.24 |
| Integer division | 100.00 ± 0.00 | 100.00 ± 0.00 | 61.33 ± 8.78 | 60.78 ± 11.18 |
| Integer-list sum | 100.00 ± 0.00 | 99.93 ± 0.05 | 63.29 ± 3.81 | 49.11 ± 9.13 |
| Modulo | 99.97 ± 0.03 | 100.00 ± 0.00 | 49.43 ± 2.98 | 62.15 ± 27.78 |
| Multiplication | 100.00 ± 0.00 | 99.77 ± 0.40 | 66.35 ± 10.74 | 50.28 ± 8.77 |
| Number of decimal digits | 100.00 ± 0.00 | 100.00 ± 0.00 | 73.99 ± 25.78 | 21.79 ± 3.05 |
| Reverse decimal digits | 99.99 ± 0.01 | 99.95 ± 0.08 | 43.48 ± 4.02 | 23.73 ± 5.83 |
| Subtraction | 99.59 ± 0.14 | 99.70 ± 0.07 | 43.78 ± 9.76 | 56.19 ± 26.84 |
| Successor | 100.00 ± 0.00 | 99.99 ± 0.01 | 45.05 ± 5.78 | 24.15 ± 6.96 |

Well-formedness must not be interpreted as mathematical correctness. For example, a model can generate an answer with valid syntax while every answer digit is wrong.

## Main findings

1. **T16 learns many tasks IID, but performance is strongly task dependent.** Classification and structural tasks such as even/odd, greater-than comparison, number of digits, successor, and reverse digits are learned much more reliably than multiplication, integer-list sum, addition, and subtraction.

2. **The Transformer consistently outperforms the MLP.** The improvement appears in both IID and OOD macro exact accuracy and is stable across the three seeds.

3. **Magnitude extrapolation and length extrapolation are different.** Performance is substantially higher on six-digit inputs that retain a familiar base-100 token width. Accuracy drops sharply once inputs require unseen token lengths at 7–10 decimal digits.

4. **Hard arithmetic does not exhibit length generalization.** Addition, multiplication, integer-list sum, and subtraction have essentially zero exact accuracy on genuine 7–10-digit OOD inputs. Modulo and reverse digits show the same collapse.

5. **Some simpler tasks retain partial Transformer generalization.** Even/odd reaches 54.62% and divisibility reaches 45.33% on 7–10-digit inputs. Digit occurrence count reaches 27.47%. These values must still be compared with their task baselines.

6. **Raw GCD OOD accuracy is potentially misleading.** The most-common target is `1`, producing a 62.28% baseline. Both architectures fall below this baseline overall, so high accuracy on some GCD subsets is not sufficient evidence of algorithmic generalization. GCD should also be reported separately for targets equal to `1` and greater than `1`.

7. **Factorial is memorized or interpolated within its finite domain but not extrapolated.** Seen-domain accuracy is high, while every architecture and seed obtains 0% exact accuracy on `101–120`.

8. **Seed variance matters for selected tasks.** Greater-than, successor, integer division, modulo, and several MLP OOD results vary substantially across seeds. Reporting only one seed would therefore be unreliable.

## Interpretation limitation

T16 uses the same 20,000-step training budget as T1, T2, T4, and T8. As the task count increases, each individual task receives fewer training examples. Therefore, the experiment measures the effect of increasing task diversity under a fixed compute budget; it does not isolate task diversity from per-task exposure.

The overall T16 macro average also should not be compared directly with a T1, T2, T4, or T8 macro average because the task composition changes. The correct comparison follows each shared task after it enters the nested curriculum:

- Decimal digit sum: T1 → T2 → T4 → T8 → T16
- GCD: T2 → T4 → T8 → T16
- Multiplication and greater-than: T4 → T8 → T16
- Remaining original tasks: T8 → T16

## Next analyses

1. Build a shared-task T1–T16 comparison using identical evaluation regimes and examples.
2. Add T16 as a source model for few-shot transfer to predecessor, LCM, modular addition, and ascending sort.
3. Repeat the most informative transfer comparisons across seeds.
4. Run linear probes and CKA representation-similarity analysis across T1, T2, T4, T8, and T16.
5. Use the shared-task, transfer, and representation results together to evaluate whether increasing task diversity improves reusable mathematical representations.

## Result files

The numerical source of truth is the 12 JSON files under `results/integer-t16-v1/`: one IID and one OOD file for each architecture and seed. This README reports aggregates derived from those files; the JSON files retain per-length, per-stratum, token-accuracy, teacher-forced, baseline, output-distribution, and mistake-level diagnostics.
