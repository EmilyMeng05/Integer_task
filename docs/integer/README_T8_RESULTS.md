# T8 Results: Eight-Task Integer Multitask Training

## Purpose

T8 tests whether training on eight integer tasks changes in-distribution learning and length generalization relative to the nested T1, T2, and T4 models.

The eight tasks were:

1. Decimal digit sum
2. Greatest common divisor (GCD)
3. Multiplication
4. Greater-than comparison
5. Integer-list sum
6. Modulo
7. Addition
8. Successor

We trained both the causal Transformer and causal MLP using seeds `17`, `42`, and `314159`. Each run used the same base-100 tokenizer, architecture settings, 20,000-step budget, and evaluation protocol as the earlier task-set sizes. Because the total training budget was fixed, an eight-task run saw about 160,000 examples per task.

## Evaluation definitions

- **IID:** 1–5 decimal-digit inputs, matching the range used to generate training data.
- **Six-digit OOD:** numerically larger inputs that still have a familiar base-100 token width. This mainly tests magnitude extrapolation.
- **7–10-digit OOD:** inputs requiring more base-100 tokens than training examples. This is the stricter token-length extrapolation test.
- **Exact accuracy:** the complete generated answer must be correct.
- **Well-formed:** the output follows the required token grammar, whether or not its numerical answer is correct.
- **Baseline:** accuracy obtained by always predicting the most common target in that evaluation set. A model below this baseline has not demonstrated useful generalization on that slice.

Each IID value below uses 5,000 examples per task. Six-digit OOD uses 1,000 examples per task, and 7–10-digit OOD uses 4,000 examples per task.

## Three-seed summary

Values are mean ± sample standard deviation across the three seeds.

### Causal Transformer

| Task | IID exact | IID TF loss | 6-digit exact | 7–10 exact | 7–10 well-formed | 7–10 baseline |
|---|---:|---:|---:|---:|---:|---:|
| Decimal digit sum | 99.77 ± 0.06% | 0.0114 | 76.73 ± 3.48% | 0.00% | 37.58 ± 26.59% | 4.78% |
| GCD | 89.21 ± 0.10% | 0.2369 | 83.30 ± 0.61% | 1.37 ± 1.20% | 33.93 ± 22.79% | 61.30% |
| Multiplication | 23.11 ± 0.18% | 1.5287 | 0.00% | 0.00% | 42.62 ± 36.17% | 0.03% |
| Greater than | 99.37 ± 0.27% | 0.0074 | 98.53 ± 0.81% | 11.46 ± 7.41% | 19.62 ± 9.71% | 50.00% |
| Integer-list sum | 17.91 ± 3.41% | 1.2600 | 0.00% | 0.00% | 44.73 ± 5.13% | 0.03% |
| Modulo | 67.85 ± 1.50% | 0.5616 | 1.90 ± 1.95% | 0.00% | 6.05 ± 3.99% | 0.03% |
| Addition | 81.47 ± 6.68% | 0.1485 | 3.47 ± 3.26% | 0.00% | 12.00 ± 10.51% | 0.03% |
| Successor | 99.71 ± 0.08% | 0.0031 | 90.13 ± 4.99% | 0.00% | 43.61 ± 14.75% | 0.03% |

### Causal MLP

| Task | IID exact | IID TF loss | 6-digit exact | 7–10 exact | 7–10 well-formed | 7–10 baseline |
|---|---:|---:|---:|---:|---:|---:|
| Decimal digit sum | 94.49 ± 2.98% | 0.1104 | 33.53 ± 22.51% | 0.00% | 0.23 ± 0.22% | 4.78% |
| GCD | 88.37 ± 0.56% | 0.2730 | 50.93 ± 5.01% | 0.00% | 3.15 ± 5.46% | 61.30% |
| Multiplication | 21.39 ± 0.36% | 1.8166 | 0.00% | 0.00% | 24.40 ± 23.23% | 0.03% |
| Greater than | 97.39 ± 0.42% | 0.0300 | 62.07 ± 8.32% | 3.44 ± 4.76% | 33.84 ± 28.44% | 50.00% |
| Integer-list sum | 10.74 ± 1.00% | 1.4254 | 0.00% | 0.00% | 24.37 ± 11.18% | 0.03% |
| Modulo | 64.39 ± 0.99% | 0.7183 | 5.40 ± 6.42% | 0.00% | 26.76 ± 24.01% | 0.03% |
| Addition | 38.80 ± 1.32% | 1.0558 | 0.00% | 0.00% | 24.99 ± 23.86% | 0.03% |
| Successor | 99.17 ± 0.03% | 0.0138 | 5.33 ± 4.02% | 0.00% | 0.52 ± 0.73% | 0.03% |

## Complete per-seed exact accuracies

### Causal Transformer

| Task | Seed | IID | 6 digits | 7–10 digits | 7–10 well-formed |
|---|---:|---:|---:|---:|---:|
| Decimal digit sum | 17 | 99.70% | 77.60% | 0.00% | 11.55% |
| Decimal digit sum | 42 | 99.80% | 79.70% | 0.00% | 36.50% |
| Decimal digit sum | 314159 | 99.82% | 72.90% | 0.00% | 64.70% |
| GCD | 17 | 89.24% | 82.60% | 0.07% | 37.60% |
| GCD | 42 | 89.30% | 83.60% | 1.62% | 54.67% |
| GCD | 314159 | 89.10% | 83.70% | 2.43% | 9.53% |
| Multiplication | 17 | 23.10% | 0.00% | 0.00% | 37.90% |
| Multiplication | 42 | 22.94% | 0.00% | 0.00% | 80.92% |
| Multiplication | 314159 | 23.30% | 0.00% | 0.00% | 9.05% |
| Greater than | 17 | 99.44% | 97.60% | 9.15% | 12.90% |
| Greater than | 42 | 99.08% | 99.00% | 19.75% | 30.75% |
| Greater than | 314159 | 99.60% | 99.00% | 5.47% | 15.22% |
| Integer-list sum | 17 | 15.20% | 0.00% | 0.00% | 44.22% |
| Integer-list sum | 42 | 21.74% | 0.00% | 0.00% | 39.88% |
| Integer-list sum | 314159 | 16.78% | 0.00% | 0.00% | 50.10% |
| Modulo | 17 | 67.10% | 1.80% | 0.00% | 2.38% |
| Modulo | 42 | 66.88% | 0.00% | 0.00% | 10.30% |
| Modulo | 314159 | 69.58% | 3.90% | 0.00% | 5.47% |
| Addition | 17 | 82.54% | 3.70% | 0.00% | 0.03% |
| Addition | 42 | 74.32% | 0.10% | 0.00% | 16.30% |
| Addition | 314159 | 87.54% | 6.60% | 0.00% | 19.68% |
| Successor | 17 | 99.64% | 86.40% | 0.00% | 53.92% |
| Successor | 42 | 99.70% | 88.20% | 0.00% | 50.20% |
| Successor | 314159 | 99.80% | 95.80% | 0.00% | 26.72% |

### Causal MLP

| Task | Seed | IID | 6 digits | 7–10 digits | 7–10 well-formed |
|---|---:|---:|---:|---:|---:|
| Decimal digit sum | 17 | 94.14% | 13.90% | 0.00% | 0.00% |
| Decimal digit sum | 42 | 97.62% | 58.10% | 0.00% | 0.27% |
| Decimal digit sum | 314159 | 91.70% | 28.60% | 0.00% | 0.43% |
| GCD | 17 | 88.64% | 48.40% | 0.00% | 0.00% |
| GCD | 42 | 87.72% | 47.70% | 0.00% | 9.45% |
| GCD | 314159 | 88.74% | 56.70% | 0.00% | 0.00% |
| Multiplication | 17 | 21.32% | 0.00% | 0.00% | 4.50% |
| Multiplication | 42 | 21.78% | 0.00% | 0.00% | 18.77% |
| Multiplication | 314159 | 21.06% | 0.00% | 0.00% | 49.93% |
| Greater than | 17 | 97.26% | 62.80% | 0.05% | 1.00% |
| Greater than | 42 | 97.86% | 53.40% | 8.88% | 50.62% |
| Greater than | 314159 | 97.04% | 70.00% | 1.40% | 49.90% |
| Integer-list sum | 17 | 10.34% | 0.00% | 0.00% | 32.77% |
| Integer-list sum | 42 | 11.88% | 0.00% | 0.00% | 28.65% |
| Integer-list sum | 314159 | 10.00% | 0.00% | 0.00% | 11.68% |
| Modulo | 17 | 65.48% | 0.00% | 0.00% | 2.00% |
| Modulo | 42 | 64.16% | 3.70% | 0.00% | 28.32% |
| Modulo | 314159 | 63.54% | 12.50% | 0.00% | 49.95% |
| Addition | 17 | 37.34% | 0.00% | 0.00% | 2.40% |
| Addition | 42 | 39.90% | 0.00% | 0.00% | 22.62% |
| Addition | 314159 | 39.16% | 0.00% | 0.00% | 49.95% |
| Successor | 17 | 99.18% | 7.40% | 0.00% | 0.00% |
| Successor | 42 | 99.20% | 7.90% | 0.00% | 1.35% |
| Successor | 314159 | 99.14% | 0.70% | 0.00% | 0.20% |

## What the results show

1. **The Transformer retained strong IID performance on simple tasks.** Digit sum, greater-than, and successor remained near 100%. Addition reached 81.47%, whereas the MLP reached 38.80%.

2. **The hardest tasks were not fully learned in-distribution.** Multiplication reached only 23.11% for the Transformer and 21.39% for the MLP. Integer-list sum reached 17.91% and 10.74%. Their OOD failures therefore cannot be attributed only to length extrapolation.

3. **Six-digit performance is not the same as true token-length generalization.** With base-100 encoding, six-digit inputs may retain a familiar token width. The Transformer remained strong on six-digit digit sum, GCD, greater-than, and successor, but almost all exact accuracy disappeared at 7–10 digits.

4. **No reliable 7–10-digit generalization appeared.** The Transformer obtained 11.46% on greater-than and 1.37% on GCD, but these are far below their 50.00% and 61.30% most-common-answer baselines. All other T8 task means were 0%. The MLP also remained below baseline.

5. **Well-formed output is not mathematical correctness.** Some models produced grammatically valid sequences while obtaining zero exact accuracy. Formatting must therefore be reported separately from exact accuracy.

6. **The effect of task diversity is architecture- and task-dependent, not monotonic.** Adding tasks did not produce a general improvement in length extrapolation. The Transformer preserved more useful capabilities than the MLP under the fixed budget, while the MLP showed substantial degradation on several six-digit tests.

## Relation to T1, T2, and T4

The nested design changes both task diversity and the number of examples available per task under a fixed 20,000-step budget. Therefore, a T8-versus-T1 difference should not be interpreted as a pure diversity effect: it may also reflect reduced task-specific training.

Selected three-seed six-digit exact accuracies illustrate the non-monotonic pattern:

| Architecture / task | T1 | T2 | T4 | T8 |
|---|---:|---:|---:|---:|
| Transformer / digit sum | 55.97% | 10.20% | 30.87% | 76.73% |
| MLP / digit sum | 95.67% | 96.70% | 39.23% | 33.53% |
| Transformer / GCD | — | 86.53% | 85.17% | 83.30% |
| MLP / GCD | — | 83.80% | 69.43% | 50.93% |
| Transformer / greater than | — | — | 99.23% | 98.53% |
| MLP / greater than | — | — | 88.97% | 62.07% |

This is evidence of task-specific tradeoffs rather than evidence that more tasks uniformly improve direct length generalization.

## Next experiment

The next scientific stage should evaluate **few-shot transfer to held-out tasks**, because direct 7–10-digit extrapolation is nearly absent and the central question is whether broader pretraining nevertheless produces more useful representations.

The four held-out tasks are predecessor, least common multiple, modular addition, and ascending sort. They were intentionally excluded from every T1–T8 pretraining run. For each pretrained checkpoint, we should fine-tune on the same small number of examples from each held-out task and compare against a randomly initialized control. This tests whether increasing task diversity improves sample efficiency even when zero-shot length generalization fails.

Before the full transfer matrix, run one pilot using:

- held-out task: predecessor
- architecture: Transformer
- seed: 17
- pretrained checkpoints: T1, T2, T4, and T8
- control: randomly initialized Transformer
- identical few-shot examples and fine-tuning hyperparameters for every model
- reported metrics: validation loss, token accuracy, exact accuracy, learning curve, and mean ± standard deviation once all seeds are run
