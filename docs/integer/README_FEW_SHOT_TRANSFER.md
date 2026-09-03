python3 -m pytest -q tests/integer
git status --short
git diff --check# Integer Few-Shot Transfer Results

## Goal

This experiment asks whether pretraining on more integer tasks produces a model that can learn a new, held-out task from only 20 examples. It compares models pretrained on nested collections of 1, 2, 4, and 8 tasks with a matched randomly initialized Transformer.

These are transfer-learning experiments, not zero-shot tests. The model receives 20 labeled examples from the new task before evaluation.

## Pretraining sources

| Source | Pretraining tasks |
|---|---|
| T1 | Decimal digit sum |
| T2 | T1 + greatest common divisor |
| T4 | T2 + multiplication and greater-than comparison |
| T8 | T4 + integer-list sum, modulo, addition, and successor |
| Random | Same Transformer architecture with no integer-task pretraining |

The task sets are nested, so every larger source contains all tasks from the smaller sources.

## Held-out tasks

The following tasks were excluded from base-model pretraining:

1. Predecessor
2. Least common multiple (LCM)
3. Sort ascending
4. Modular addition

For each held-out task, a new task token is appended to the vocabulary without changing existing token IDs or pretrained parameter rows.

## Shared transfer protocol

- Architecture: causal Transformer
- Model seed reported here: `17`
- Support set: 20 examples covering 1–5 decimal digits
- Fine-tuning steps: 500
- Learning rate: `1e-4`
- Batch size: 4
- Validation interval: 50 steps
- Checkpoint selection: lowest validation teacher-forced loss
- IID evaluation: new 1–5-digit examples
- OOD evaluation: new 6–10-digit examples
- Primary metric: autoregressive exact-sequence accuracy

The validation set selects a checkpoint; it is not used as the final test set. Final IID and OOD evaluation is performed only after restoring the selected checkpoint.

Split sizes differ slightly because small one-digit domains cannot always provide hundreds of distinct, leakage-free examples:

| Task | Train | Validation | IID | OOD |
|---|---:|---:|---:|---:|
| Predecessor | 20 | 82 | 406 | 500 |
| LCM | 20 | 90 | 431 | 500 |
| Sort ascending | 20 | 100 | 500 | 500 |
| Modular addition | 20 | 100 | 500 | 500 |

## Predecessor

| Source | Best step | Best validation loss | IID exact | OOD exact |
|---|---:|---:|---:|---:|
| T1 | 100 | 4.092403 | 0.00% | 0.00% |
| T2 | 100 | 2.851512 | 0.00% | 0.00% |
| T4 | 50 | 2.084223 | 0.27% | 0.00% |
| T8 | 100 | **0.206938** | **83.02%** | **11.20%** |
| Random | 150 | 2.212291 | 0.00% | 0.00% |

T8 transfers strongly to predecessor, whereas the other sources and random initialization remain near zero. The most likely explanation is related-task transfer because successor appears only in T8. Therefore, this result should not yet be interpreted as proof that task diversity alone causes better transfer.

An earlier predecessor evaluation used the final step-500 model and reported 66.31% IID and 10.40% OOD for T8. The table above supersedes that result because the corrected protocol restores the lowest-validation-loss checkpoint before testing.

## Least common multiple

| Source | Best step | Best validation loss | IID exact | OOD exact |
|---|---:|---:|---:|---:|
| T1 | 100 | 5.093615 | 0.46% | 0.00% |
| T2 | 50 | 3.220899 | **2.55%** | 0.00% |
| T4 | 50 | **2.790199** | **2.55%** | 0.00% |
| T8 | 50 | 3.063523 | 1.16% | 0.00% |
| Random | 100 | 3.006168 | 0.23% | 0.00% |

T2 and T4 slightly outperform random initialization, which may reflect transfer from GCD. However, absolute accuracy is very low, T8 does not improve further, and no source generalizes OOD.

## Sort ascending

| Source | Best step | Best validation loss | IID exact | OOD exact |
|---|---:|---:|---:|---:|
| T1 | 150 | 2.912993 | 0.00% | 0.00% |
| T2 | 150 | 2.485547 | 0.00% | 0.00% |
| T4 | 100 | 2.710584 | 0.00% | 0.00% |
| T8 | 100 | 2.172181 | 0.00% | 0.00% |
| Random | 150 | **2.143620** | 0.00% | 0.00% |

All conditions obtain zero exact-match accuracy. Falling validation loss suggests that the models learn some token-level regularities, but 20 examples are insufficient for producing completely correct sorted lists. There is no evidence of a pretraining advantage under this protocol.

## Modular addition

Modular addition is defined as `(a + b) mod m`, where `m` is positive.

| Source | Best step | Best validation loss | IID exact | OOD exact |
|---|---:|---:|---:|---:|
| T1 | 100 | 3.981597 | 2.60% | 0.00% |
| T2 | 50 | 3.053691 | 3.40% | 0.00% |
| T4 | 50 | 3.357094 | 5.40% | 0.00% |
| T8 | 50 | 3.348253 | **7.60%** | 0.00% |
| Random | 100 | **2.447392** | 6.00% | 0.00% |

IID exact accuracy increases from T1 through T8, but T8 exceeds random initialization by only 1.60 percentage points. Because these results use one seed, this is suggestive rather than conclusive. No condition generalizes OOD.

## Current conclusions

1. Pretraining can greatly improve few-shot transfer when the source contains a directly related task, as shown by successor-to-predecessor transfer in T8.
2. Related-task transfer is not uniformly strong: GCD provides at most a small advantage for LCM.
3. Twenty examples are insufficient for exact sorting under the current representation and optimization schedule.
4. Modular addition shows a possible T1-to-T8 trend, but its advantage over random initialization is small.
5. Length extrapolation remains difficult: predecessor is the only held-out task with nonzero OOD exact accuracy.
6. Task count and task identity are confounded in the nested design. T8 uniquely contains successor, addition, and modulo, so improvements cannot automatically be attributed to diversity alone.
7. These are seed-17 Transformer results. Error bars require seeds `42` and `314159`, and comparison with Xuanyu's complete protocol additionally requires T16 and MLP transfer experiments.

## Result locations

Corrected predecessor results:

```text
results/integer-transfer-v2/predecessor/
```

LCM, sorting, and modular-addition results:

```text
results/integer-transfer-v1/least_common_multiple/
results/integer-transfer-v1/sort_ascending/
results/integer-transfer-v1/modular_addition/
```

## Next steps

1. Implement and train T16 for both architectures and all three seeds.
2. Add T16 as a transfer source.
3. Repeat the four transfer experiments using seeds `42` and `314159`.
4. Extend the transfer pipeline to the causal MLP.
5. Report mean and sample standard deviation across seeds.
6. Use probing and representation-similarity analyses to separate task diversity from direct task relatedness.
