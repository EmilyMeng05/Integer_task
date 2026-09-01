# Integer Multitask Study: Progress Summary

## Research question

We are studying whether training a small model on a broader collection of integer tasks improves its generalization. In particular, we will compare models trained on nested sets of **1, 2, 4, and 8 tasks**, then test:

- IID performance on familiar input lengths;
- length-OOD performance on longer integers;
- later, transfer learning, linear probing, and representation similarity.

The comparison uses the same data format, training budget, architectures, and random seeds as the permutation study.

## Experimental design

The eight training tasks, shuffled once and then held fixed, are:

1. Decimal digit sum
2. Greatest common divisor
3. Multiplication
4. Greater-than comparison
5. Integer-list sum
6. Modulo
7. Addition
8. Successor

The nested task sets are therefore T1 = task 1, T2 = tasks 1–2, T4 = tasks 1–4, and T8 = all eight tasks. Four related tasks—predecessor, least common multiple, modular addition, and ascending sort—are held out for later transfer and probing experiments.

We compare the same two architectures used in the permutation experiments:

- Causal Transformer
- Causal MLP

Every condition is trained with three seeds: `17`, `42`, and `314159`. This allows us to report means and standard deviations instead of relying on one lucky run.

## Data and encoding

We generated and fully verified **4,000,000 balanced records**:

- 500,000 examples per task;
- 100,000 examples for each decimal length from 1–5, per task;
- 98 training shards, one validation shard, and one IID test shard;
- base-100 integer tokenization;
- one task and one answer per sequence;
- answer-only causal language-model loss.

A simplified digit-sum example is:

```text
<BOS> <SIZE> 03
<ONE_START> 01 37 <ONE_END>
<DECIMAL_DIGIT_SUM> = <NUM_START> 11 <NUM_END>
<EOS>
```

Here, the mathematical input is 137, represented by base-100 chunks `01 37`, and the correct decimal digit sum is 11. Prompt tokens are provided as context, but training loss is applied only to the answer and its ending tokens.

We also generated and verified a separate **40,000-example length-OOD corpus** covering decimal lengths 6–10, with 1,000 examples per task per length.

## Completed stages

1. Implemented and unit-tested the eight integer operations, encoding, generation, and verification pipeline.
2. Generated and fully verified the 20,000-record pilot corpus.
3. Passed the complete smoke matrix for T1/T2/T4/T8 × Transformer/MLP.
4. Generated and fully verified the four-million-record production corpus.
5. Completed all six formal T1 training runs: two architectures × three seeds.
6. Implemented true autoregressive IID/OOD evaluation in which the model receives only the prompt and must generate the full answer.
7. Completed full IID and OOD evaluation for all six T1 models.

## T1 training result

T1 contains only `decimal_digit_sum`. All six models completed 20,000 training steps and achieved 100% teacher-forced sequence and token accuracy on the training-time validation sample.

| Architecture | Seed | Validation loss | Validation sequence accuracy |
|---|---:|---:|---:|
| Transformer | 17 | 0.001990 | 100% |
| Transformer | 42 | 0.001068 | 100% |
| Transformer | 314159 | 0.001989 | 100% |
| Causal MLP | 17 | 0.00000462 | 100% |
| Causal MLP | 42 | 0.00000279 | 100% |
| Causal MLP | 314159 | 0.00000299 | 100% |

## Full autoregressive results

Each model was evaluated on 5,000 IID examples and 5,000 OOD examples. Values below are mean ± sample standard deviation across three seeds.

| Architecture | IID exact accuracy | OOD exact accuracy | OOD well-formed rate | OOD generated-token accuracy |
|---|---:|---:|---:|---:|
| Transformer | **99.97% ± 0.02%** | **11.19% ± 1.03%** | 47.13% ± 5.35% | 20.09% ± 4.88% |
| Causal MLP | **100.00% ± 0.00%** | **19.16% ± 0.58%** | 42.42% ± 12.90% | 23.81% ± 13.18% |

### Exact accuracy for every run

| Architecture | Seed | IID | OOD |
|---|---:|---:|---:|
| Transformer | 17 | 99.98% | 10.14% |
| Transformer | 42 | 99.94% | 11.24% |
| Transformer | 314159 | 99.98% | 12.20% |
| Causal MLP | 17 | 100.00% | 18.52% |
| Causal MLP | 42 | 100.00% | 19.64% |
| Causal MLP | 314159 | 100.00% | 19.32% |

### OOD exact accuracy by decimal length

| Architecture | 6 digits | 7 digits | 8 digits | 9 digits | 10 digits |
|---|---:|---:|---:|---:|---:|
| Transformer | 55.97% ± 5.15% | 0.00% | 0.00% | 0.00% | 0.00% |
| Causal MLP | 95.67% ± 2.87% | 0.03% ± 0.06% | 0.03% ± 0.06% | 0.07% ± 0.06% | 0.00% |

## Main finding so far

Both architectures learned the in-distribution digit-sum task almost perfectly, but neither showed meaningful generalization to genuinely longer base-100 sequences.

This distinction matters because base 100 stores roughly two decimal digits per token:

| Decimal length | Base-100 number-token length |
|---:|---:|
| 1–2 | 1 |
| 3–4 | 2 |
| 5–6 | 3 |
| 7–8 | 4 |
| 9–10 | 5 |

Training used 1–5 decimal digits, so both five- and six-digit inputs can use three base-100 tokens. Six-digit inputs are outside the numerical training range but not necessarily longer at the token level. Seven digits is the first condition that always requires a longer base-100 sequence, and accuracy collapses at exactly that boundary.

Therefore, the current result is not simply “19% versus 11% OOD.” The stronger conclusion is:

> Near-perfect IID learning does not imply token-length generalization; almost all apparent OOD success came from the six-digit bridge condition that retained a familiar base-100 token width.

## Next step

We will train the T2 models on both `decimal_digit_sum` and `greatest_common_divisor`, again using both architectures and all three seeds. Every model will report validation loss and accuracy separately for each task. We will then repeat the same IID/OOD evaluation and determine whether adding a second task changes digit-sum generalization before moving to T4 and T8.
