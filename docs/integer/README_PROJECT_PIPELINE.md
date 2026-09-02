# Integer Multitask Project Pipeline

## Research question

We study whether training a model on a broader collection of integer tasks changes its ability to generalize. We evaluate three forms of generalization:

1. **IID generalization:** new examples with the same 1-5-digit range used during training.
2. **Magnitude generalization:** six-digit inputs, which are numerically larger but still use a familiar three-token base-100 width.
3. **Token-length generalization:** seven-to-ten-digit inputs, which require longer base-100 token sequences than the model saw during training.

Later stages will also test few-shot transfer, linear probes, and representation similarity.

## Training tasks and nested conditions

The current production corpus contains these eight tasks in one fixed order:

1. `decimal_digit_sum`
2. `greatest_common_divisor`
3. `multiplication`
4. `greater_than`
5. `integer_list_sum`
6. `modulo`
7. `addition`
8. `successor`

The models use nested task subsets:

| Condition | Training tasks |
|---|---|
| T1 | Decimal digit sum |
| T2 | T1 + greatest common divisor |
| T4 | T2 + multiplication + greater-than comparison |
| T8 | All eight tasks |

Because the subsets are nested, each larger condition retains all tasks in the previous condition. We do not train every possible pair or subset.

Four future transfer tasks are held out from base-model training: predecessor, least common multiple, modular addition, and ascending sort. They were chosen because each is related to one or more training tasks, allowing us to test whether learned knowledge transfers.

## Data

The verified production corpus contains **4,000,000 records**:

- 500,000 records per task
- 100,000 records per task at each decimal length from 1 through 5
- Balanced task representation in the full corpus
- Answer-only causal language-model supervision
- One task and one answer in each sequence

The evaluation sets contain 1,000 examples per task per decimal length:

| Split | Decimal lengths | Examples per task |
|---|---|---:|
| IID | 1-5 | 5,000 |
| OOD | 6-10 | 5,000 |

## Base-100 encoding

Integers are divided into two-decimal-digit chunks. For example:

```text
7   -> 07
99  -> 99
100 -> 01 00
137 -> 01 37
```

Base 100 changes only the model representation. Mathematical tasks involving decimal digits retain their ordinary decimal definitions. Therefore:

```text
digit_sum(137) = 1 + 3 + 7 = 11
```

It is not `01 + 37 = 38`.

An addition example is represented approximately as:

```text
<BOS>
<NUM_START> 02 47 <NUM_END>
+
<ARG_START> 85 <ARG_END>
<ADDITION>
=
<NUM_START> 03 32 <NUM_END>
<EOS>
```

Prompt tokens are masked from the loss. Only answer tokens and the terminating token are supervised.

## Models and training

We use the same two architecture families and core setup as the permutation experiment:

| Setting | Value |
|---|---|
| Architectures | Causal Transformer and Causal MLP |
| Transformer | 4 layers, 8 attention heads |
| Causal MLP | 1 causal mixing layer |
| Hidden dimension | 256 |
| Maximum context | 1,024 tokens |
| Training budget | 20,000 optimizer steps per model |
| Seeds | 17, 42, 314159 |

The current models use ordinary sequential learned absolute position embeddings. They do **not** use RoPE or the position-coupling method from the earlier PermuFormer experiments.

Running both architectures and three seeds produces six models per task-set size. T1, T2, T4, and T8 therefore require 24 models in the current phase.

## Evaluation metrics

Metrics are always reported separately for every task:

| Metric | Meaning |
|---|---|
| Teacher-forced loss | Answer-token loss when preceding correct answer tokens are supplied |
| Generated token accuracy | Proportion of generated answer tokens that match |
| Exact-sequence accuracy | Entire generated answer is correct |
| Well-formed rate | Output follows the required grammar |
| Most-common-target baseline | Accuracy from always predicting the most frequent target in that evaluation subset |

Per-task reporting is essential because an overall multitask average can hide that one task is learned very well while another is learned poorly.

## Why six digits are reported separately

| Decimal length | Base-100 width | Evaluation meaning |
|---:|---:|---|
| 1-2 | 1 token | Training/IID width |
| 3-4 | 2 tokens | Training/IID width |
| 5-6 | 3 tokens | Six digits are magnitude OOD but familiar in token width |
| 7-8 | 4 tokens | True token-length OOD |
| 9-10 | 5 tokens | True token-length OOD |

This distinction prevents strong six-digit performance from being mistaken for genuine length extrapolation.

## Current status

- Production corpus: generated and fully verified
- T1: six models trained and evaluated
- T2: six models trained and evaluated
- Next: run and validate `transformer-tasks04-seed17`, then complete the remaining T4 seeds and architecture
