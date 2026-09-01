# Integer Multitask Protocol

## Goal

Study how generalization changes as models are trained on increasingly diverse integer tasks. The setup mirrors Xuanyu's permutation experiment.

## Tasks

I reduced the original list by five tasks: maximum of two integers, minimum of two integers, most-significant digit, ones digit, and digit at a specified decimal place.

The corpus contains 20 tasks:

1. Successor
2. Predecessor
3. Addition
4. Subtraction
5. Multiplication
6. Integer division
7. Modulo
8. Greater-than comparison
9. Number of decimal digits
10. Sum of decimal digits
11. Reverse decimal digits
12. Count occurrences of a decimal digit
13. Even/odd classification
14. Divisibility classification
15. Greatest common divisor
16. Least common multiple
17. Modular addition
18. Sort ascending
19. Sum of an integer list
20. Factorial
 

**Check:** Do you approve this exact 20-task list? In particular, should we remove `maximum`, `minimum`, `most-significant digit`, `ones digit`, and `digit at a specified decimal place`, or should any of them replace one of the retained tasks?

## Fixed Holdout Tasks

As in the permutation setup, four tasks are fixed holdouts: **predecessor, least common multiple, modular addition, and sort ascending**. They are not used to train the base models. Later, they are used to test whether broader multitask pretraining helps models adapt to unseen tasks.

- Predecessor is related to successor and subtraction.
- Least common multiple is related to greatest common divisor and multiplication.
- Modular addition is related to addition and modulo.
- Sort ascending is related to comparison and integer-list sum.

The other 16 tasks are shuffled once and used as nested training subsets containing **1, 2, 4, 8, and 16 tasks**.

**Check:** Do you approve using `predecessor`, `least common multiple`, `modular addition`, and `sort ascending` as the four tasks that receive no base-model training updates? If not, which specific tasks should replace them before we freeze the shuffled 16-task training order?

## Experimental Setup

The following settings mirror Xuanyu's permutation experiment.

### Data

- Same canonical base-100 encoding as the permutation model
- One task and one answer per sequence
- Answer-only causal language-model loss
- 10,000,000 balanced records: 500,000 per task
- 98 training shards, 1 validation shard, and 1 test shard
- Small pilot datasets generated and verified before the production corpus

Example base-100 encoding:

```text
7   -> 07
99  -> 99
100 -> <NUM_START> 01 00 <NUM_END>
137 -> <NUM_START> 01 37 <NUM_END>
```

Tasks involving digits still use ordinary decimal definitions. For example, the decimal digit sum of `137` is `11`, even though the model receives the base-100 tokens `01 37`.

**Check:** For `number of digits`, `digit sum`, `reverse digits`, and `digit occurrence count`, should the mathematical target be defined using ordinary decimal digits while only the model input/output encoding uses base 100? For example, should `digit_sum(137)` equal `11` even though `137` is tokenized as `01 37`, as in `digit_sum(137)` equals to `38` base 100?

For example, 247 + 85 = 332 would be represented approximately as:

<BOS>
<NUM_START> 02 47 <NUM_END>
+ 
<ARG_START> 85 <ARG_END>
<ADDITION>
=
<NUM_START> 03 32 <NUM_END>
<EOS>

For ordinary integer tasks, the proposed length split is **1-5 decimal digits for training/IID** and **6-10 decimal digits for length OOD**. List tasks will additionally test longer lists. Factorial will use a separate bounded range because its answers grow quickly.

**Check:** Do you approve sampling 1-5 decimal-digit operands for training/IID and 6-10 decimal-digit operands for length OOD? Do you also approve generating exactly 10,000,000 records (500,000 per task), rather than using a smaller corpus while keeping the same 20,000-update training budget?

### Models

We use the same **Causal Transformer** and **Causal MLP**, model dimensions, training budget, and seeds (`17`, `42`, and `314159`) as the permutation experiment.

```text
5 task-set sizes x 2 architectures x 3 seeds = 30 models
```

**Check:** Should the integer experiment reuse Xuanyu's exact model dimensions and optimization settings—4-layer Transformer, 1-layer Causal MLP, `d_model=256`, 8 heads, context length 1024, and 20,000 optimizer updates—and train all 30 combinations? If yes, which GPU should be used for the formal runs?

### Evaluation

We report:

- Per-task loss
- Token accuracy
- Exact-sequence accuracy
- IID performance
- Length-OOD performance
- Few-shot adaptation on the holdout tasks
- Linear-probe performance on the holdout tasks

Because unseen task tokens have no grounded meaning before adaptation, zero-shot holdout accuracy will be treated as a diagnostic rather than the primary result.

**Check:** For each of the four holdout tasks, should we fine-tune every base model using exactly 20, 50, and 100 labeled examples, with identical learning rates and update counts, and compare against a randomly initialized model trained on the same examples? Should these few-shot results and linear probes be the primary holdout analyses rather than zero-shot exact match?