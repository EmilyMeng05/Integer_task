# Integer Multitask Protocol

## Goal

Study how generalization changes as models are trained on increasingly diverse integer tasks. The setup mirrors the permutation experiment.

## Tasks

**Note, I have cut down 5 tasks: maximum of two integers, minimum of two integers, most-significant digit, ones digit, and digit at a specified decimal place**

We have 20 tasks: 

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

Similar to Xuanyu's set up, I will fix four tasks: **predecessor, least common multiple, modular addition, and sort ascending**. The other 16 tasks will be shuffled once and used as nested training subsets of the **1, 2, 4, 8, and 16 tasks**.

# Below are experimentation details that are exactly the same as written in Xuanyu's doc

## Data

- Same base-100 encoding as the permutation model
- One task and one answer per sequence
- Answer-only causal language-model loss
- 10,000,000 balanced records: 500,000 per task
- 98 training shards, 1 validation shard, and 1 test shard
- Small pilot datasets are generated and verified before the production corpus

## Models

We use the same **Causal Transformer** and **Causal MLP**, model dimensions, training budget, and seeds (`17`, `42`, `314159`) as the permutation experiment. This gives:

```text
5 task-set sizes x 2 architectures x 3 seeds = 30 models
```

## Evaluation

We report per-task loss, token accuracy, exact-sequence accuracy, IID and length-OOD performance, and later evaluate few-shot adaptation and linear probes on the holdout tasks.