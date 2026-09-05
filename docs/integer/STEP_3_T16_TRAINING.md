# Step 3: T16 Training

## Goal

Train the complete 16-task integer model after verifying the 8-million-record corpus. This extends the earlier nested sequence T1, T2, T4, and T8 to T16.

## Fixed T16 task set

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
13. Decimal-digit occurrence count
14. Even/odd classification
15. Divisibility classification
16. Factorial

The four transfer-only holdouts remain predecessor, least common multiple, modular addition, and ascending sort. They are not included in T16 pretraining.

## Controlled comparison

T16 uses the same settings as T1/T2/T4/T8:

- Causal Transformer: 4 layers, width 256, 8 heads
- Causal MLP: 1 layer, width 256
- 20,000 optimizer steps
- Batch size 16 and gradient accumulation 4
- Learning rate `3e-4`
- Seeds `17`, `42`, and `314159`
- Answer-only causal language-model loss
- Training shards `000–097`, validation shard `098`, test shard `099`

This creates six formal runs: `1 task-set size × 2 architectures × 3 seeds`.

The optimizer-step budget remains fixed. Therefore T16 does not receive sixteen times more optimization than T1; it divides the same training budget across a more diverse task mixture. That is necessary for asking whether task diversity helps under fixed compute.

The shared loss first averages answer-token loss within each example and then averages examples. Consequently, tasks with long answers, especially factorial, do not receive extra weight merely because their answers contain more tokens.

## Per-task monitoring

Every validation event reports, separately for all 16 tasks:

- teacher-forced loss;
- exact-sequence accuracy;
- answer-token accuracy;
- example and supervised-token counts.

This lets us identify task interference. For example, high aggregate accuracy must not hide poor multiplication or factorial performance.

## Commands

Activate the environment and install the edited package:

```bash
source .venv/bin/activate
python3 -m pip install -e ".[test,train]"
```

Run the integer tests:

```bash
python3 -m pytest -q tests/integer
```

Confirm the six-run plan:

```bash
python3 -m neurips_integers.production_t16 --plan
```

Run a short Transformer and MLP preflight before formal training:

```bash
caffeinate -i python3 -m neurips_integers.production_t16 \
  --preflight \
  --steps 10 \
  --only transformer-tasks16-seed17 \
  --only mlp-tasks16-seed17
```

If both preflights finish and print validation entries for all 16 tasks, begin formal training one run at a time:

```bash
caffeinate -i python3 -m neurips_integers.production_t16 \
  --run \
  --only transformer-tasks16-seed17
```

Then run the remaining five IDs:

```text
transformer-tasks16-seed42
transformer-tasks16-seed314159
mlp-tasks16-seed17
mlp-tasks16-seed42
mlp-tasks16-seed314159
```

Replace the final `--only` value with each ID. The controller resumes interrupted checkpoints and skips a formally completed run only after validating its hashes, step count, task list, and per-task validation results.

Check progress at any time:

```bash
python3 -m neurips_integers.production_t16 --status
```

## Completion criterion

T16 training is complete only when all six run IDs report `completed`. After that, extend the evaluator to the new T16 vocabulary and report IID, magnitude-OOD, and true token-length-OOD accuracy per task before adding T16 to the few-shot-transfer comparison.
