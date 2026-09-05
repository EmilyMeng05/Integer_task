# Step 5: Shared-Task T1–T16 Comparison

## Goal

Measure how each task changes as the nested training set grows. A task is only compared after it enters training:

- Decimal digit sum: T1, T2, T4, T8, T16
- GCD: T2, T4, T8, T16
- Multiplication and greater-than: T4, T8, T16
- Integer-list sum, modulo, addition, and successor: T8, T16

The script reads existing evaluation JSON files. It does not train or evaluate any model.

## Required inputs

The following directories must contain the full results generated with 1,000 examples per task per length:

```text
results/integer-v1/
results/integer-t16-v1/
```

All T1, T2, T4, T8, and T16 combinations must be present for both architectures and all three seeds.

## Run

```bash
source .venv/bin/activate
python3 -m pytest -q tests/integer

python3 -m neurips_integers.compare_task_scaling
```

## Outputs

```text
results/integer-task-scaling-v1/task_scaling_records.csv
results/integer-task-scaling-v1/task_scaling_summary.json
results/integer-task-scaling-v1/README_TASK_SCALING_RESULTS.md
```

The Markdown report contains IID, six-digit magnitude-OOD, and 7–10-digit token-length-OOD trajectories, plus the change from the first trained stage to T16.

## Important interpretation constraint

The existing T1–T8 and T16 JSON files record different evaluation-data identities. The analysis therefore provides a distribution-level comparison with matched decimal-length definitions and sample counts, not a paired comparison on identical examples. The generated report detects and states this automatically.

Also, all stages use 20,000 training steps. Later stages consequently receive fewer examples per task. The results measure task diversity under fixed compute, not task diversity with fixed per-task exposure.
