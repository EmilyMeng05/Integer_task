# Stage 2: Integer Production Training

## Goal

Stage 2 trains the verified integer models on nested collections of 1, 2, 4,
and 8 tasks. The central comparison is whether increasing task diversity
changes in-distribution learning, length generalization, transfer to held-out
tasks, and internal representation geometry.

This stage reuses the permutation project's streaming trainer, Causal
Transformer, and Causal MLP. The new integer production controller only selects
the integer vocabulary, nested task subsets, formal configuration, run IDs,
and output directories.

## Formal matrix

The planned baseline matrix is:

```text
4 task counts x 2 architectures x 3 seeds = 24 models
```

| Variable | Values |
| --- | --- |
| Task count | `T1`, `T2`, `T4`, `T8` |
| Architecture | Transformer, MLP |
| Seed | `17`, `42`, `314159` |
| Training steps | 20,000 |
| Production data | 4,000,000 verified records |

The nested tasks remain frozen:

```text
T1: decimal_digit_sum
T2: T1 + greatest_common_divisor
T4: T2 + multiplication + greater_than
T8: T4 + integer_list_sum + modulo + addition + successor
```

## Data isolation

| Use | Shards |
| --- | --- |
| Training | `000-097` |
| Validation during training | `098` |
| Final IID test | `099` |

Test shard `099` is not read by the production trainer. Length-OOD data and
held-out transfer tasks are also evaluated only after base-model training.

## Metrics retained during training

The shared trainer accumulates the following separately for every included
task:

- number of training examples;
- number of supervised answer tokens;
- mean answer-only training loss;
- validation loss;
- validation answer-token accuracy; and
- validation exact-sequence accuracy.

This prevents a good overall loss from hiding a task that is not being learned.
The training objective averages answer loss within each example before
averaging examples, so tasks with longer answers do not automatically dominate
the optimization objective.

## Safety and resumability

Each run has a stable identifier such as:

```text
transformer-tasks01-seed17
```

Formal outputs are stored under:

```text
runs/henry-integer-v1/<run-id>/
```

The trainer writes checkpoints atomically and records:

- model, optimizer, scheduler, and precision-scaler states;
- Python and Torch random-number-generator states;
- exact training cursor and optimizer step;
- per-task accounting;
- last validation metrics;
- full training configuration; and
- hashes of the data manifests.

Rerunning the same formal command automatically resumes an incomplete run.
Resume is refused if the configuration or dataset fingerprint changed.

## Step 1: run tests

From the repository root:

```bash
python3 -m pytest -q \
  tests/test_integer_math_ops.py \
  tests/test_integer_pipeline.py \
  tests/test_integer_training_adapter.py \
  tests/test_integer_production.py
```

## Step 2: inspect the 24-run plan

```bash
python3 -m neurips_integers.production --plan
```

This command does not train. It prints each run ID, architecture, seed, and
nested task subset.

Inspect the exact first-run configuration without training:

```bash
python3 -m neurips_integers.production \
  --preflight \
  --only transformer-tasks01-seed17 \
  --steps 10 \
  --dry-run
```

## Step 3: isolated ten-step preflight

Run only the first Transformer condition:

```bash
python3 -m neurips_integers.production \
  --preflight \
  --only transformer-tasks01-seed17 \
  --steps 10
```

The preflight uses the real 4M manifest, real training/validation shards, real
model dimensions, and real integer vocabulary. It has a separate short
schedule and output directory:

```text
runs/henry-integer-v1-preflight/steps00010/
```

It cannot contaminate or be resumed as a formal 20,000-step checkpoint. A
successful preflight must finish all ten optimizer steps, validate the included
task, and write both `checkpoint.pt` and `completed.json`.

## Step 4: inspect preflight evidence

Check the output files:

```bash
find \
  runs/henry-integer-v1-preflight/steps00010/transformer-tasks01-seed17 \
  -maxdepth 1 -type f -print
```

Do not start the full matrix until the preflight summary confirms:

- status is `completed`;
- global step is `10`;
- training loss is finite;
- validation loss is finite;
- the validation example/token counts are nonzero; and
- the checkpoint and completion marker exist.

## Step 5: formal training

After reviewing the preflight, start only the first formal run:

```bash
python3 -m neurips_integers.production \
  --run \
  --only transformer-tasks01-seed17
```

Check matrix status at any time:

```bash
python3 -m neurips_integers.production --status
```

Once the first formal run is confirmed, the remaining matrix can be launched
sequentially with:

```bash
python3 -m neurips_integers.production --run
```

Completed runs are authenticated and skipped; interrupted runs resume from
their rolling checkpoint.

## Compute caution

The complete 24-model matrix is much larger than the smoke test and may be
impractical to finish entirely on one MacBook. Runs should remain sequential,
and the first formal runtime should be measured before committing to all seeds.
If compute is limited, prioritize complete experimental cells over many
partially trained models and ask the mentors which seeds/architectures should
receive priority.

## Scaling experiments

The mentor-requested `1x/10x data` and `1x/2x model` comparison is a separate
controlled experiment. The 4M corpus can serve as the 10x-data condition if a
matched 400K corpus is approved. Stage 2 baseline training does not silently
change model capacity or data scale.
