# Step 4: T16 IID and OOD Evaluation

## Purpose

Evaluate all six completed T16 checkpoints using the T16 vocabulary and the same autoregressive metrics used for T1–T8.

The evaluator reports:

- numerical exact-sequence accuracy;
- generated-token accuracy;
- well-formed output rate;
- teacher-forced loss and accuracy;
- most-common-answer baseline;
- results by task, evaluation bucket, actual decimal length, generalization regime, and sampling stratum;
- the first 20 generation mistakes;
- macro task exact accuracy, so tasks with many examples do not dominate the headline result.

## Generalization regimes

For the fifteen ordinary length-based tasks:

- **IID:** held-out shard `099` with training sampling buckets 1–5.
- **Six-digit magnitude OOD:** six decimal digits require three base-100 tokens, which equals the maximum input-token width observed during 1–5-digit training.
- **Seven-to-ten-digit token-length OOD:** these inputs require more base-100 tokens than any training operand.

Factorial requires a separate definition:

- **Factorial seen domain:** every unique input from 0 through 100.
- **Factorial value OOD:** every unique input from 101 through 120.

The factorial training domain contains only 101 possible inputs, so ordinary random train/test splitting cannot make its 0–100 evaluation semantically disjoint. The report labels this explicitly instead of presenting memorization on 0–100 as unseen IID generalization.

## Data behavior

The evaluator reads IID records from the verified T16 test shard. It creates OOD records deterministically in memory, so no additional large dataset is required.

The OOD protocol preserves balanced task-specific cases:

- subtraction: borrow versus no borrow;
- integer division: zero quotient, one quotient, exact multi-unit quotient, and quotient with remainder;
- reverse digits: trailing-zero versus no-trailing-zero inputs;
- digit occurrence: zero, one, and multiple occurrences;
- even/odd and divisibility: balanced classes;
- addition: zero, one, or multiple decimal carry events;
- successor: ordinary versus trailing-nine cascade cases;
- GCD: target equal to 1 versus target greater than 1.

## 1. Install and test

```bash
source .venv/bin/activate
python3 -m pip install -e ".[test,train]"
python3 -m pytest -q tests/integer
```

## 2. Seed-17 pilot

Run both architectures on 20 examples per task per length:

```bash
caffeinate -i python3 -m neurips_integers.evaluate_t16 \
  --only transformer-tasks16-seed17 \
  --only mlp-tasks16-seed17 \
  --split both \
  --examples-per-task-per-length 20
```

This evaluates 1,601 IID records and 1,520 OOD records per model. Factorial always uses its complete unique domains: 101 IID-domain values and 20 OOD values.

Inspect the printed per-task metrics and confirm that JSON files appear in:

```text
results/integer-t16-v1/
```

## 3. Full seed-17 evaluation

After the pilot succeeds:

```bash
caffeinate -i python3 -m neurips_integers.evaluate_t16 \
  --only transformer-tasks16-seed17 \
  --only mlp-tasks16-seed17 \
  --split both \
  --examples-per-task-per-length 1000 \
  --force
```

## 4. Full six-model evaluation

This is computationally expensive and is a good candidate for the mentor’s faster machine:

```bash
caffeinate -i python3 -m neurips_integers.evaluate_t16 \
  --split both \
  --examples-per-task-per-length 1000 \
  --force
```

The full run evaluates 75,101 IID records and 75,020 OOD records per model.

## Optional task-specific run

Use one or more `--task` flags to diagnose difficult operations without evaluating everything:

```bash
caffeinate -i python3 -m neurips_integers.evaluate_t16 \
  --only transformer-tasks16-seed17 \
  --split both \
  --task addition \
  --task multiplication \
  --task integer_list_sum \
  --examples-per-task-per-length 1000 \
  --force
```

Task-filtered results receive separate filenames and do not overwrite complete evaluations.

## Important interpretation rule

Do not infer length generalization from validation accuracy. Validation uses the training-domain distribution. The OOD report—especially the seven-to-ten-digit `token_length_ood` regime—is the relevant measurement of extrapolation beyond the trained base-100 sequence width.
