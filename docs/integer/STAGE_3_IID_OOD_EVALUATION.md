# Stage 3: IID and Length-OOD Evaluation

## Goal

Stage 3 measures whether models trained on 1–5-digit inputs can generate
correct answers for both familiar input lengths and unseen 6–10-digit inputs.
All results are reported separately by task and decimal length.

This stage deliberately distinguishes:

- **teacher-forced evaluation**, where the correct earlier answer tokens are
  available while predicting the next token; and
- **autoregressive evaluation**, where the model receives only the prompt and
  must generate its complete answer.

The primary behavioral metric is autoregressive exact-answer accuracy.

## Evaluation datasets

### IID test

Production shard `099` is the frozen IID test set. It contains:

```text
8 tasks x 5 lengths x 1,000 examples = 40,000 examples
```

These examples use 1–5-digit inputs like training data, but they were never
used for optimization or validation.

### Length-OOD test

The separate OOD corpus contains:

```text
8 tasks x 5 lengths x 1,000 examples = 40,000 examples
```

Its inputs have 6–10 decimal digits. Because training uses only 1–5-digit
inputs, the two ranges cannot overlap.

Generate and independently verify the OOD corpus:

```bash
python3 -m neurips_integers.generate_ood
python3 -m neurips_integers.verify data/integer-ood-40k-v1
```

Expected verification totals are 40,000 records and 5,000 records for each of
the eight tasks.

## No answer leakage

For every record, the evaluator splits the canonical sequence immediately
after `=`. For example:

```text
Prompt supplied to model:
<BOS> <SIZE> 03 <NUM_START> 02 47 <NUM_END> +
<ARG_START> 85 <ARG_END> <ADDITION> =

Hidden expected answer:
<NUM_START> 03 32 <NUM_END> <EOS>
```

The model receives only the prompt. Greedy decoding continues until `<EOS>` or
a fixed cap of 32 generated tokens. The cap is identical for every example, so
the evaluator does not reveal the correct answer length.

## Metrics

For each model, split, task, and input length, the evaluator reports:

- teacher-forced loss;
- teacher-forced token accuracy;
- teacher-forced sequence accuracy;
- autoregressive exact-answer accuracy;
- autoregressive token accuracy;
- well-formed answer rate;
- supervised-token and example counts; and
- the first 20 generation mistakes.

Exact accuracy requires a complete canonical integer followed by `<EOS>` whose
decoded value equals the mathematical answer. Missing, malformed, truncated,
or extra tokens cause exact-match failure.

## Step 1: test the evaluation code

```bash
python3 -m pytest -q tests/integer
```

The tests include a miniature 6–10-digit corpus, independent full verification,
prompt/answer separation, canonical answer parsing, and prompt-only greedy
decoding.

## Step 2: generate and verify OOD data

```bash
python3 -m neurips_integers.generate_ood
python3 -m neurips_integers.verify data/integer-ood-40k-v1
```

The OOD data remains local and must not be committed to Git.

## Step 3: small evaluator preflight

Evaluate 20 examples per length for the first Transformer checkpoint:

```bash
caffeinate -i python3 -m neurips_integers.evaluate \
  --only transformer-tasks01-seed17 \
  --split both \
  --examples-per-task-per-length 20
```

This evaluates 100 IID and 100 OOD digit-sum examples. Confirm that both result
files are written under `results/integer-v1/` and that all metric values are
finite.

## Step 4: full T1 evaluation

After the preflight succeeds, evaluate every currently completed T1 model on
all 1,000 examples per length:

```bash
caffeinate -i python3 -m neurips_integers.evaluate \
  --split both \
  --examples-per-task-per-length 1000
```

With only the six T1 checkpoints completed, this evaluates those six models.
Later, the same command automatically includes newly completed T2, T4, and T8
models. Existing result files are reused only when their checkpoint hash,
evaluation-manifest hash, sample count, and decoding cap all match.

## Output files

Each model receives separate IID and OOD JSON files:

```text
results/integer-v1/<run-id>-iid.json
results/integer-v1/<run-id>-ood.json
```

The files retain checkpoint and data hashes, making it possible to verify that
every model was evaluated on the same frozen examples.

## Interpretation

High teacher-forced accuracy with low autoregressive exact match means the model
can predict answer tokens when earlier correct tokens are supplied but cannot
reliably generate the full answer itself. High IID accuracy with low OOD
accuracy indicates learning within the training length range without length
generalization.

The main cross-model analysis will compare mean performance and seed variation
for `T1`, `T2`, `T4`, and `T8`, separately for Transformer and MLP. A task is
never hidden inside an overall average; every task retains its own loss and
accuracy rows.
