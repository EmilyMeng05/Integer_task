# Predecessor Few-Shot Transfer Pilot

## Question

Does broader integer-task pretraining make a model faster or easier to adapt to a held-out task?

Predecessor was never included in T1, T2, T4, or T8 pretraining. We fine-tune five otherwise matched Transformers on the same 20 predecessor examples:

- T1 pretrained checkpoint
- T2 pretrained checkpoint
- T4 pretrained checkpoint
- T8 pretrained checkpoint
- Randomly initialized control

The pretrained checkpoints do not contain a `<PREDECESSOR>` token. The transfer code safely appends one vocabulary row without changing any existing token ID. It copies every compatible pretrained parameter and initializes only the new task-token row randomly.

Example:

```text
<BOS> <SIZE> 03 <NUM_START> 02 47 <NUM_END> <PREDECESSOR> = <NUM_START> 02 46 <NUM_END> <EOS>
```

Only the answer tokens `<NUM_START> 02 46 <NUM_END> <EOS>` contribute to the causal language-model loss.

## 1. Install and test

From the repository root:

```bash
source .venv/bin/activate
python3 -m pip install -e ".[test,train]"
python3 -m pytest -q tests/integer/test_integer_transfer_predecessor.py
```

## 2. Inspect the plan

```bash
python3 -m neurips_integers.transfer_predecessor
```

This is a dry run and does not train anything.

## 3. Run one pipeline check

Start with only T8 and 10 update steps:

```bash
caffeinate -i python3 -m neurips_integers.transfer_predecessor \
  --run \
  --source t8 \
  --seed 17 \
  --steps 10 \
  --evaluation-every 10
```

Confirm that it finishes and writes:

```text
results/integer-transfer-v1/predecessor/transformer-seed17-t8.json
```

Delete or ignore this smoke result before the formal pilot because it used only 10 steps.

## 4. Run the five-way seed-17 pilot

```bash
caffeinate -i python3 -m neurips_integers.transfer_predecessor \
  --run \
  --seed 17 \
  --shots 20 \
  --steps 500 \
  --learning-rate 1e-4 \
  --evaluation-every 50
```

With no `--source` arguments, the command runs T1, T2, T4, T8, and the random control sequentially.

## Outputs

Fine-tuned checkpoints:

```text
runs/henry-integer-transfer-v1/predecessor/transformer/seed17/{t1,t2,t4,t8,random}/checkpoint.pt
```

Metrics:

```text
results/integer-transfer-v1/predecessor/transformer-seed17-{t1,t2,t4,t8,random}.json
```

Each result contains:

- validation loss and exact accuracy every 50 steps
- final IID exact accuracy and well-formed rate
- IID accuracy by decimal length
- final 6–10-digit OOD accuracy by decimal length

The positive one-digit predecessor domain contains only nine unique inputs.
To keep all splits input-disjoint, one is used for few-shot training, two for
validation, and six for IID testing. Lengths 2–5 use the larger requested
evaluation counts.

## Interpretation

The main comparison is not simply which model has the highest final score. We also compare learning curves. Evidence for positive transfer would be that T8 reaches a useful exact accuracy in fewer updates or obtains a higher final held-out accuracy than T1, T2, T4, and the random control.

Do not launch all three seeds until this seed-17 pilot has been inspected. If it behaves correctly, repeat it with seeds `42` and `314159`, then report mean and standard deviation.
