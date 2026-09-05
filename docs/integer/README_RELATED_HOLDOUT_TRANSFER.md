# LCM and Ascending-Sort Transfer Experiments

## Why these tasks

The predecessor pilot mainly tests transfer from successor because successor appears only in T8. Its strong T8 result is useful, but it does not isolate task diversity.

The next two held-out tasks provide more informative comparisons:

- **Least common multiple (LCM):** GCD first appears in T2, while multiplication appears in T4. Comparing T2, T4, and T8 asks whether progressively broader related pretraining changes LCM adaptation.
- **Ascending sort:** greater-than first appears in T4, while T8 additionally contains an integer-list task. Comparing T4 and T8 asks whether list and comparison experience helps sorting.

Neither target task appeared in pretraining.

## Important improvement

The predecessor validation curve peaked before step 500. This script records the complete curve and restores the checkpoint with the **lowest validation loss** before calculating final IID and OOD results. This reduces the effect of few-shot overfitting.

## Encodings

LCM example:

```text
LCM(12, 18) = 36
<BOS> <SIZE> 02 12 <ARG_START> 18 <ARG_END> <LEAST_COMMON_MULTIPLE> = 36 <EOS>
```

Sorting example:

```text
sort([247, 85, 103]) = [85, 103, 247]
<BOS> <SIZE> 03
<LIST_START> <NUM_START> 02 47 <NUM_END> , 85 , <NUM_START> 01 03 <NUM_END> <LIST_END>
<SORT_ASCENDING> =
<LIST_START> 85 , <NUM_START> 01 03 <NUM_END> , <NUM_START> 02 47 <NUM_END> <LIST_END>
<EOS>
```

Only tokens after `=` contribute to training loss.

## 1. Install and test

```bash
source .venv/bin/activate
python3 -m pip install -e ".[test,train]"
python3 -m pytest -q tests/integer/test_integer_transfer_related_holdouts.py
```

## 2. LCM smoke test

```bash
caffeinate -i python3 -m neurips_integers.transfer_related_holdouts \
  --task least_common_multiple \
  --run \
  --source t2 \
  --seed 17 \
  --steps 10 \
  --evaluation-every 10
```

T2 is used for the smoke test because it is the first checkpoint containing GCD.

## 3. Formal LCM comparison

After the smoke test passes:

```bash
caffeinate -i python3 -m neurips_integers.transfer_related_holdouts \
  --task least_common_multiple \
  --run \
  --seed 17 \
  --shots 20 \
  --steps 500 \
  --learning-rate 1e-4 \
  --evaluation-every 50
```

## 4. Sorting smoke test

```bash
caffeinate -i python3 -m neurips_integers.transfer_related_holdouts \
  --task sort_ascending \
  --run \
  --source t4 \
  --seed 17 \
  --steps 10 \
  --evaluation-every 10
```

T4 is used because it is the first checkpoint containing greater-than comparison.

## 5. Formal sorting comparison

```bash
caffeinate -i python3 -m neurips_integers.transfer_related_holdouts \
  --task sort_ascending \
  --run \
  --seed 17 \
  --shots 20 \
  --steps 500 \
  --learning-rate 1e-4 \
  --evaluation-every 50
```

With no `--source`, each command runs T1, T2, T4, T8, and random initialization sequentially.

## Results

```text
results/integer-transfer-v1/least_common_multiple/
results/integer-transfer-v1/sort_ascending/
```

Each result reports the best validation step, validation learning curve, IID exact and token accuracy, well-formed output rate, teacher-forced loss, and 6–10-digit OOD metrics by length.

Run seed 17 first. Do not launch seeds 42 and 314159 until the seed-17 curves and generated outputs have been checked.
