# Step 1: Freeze and Audit T16 Sampling

## Purpose

Before generating the eight-million-record T16 corpus, we freeze and test how
the eight new tasks are sampled. This avoids generating a large corpus with
class imbalance, missing arithmetic behaviors, or outputs that exceed the
model context.

The sampling for the original eight tasks is **not changed**. Therefore the
existing T1, T2, T4, and T8 checkpoints remain directly comparable with T16.

In particular, this step does not introduce carry-balanced addition into T16.
Doing that only for T16 would mix two experimental changes: task count and data
distribution. A future carry-balanced protocol must be given a new version and
must retrain T1, T2, T4, T8, and T16 under the same revised sampling rules.

## New-task sampling rules

| Task | Sampling strata |
|---|---|
| Subtraction | Borrow and no-borrow; `left >= right` |
| Integer division | Quotient zero, quotient one, exact quotient >=2, and nonzero remainder |
| Number of decimal digits | Equal counts for ordinary decimal input lengths 1–5 |
| Reverse decimal digits | With and without trailing zeros |
| Digit occurrence count | Target digit appears zero, once, or multiple times |
| Even/odd | 50% even and 50% odd |
| Divisibility | 50% divisible and 50% not divisible; divisors 2–9 |
| Factorial | Inputs 0–100, balanced across 0–20, 21–50, and 51–100 |

All digit-based mathematical definitions use ordinary decimal digits. Base 100
is only the model's token encoding. For example, the decimal digit sum of 137
is 11 even though 137 is encoded using the base-100 chunks `01 37`.

## Important factorial exception

Factorial is not sampled with 1–5-digit inputs. A five-digit factorial has an
answer far beyond the model context. We instead use inputs 0–100; `100!` uses
79 base-100 answer chunks and fits the current conservative 80-chunk bound.
This bounded finite domain must be stated in the paper.

## Run the audit

From the repository root with the virtual environment active:

```bash
source .venv/bin/activate
python3 -m pip install -e ".[test,train]"
python3 -m pytest -q tests/integer/test_integer_sampling_protocol.py
python3 -m neurips_integers.sampling_protocol \
  --samples-per-task-per-length 1000
```

The command checks 40,000 candidate samples and saves:

```text
results/integer-sampling/t16-sampling-audit.json
```

The report must contain `"status": "passed"`. Review the per-task strata before
starting production generation.

## What this step does not do

This step does not generate the T16 corpus and does not train a model. Once the
sampling report passes and the factorial convention is accepted, Step 2 will
integrate these samplers into the base-100 production generator.
