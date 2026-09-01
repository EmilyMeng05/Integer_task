# Stage 1: Integer Multitask Data and Pipeline

## Stage 1 outcome

Stage 1 established and verified the data and training infrastructure for the
integer multitask study. We now have:

- a frozen eight-task training suite;
- nested `T1`, `T2`, `T4`, and `T8` task collections;
- four related tasks reserved for transfer evaluation;
- a shared base-100 sequence format;
- deterministic mathematical generation and independent verification;
- a verified 20,000-record pilot corpus;
- a verified 4,000,000-record production corpus;
- working adapters for the shared Causal Transformer and Causal MLP; and
- a completed eight-run training smoke matrix.

No formal model accuracy results are reported in Stage 1. The purpose of this
stage was to ensure that later comparisons measure task diversity rather than
differences in data formatting, model code, or corrupted examples.

## Research question

The broader project asks:

> How does pretraining on increasingly diverse collections of mathematical
> tasks affect generalization, few-shot transfer, and learned representation
> geometry?

The integer study mirrors the permutation study so that conclusions can be
compared across two different mathematical domains.

## Eight Phase 1 training tasks

The first task pool intentionally combines arithmetic, number theory,
classification, digit structure, and structured-list computation.

| Frozen position | Task | Capability represented |
| ---: | --- | --- |
| 1 | Decimal digit sum | Decimal structure |
| 2 | Greatest common divisor | Number theory |
| 3 | Multiplication | Multi-step arithmetic |
| 4 | Greater-than comparison | Boolean classification |
| 5 | Integer-list sum | Variable-length structured input |
| 6 | Modulo | Division and remainder structure |
| 7 | Addition | Multi-digit arithmetic and carries |
| 8 | Successor | Simple local arithmetic |

The order was shuffled once using seed `20260830` and then frozen. We do not
train every possible combination. Instead, the models receive nested prefixes:

| Condition | Included tasks |
| --- | --- |
| `T1` | Decimal digit sum |
| `T2` | `T1` + greatest common divisor |
| `T4` | `T2` + multiplication + greater-than comparison |
| `T8` | `T4` + integer-list sum + modulo + addition + successor |

Thus:

```text
T1 ⊂ T2 ⊂ T4 ⊂ T8
```

This design changes task diversity in a controlled way while preserving all
tasks from each smaller condition in the larger conditions.

## Transfer holdout tasks

Four tasks are implemented conceptually but excluded from base-model training:

| Holdout task | Related training knowledge |
| --- | --- |
| Predecessor | Successor and subtraction-like behavior |
| Least common multiple | GCD and multiplication |
| Modular addition | Addition and modulo |
| Sort ascending | Comparison and list structure |

The base models never receive training gradients from these tasks. Later, each
trained model will receive the same small number of holdout-task examples. We
can then test whether broader pretraining improves few-shot adaptation. A
randomly initialized model trained on the same examples will provide the
control condition.

## Shared representation format

To stay comparable with the permutation experiment, the integer pipeline
reuses its base-100 tokenizer and sequence grammar. Integers are split into
two-decimal-digit chunks:

```text
7   -> 07
85  -> 85
100 -> <NUM_START> 01 00 <NUM_END>
247 -> <NUM_START> 02 47 <NUM_END>
```

The mathematical meaning remains ordinary decimal arithmetic. Base 100 is
only the model-facing tokenization. For example, `digit_sum(137)` is still
`1 + 3 + 7 = 11`, even though `137` is encoded as `01 37`.

### Complete addition example

The expression `247 + 85 = 332` is encoded as:

```text
<BOS> <SIZE> 03 <NUM_START> 02 47 <NUM_END> +
<ARG_START> 85 <ARG_END> <ADDITION> =
<NUM_START> 03 32 <NUM_END> <EOS>
```

The important pieces are:

- `<SIZE> 03`: the longest input operand contains three decimal digits;
- `02 47`: the base-100 chunks representing 247;
- `+`: the explicit mathematical operator;
- `<ARG_START> ... <ARG_END>`: boundaries around the second operand;
- `<ADDITION>`: the task-identification token; and
- `03 32`: the base-100 representation of the answer 332.

The operator and task token have different purposes: `+` preserves the visible
mathematical expression, while `<ADDITION>` identifies which supervised task
the sequence belongs to.

## Meaning of size

`<SIZE>` is defined using ordinary decimal digit length, not the number of
base-100 tokens:

- unary task: digits in the input integer;
- binary task: maximum digit length of the two operands; and
- list task: maximum digit length of any list element.

Training, validation, and IID testing use inputs with 1–5 decimal digits.
Separate 6–10-digit examples will later measure length extrapolation.

## Answer-only causal loss

Each record contains one task and one answer. During training, prompt tokens,
operands, operators, the task token, and `=` are context. Loss is applied only
to the answer tokens and `<EOS>`. This matches the permutation setup and
prevents longer prompts from dominating the learning signal.

## Deterministic data generation

Mathematical operations are defined in one authoritative implementation. The
generator samples operands, computes the correct result, renders the token
sequence, and writes compressed JSONL shards. Fixed seeds make the corpus
reproducible.

The verifier independently checks:

- schema and record structure;
- mathematical answers;
- token rendering;
- task labels and task counts;
- decimal-length allocations;
- shard sizes and record totals; and
- recorded SHA-256 shard hashes.

Production shards are written atomically. If generation is interrupted, the
same command validates and reuses completed shards before continuing.

## Pilot corpus

The pilot corpus contains 20,000 records:

| Allocation | Count |
| --- | ---: |
| Total records | 20,000 |
| Records per task | 2,500 |
| Records per task and digit length | 500 |
| Compressed shards | 100 |

The pilot was generated and fully verified. It was then used for an eight-run
smoke matrix:

```text
T1, T2, T4, T8 × Transformer, MLP = 8 smoke runs
```

Every smoke run completed its forward pass, backward pass, optimizer steps,
validation, checkpoint writing, and summary writing successfully.

## Production corpus

The production corpus contains 4,000,000 records, preserving the permutation
study's allocation of 500,000 examples per task:

```text
8 tasks × 500,000 examples = 4,000,000 records
```

Every task receives 100,000 records at each input length from 1 through 5
decimal digits.

| Split | Shards | Total records | Records per task |
| --- | --- | ---: | ---: |
| Training | `000–097` | 3,920,000 | 490,000 |
| Validation | `098` | 40,000 | 5,000 |
| IID test | `099` | 40,000 | 5,000 |

The completed production verification reported:

```text
Full verification passed: 4,000,000 records
decimal_digit_sum:          500,000
greatest_common_divisor:    500,000
multiplication:             500,000
greater_than:               500,000
integer_list_sum:           500,000
modulo:                     500,000
addition:                   500,000
successor:                  500,000
```

Generated corpora remain local and are excluded from Git. Code, configurations,
tests, and documentation are version controlled.

## Model infrastructure

This multitask study does not use the earlier PermuFormer checkpoints. It
reuses the same model implementations as the permutation study:

- Causal Transformer; and
- Causal MLP.

The integer adapter changes only the integer vocabulary mapping and required
output vocabulary size. The shared streaming loader, batching, answer masking,
optimizer, scheduler, model code, validation, checkpointing, and summary logic
remain unchanged.

The planned standard model settings are:

| Setting | Value |
| --- | ---: |
| Model dimension | 256 |
| Transformer layers | 4 |
| Transformer heads | 8 |
| MLP layers | 1 |
| Feed-forward multiplier | 4 |
| Dropout | 0.1 |
| Maximum sequence length | 1,024 |

No new position-coupling or RoPE modification is introduced in this stage.
Keeping the shared architectures unchanged makes the integer and permutation
experiments directly comparable. The earlier position-coupled PermuFormer
experiments remain separate preliminary evidence about length generalization.

## Reproduction commands

Install the shared project dependencies:

```bash
python3 -m pip install -e ".[test,train]"
```

Generate and verify the pilot:

```bash
python3 -m neurips_integers.generate
python3 -m neurips_integers.verify data/integer-20k-pilot
```

Inspect and execute the smoke matrix:

```bash
python3 -m neurips_integers.smoke --dry-run
python3 -m neurips_integers.smoke
```

Generate and verify the production corpus:

```bash
python3 -m neurips_integers.generate_production --workers 4
python3 -m neurips_integers.verify data/integer-4m-v1
```

## Stage 1 conclusions

Stage 1 demonstrates that the integer dataset can be generated reproducibly,
fully audited, and consumed by both shared model architectures. It does not yet
answer whether more pretraining tasks improve generalization. That question
begins with formal training and controlled downstream evaluation.

## Next stage

Stage 2 will:

1. add a resumable formal-production training launcher;
2. sanity-check `Transformer | T1 | seed 17` before launching the matrix;
3. train `T1`, `T2`, `T4`, and `T8` Transformer and MLP models;
4. repeat formal runs with seeds `17`, `42`, and `314159` as compute permits;
5. report overall and per-task loss and IID exact accuracy;
6. evaluate 6–10-digit length-OOD generalization;
7. perform few-shot transfer on held-out tasks; and
8. compare hidden representations using CKA and related analyses.

The 4-million-record corpus may also serve as the mentor-requested `10× data`
condition if a matched 400,000-record `1× data` corpus is approved. Model-size
scaling will be treated as a separate controlled variable.
