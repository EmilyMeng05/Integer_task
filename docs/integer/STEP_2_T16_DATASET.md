# Step 2: Generate and Verify the T16 Dataset

## Outcome

This stage creates a new eight-million-record corpus containing sixteen tasks.
The existing `integer-4m-v1` corpus and T1–T8 checkpoints are not modified.

## Fair-comparison guarantees

- The original eight task names, order, sampling seed, inputs, answers, and
  base-100 rendering are preserved by mapping each T16 occurrence back to the
  corresponding legacy record.
- New task/operator tokens are appended to the vocabulary. Existing token IDs
  do not change.
- Each task receives 500,000 records.
- The corpus still uses 98 training shards, one validation shard, and one test
  shard.
- The eight new tasks use the audited `integer-t16/sampling-v1` protocol.
- Factorial is trained only on inputs 0–100.

## Sequence examples

Subtraction:

```text
<BOS> <SIZE> 03 <NUM_START> 02 47 <NUM_END> -
<ARG_START> 85 <ARG_END> <SUBTRACTION> =
<NUM_START> 01 62 <NUM_END> <EOS>
```

Factorial:

```text
<BOS> <SIZE> 01 05 ! <FACTORIAL> =
<NUM_START> 01 20 <NUM_END> <EOS>
```

## Pilot gate

Run the complete tests, then generate and verify a 16,000-record pilot:

```bash
python3 -m pytest -q tests/integer

python3 -m neurips_integers.generate_t16 \
  --output-dir data/integer-t16-16k-pilot \
  --count 16000 \
  --shard-size 160

python3 -m neurips_integers.verify_t16 \
  data/integer-t16-16k-pilot
```

Do not generate production data unless all tests and pilot verification pass.

## Production generation

```bash
caffeinate -i python3 -m neurips_integers.generate_t16_production \
  --output-dir data/integer-8m-t16-v1 \
  --count 8000000 \
  --shard-size 80000 \
  --workers 4
```

Generation is resumable. If interrupted, rerun the same command; verified
completed shards are reused.

## Full verification

```bash
caffeinate -i python3 -m neurips_integers.verify_t16 \
  data/integer-8m-t16-v1
```

Expected final counts:

```text
Full T16 verification passed: 8,000,000 records
Each of the 16 tasks: 500,000 records
```

The generated dataset and checkpoints should remain ignored by Git. Commit the
code, tests, configuration, documentation, and small manifest/report files.
