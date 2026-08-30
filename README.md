# NeurIPS permutation multitask data

This repository generates an auditable corpus for the permutation half of the
multitask-generalization study. It covers exactly 20 tasks:

- 4 encodings/translations;
- 9 statistics/properties;
- 7 algebraic operations/comparisons.

The default production run writes **10,000,000 final Passage Math sequences**,
balanced exactly across the 20 tasks (500,000 per task). A record is one model
sequence with one task target, matching the supplied Passage Math convention.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest

# Small, deterministic smoke corpus.
permutation-generate --count 2000 --output-dir data/smoke --workers 1
permutation-verify data/smoke/manifest.json --full

# Full corpus: 100 gzip shards of 100,000 records each.
permutation-generate \
  --count 10000000 \
  --max-entries 30 \
  --base 100 \
  --shard-size 100000 \
  --workers 20 \
  --output-dir data/permutation-10m-v2
permutation-verify data/permutation-10m-v2/manifest.json --full --workers 20
permutation-split data/permutation-10m-v2/manifest.json
```

Generation is deterministic, streaming, parallel, and resumable. Completed
shards are reused only after their checksum and record count are verified.
The manifest records the seed, protocol version, task counts, shard hashes, and
byte sizes.

## Important interpretation

The mathematical objects are standard permutations of `{1, ..., n}` with
`2 <= n <= 30`. Thus the largest permutation entry is 30. The requested
"maximum number 100" is implemented as the supplied **base-100 tokenizer**:
`00` through `99` are atomic tokens, while values at least 100 are encoded
canonically between `<NUM_START>` and `<NUM_END>`.

The generated multi-gigabyte corpus is ignored by Git and should be stored as a
release artifact, object-store dataset, or local research artifact rather than
committed to GitHub. A small checked-in sample and its manifest are included for
format review.

See [PROTOCOL.md](PROTOCOL.md) for exact definitions, composition conventions,
canonical output rules, and the extended Passage Math grammar.
