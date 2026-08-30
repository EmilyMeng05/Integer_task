# Henry permutation experiment matrix

The frozen experiment protocol is in `configs/henry_permutation.toml`.

Twenty tasks are deterministically shuffled once. The first 16 form the
training pool; four complete tasks are never shown during base pretraining and
are reserved for zero-shot evaluation, few-shot fine-tuning, and linear probes.
The nested training subsets contain 1, 2, 4, 8, and 16 tasks.

For every subset we train both architectures with three independent parameter
seeds:

```text
5 task subsets x 2 architectures x 3 seeds = 30 base models
```

All runs use a fixed optimizer-update budget. This controls total compute while
the number of tasks changes. Checkpoints and optimizer/RNG state are resumable;
a run is complete only when it has an atomic `completed.json` marker with final
validation metrics and the checksum of its final checkpoint.

The initial model size and 20,000-step cap are deliberately modest for the
single RTX 5070 12 GB host. A throughput/convergence pilot on one Transformer
and one MLP run is required before the matrix starts; the cap may only be
changed before any matrix run is accepted as final, and the frozen config hash
must then change for every run.

