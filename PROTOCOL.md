# Dataset protocol

Protocol version: `permutation-20/v2`.

## Corpus unit and sampling

One data record is one causal-language-model sequence containing one task. The
production corpus contains 10,000,000 records and is exactly balanced across
the 20 tasks below. Permutation size is sampled uniformly from 2 through 30,
unless a task requires a deliberately balanced construction. Sampling is
deterministic from the global seed and shard number; duplicates are allowed.

All permutations belong to the symmetric group `S_n` and use 1-based values
and positions. Numbers use canonical big-endian base 100. Values 0 through 99
are single two-digit tokens. Values of 100 or more are wrapped in
`<NUM_START>` and `<NUM_END>`.

## The 20 tasks

### Encoding / translation

1. `to_cycle`: canonical disjoint cycles. Singleton cycles are included; each
   cycle begins at its least value; cycles are ordered by their least values.
2. `to_lehmer`: `L_i = #{j > i : pi_j < pi_i}`.
3. `to_inversion_vector`: value-indexed
   `I_v = #{u > v : position(u) < position(v)}`.
4. `to_reduced_word`: the deterministic reduced word in adjacent generators
   returned by stable bubble sorting. Products act on the right, so `pi s_i`
   swaps positions `i` and `i+1`.

### Statistics / properties

5. `length`: Coxeter length, equal to inversion count.
6. `descents`: the number of indices `i` with `pi_i > pi_(i+1)`.
7. `fixed_points`: the number of `i` with `pi_i = i`.
8. `parity`: inversion count modulo 2 (`00` even, `01` odd).
9. `cycle_type`: all cycle lengths, including 1s, sorted decreasingly.
10. `rsk_shape`: row lengths of the RSK insertion tableau, decreasingly.
11. `lis_length`: longest strictly increasing subsequence length.
12. `lds_length`: longest strictly decreasing subsequence length.
13. `pattern_avoidance`: whether `pi` avoids the supplied classical pattern;
    `01` means avoids and `00` means contains.

### Algebraic operations / comparisons

Function composition is `(a o b)(i) = a(b(i))`.

14. `inverse`: `pi^-1`.
15. `compose`: `pi o sigma` for the supplied second permutation `sigma`.
16. `power`: `pi^k` for a supplied nonnegative `0 <= k <= 100`.
17. `conjugate`: `g o pi o g^-1` for the supplied conjugator `g`.
18. `commutator`: `[pi,sigma] = pi o sigma o pi^-1 o sigma^-1`.
19. `right_multiply_simple`: `pi s_i`, swapping one-line positions `i` and
    `i+1`, for `1 <= i < n`.
20. `bruhat_leq`: strong Bruhat comparison `u <= v`. Positive and negative
    examples are matched by permutation size and positive Coxeter-length gap
    (gaps 1 through 4), so the label cannot be inferred from inversion counts.
    The positives include non-cover comparable pairs and the negatives are
    incomparable pairs. With
    `r_w(p,q) = #{i <= p : w(i) <= q}`, this holds exactly when
    `r_u(p,q) >= r_v(p,q)` for every `p,q`.

## Passage Math grammar

The supplied prefix and scalar-task layout is preserved:

```text
<BOS> <SIZE> ENCODE(n) PRIMARY [OPERANDS] <TASK> = ANSWER <EOS>
```

`PRIMARY` is one-line notation bounded by `<ONE_START>` and `<ONE_END>`.
Entries are comma separated. A second permutation uses `<ARG_START>` and
`<ARG_END>`. Pattern, exponent, and simple-reflection operands have their own
typed boundary/label tokens. Structured answers have typed boundaries; scalar
and Boolean answers use number encoding. There is exactly one task token and
one answer in every sequence, so the end-of-input, task, and equals positions
remain useful mechanistic-interpretability landmarks.

Example inherited from the supplied format:

```text
<BOS> <SIZE> 04 <ONE_START> 03 , 01 , 04 , 02 <ONE_END> <DESCENTS> = 02 <EOS>
```

The authoritative token construction is implemented in
`neurips_permutations.passage`; tests freeze every operand and answer form.

## Reproducibility and storage

- JSON Lines is used as the container; each line stores an ID, task, size,
  token list, canonical space-separated text, and minimal structured metadata.
- Production shards use deterministic gzip with no timestamp in the header.
- Files are written to a temporary name, flushed and fsynced, then atomically
  renamed.
- A manifest contains SHA-256 hashes and exact per-task counts.
- Data shards are not committed to Git because GitHub is not suitable for the
  resulting multi-gigabyte artifact.
