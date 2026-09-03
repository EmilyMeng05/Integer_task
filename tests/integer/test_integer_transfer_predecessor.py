from __future__ import annotations

import torch

from neurips_integers import transfer_predecessor as transfer
from neurips_integers.passage import TOKEN_TO_ID as BASE_TOKEN_TO_ID
from neurips_permutations.models import build_model


def test_transfer_vocabulary_preserves_every_base_id() -> None:
    assert all(transfer.TOKEN_TO_ID[token] == index for token, index in BASE_TOKEN_TO_ID.items())
    assert transfer.TOKEN_TO_ID["<PREDECESSOR>"] == len(BASE_TOKEN_TO_ID)


def test_predecessor_rendering() -> None:
    tokens = transfer.predecessor_tokens(247)
    assert tokens == (
        "<BOS>",
        "<SIZE>",
        "03",
        "<NUM_START>",
        "02",
        "47",
        "<NUM_END>",
        "<PREDECESSOR>",
        "=",
        "<NUM_START>",
        "02",
        "46",
        "<NUM_END>",
        "<EOS>",
    )


def test_splits_are_deterministic_and_input_disjoint() -> None:
    first = transfer.build_splits(shots=20, validation_per_length=2, test_per_length=3)
    second = transfer.build_splits(shots=20, validation_per_length=2, test_per_length=3)
    assert first == second
    seen: set[int] = set()
    for records in first.values():
        values = {int(record["inputs"]["primary"]) for record in records}
        assert not (seen & values)
        seen.update(values)
    assert len(first["train"]) == 20


def test_default_splits_retain_one_digit_validation_and_iid_examples() -> None:
    splits = transfer.build_splits()
    validation_one_digit = [row for row in splits["validation"] if row["n_digits"] == 1]
    iid_one_digit = [row for row in splits["iid"] if row["n_digits"] == 1]
    assert len(validation_one_digit) == 2
    assert len(iid_one_digit) == 6


def test_pretrained_copy_preserves_old_rows_and_new_row_shape() -> None:
    old = build_model(
        model_type="transformer",
        vocab_size=len(BASE_TOKEN_TO_ID),
        max_seq_len=32,
        d_model=16,
        layers=1,
        n_heads=4,
        dropout=0.0,
        tie_embeddings=True,
    )
    new = build_model(
        model_type="transformer",
        vocab_size=len(transfer.TOKEN_TO_ID),
        max_seq_len=32,
        d_model=16,
        layers=1,
        n_heads=4,
        dropout=0.0,
        tie_embeddings=True,
    )
    transfer.copy_pretrained_weights(new, old.state_dict())
    assert torch.equal(
        new.state_dict()["token_embedding.weight"][: len(BASE_TOKEN_TO_ID)],
        old.state_dict()["token_embedding.weight"],
    )
    assert new.state_dict()["token_embedding.weight"].shape[0] == len(BASE_TOKEN_TO_ID) + 1
