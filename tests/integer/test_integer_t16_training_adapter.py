from __future__ import annotations

from neurips_integers.generate_t16 import TASK_NAMES
from neurips_integers.passage_t16 import TOKEN_TO_ID
from neurips_integers import training_t16


def test_t16_registry_contains_all_tasks_once() -> None:
    assert len(TASK_NAMES) == 16
    assert len(set(TASK_NAMES)) == 16
    assert training_t16.T16_TASKS == tuple(TASK_NAMES)
    assert training_t16.NESTED_TASKS == {16: tuple(TASK_NAMES)}


def test_smoke_config_uses_t16_vocabulary_and_all_tasks() -> None:
    config = training_t16.smoke_config(architecture="transformer", max_steps=2)
    assert config.tasks == tuple(TASK_NAMES)
    assert config.validation_tasks == tuple(TASK_NAMES)
    assert config.d_model == 256
    assert config.num_layers == 4
    assert config.max_steps == 2
    assert config.shard_indices == tuple(range(98))
    assert config.validation_shard_indices == (98,)
    assert len(TOKEN_TO_ID) > 100


def test_mlp_smoke_config_matches_shared_dimensions() -> None:
    config = training_t16.smoke_config(architecture="mlp", max_steps=2)
    assert config.d_model == 256
    assert config.num_layers == 1
    assert config.max_seq_len == 1024


def test_invalid_architecture_is_rejected() -> None:
    try:
        training_t16.smoke_config(architecture="permuformer")
    except ValueError as exc:
        assert "transformer or mlp" in str(exc)
    else:
        raise AssertionError("invalid architecture was accepted")
