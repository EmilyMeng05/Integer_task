"""Checks for the nested integer matrix and shared-trainer adapter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurips_integers.generate import TASK_NAMES
from neurips_integers.passage import TOKEN_TO_ID
from neurips_integers.training import NESTED_TASKS, smoke_config


def test_nested_task_prefixes() -> None:
    assert tuple(NESTED_TASKS) == (1, 2, 4, 8)
    for size, tasks in NESTED_TASKS.items():
        assert tasks == TASK_NAMES[:size]
    assert set(NESTED_TASKS[8]) == set(TASK_NAMES)


@pytest.mark.parametrize("architecture", ["transformer", "mlp"])
@pytest.mark.parametrize("task_count", [1, 2, 4, 8])
def test_smoke_config_matrix(architecture: str, task_count: int) -> None:
    config = smoke_config(
        architecture=architecture,
        task_count=task_count,
        device="cpu",
    )
    assert config.tasks == TASK_NAMES[:task_count]
    assert config.validation_tasks == config.tasks
    assert config.shard_indices == tuple(range(98))
    assert config.validation_shard_indices == (98,)
    assert config.d_model == 256
    assert config.num_layers == (4 if architecture == "transformer" else 1)
    assert config.max_steps == 2
    assert len(TOKEN_TO_ID) > 100
