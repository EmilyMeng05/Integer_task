from __future__ import annotations

from pathlib import Path

import pytest

from neurips_integers import production_t16


CONFIG = Path("configs/henry_integer_t16_training.toml")


def test_t16_matrix_has_six_unique_runs() -> None:
    runs = production_t16.build_matrix(CONFIG)
    assert len(runs) == 6
    assert len({run.run_id for run in runs}) == 6
    assert {run.architecture for run in runs} == {"transformer", "mlp"}
    assert {run.seed for run in runs} == {17, 42, 314159}
    assert all(len(run.tasks) == 16 for run in runs)
    assert all(run.tasks == production_t16.TASK_NAMES for run in runs)


def test_formal_train_config_matches_fixed_compute_protocol() -> None:
    run = production_t16.build_matrix(CONFIG)[0]
    train = production_t16.build_train_config(run, config_path=CONFIG)
    assert train.manifest == "data/integer-8m-t16-v1/manifest.json"
    assert train.max_steps == 20_000
    assert train.batch_size == 16
    assert train.gradient_accumulation_steps == 4
    assert train.learning_rate == 3.0e-4
    assert train.tasks == production_t16.TASK_NAMES
    assert train.output_dir == str(Path("runs/henry-integer-t16-v1") / run.run_id)
    assert train.resume == "auto"


def test_preflight_is_short_and_uses_separate_output() -> None:
    run = production_t16.build_matrix(CONFIG)[0]
    train = production_t16.build_train_config(
        run, config_path=CONFIG, preflight_steps=3
    )
    assert train.max_steps == 3
    assert train.batch_size == 16
    assert train.gradient_accumulation_steps == 4
    assert train.validate_every == 3
    assert "preflight" in str(train.output_dir)


def test_protocol_rejects_changed_task_order() -> None:
    config, _ = production_t16.read_config(CONFIG)
    config["task_order"] = list(reversed(config["task_order"]))
    with pytest.raises(ValueError, match=r"task[_ ]order"):
        production_t16.validate_protocol(config)


def test_select_runs_rejects_unknown_id() -> None:
    runs = production_t16.build_matrix(CONFIG)
    with pytest.raises(ValueError, match="unknown run IDs"):
        production_t16._selected_runs(runs, ["transformer-tasks08-seed17"])
