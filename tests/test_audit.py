from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch

from neurips_permutations.audit import (
    audit_experiment,
    main,
    training_config_sha256,
)
from neurips_permutations.experiments import build_matrix


SOURCE_CONFIG = Path(__file__).parents[1] / "configs" / "henry_permutation.toml"


def _replace_toml_string(text: str, key: str, value: Path) -> str:
    replacement = f"{key} = {json.dumps(str(value))}"
    updated, count = re.subn(
        rf"^{re.escape(key)}\s*=.*$", replacement, text, flags=re.M
    )
    assert count == 1
    return updated


def _test_config(tmp_path: Path) -> tuple[Path, Path, str]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"shards": []}), encoding="utf-8")
    output_root = tmp_path / "runs"
    text = SOURCE_CONFIG.read_text(encoding="utf-8")
    text = _replace_toml_string(text, "dataset_manifest", manifest)
    text = _replace_toml_string(text, "validation_manifest", manifest)
    text = _replace_toml_string(text, "test_manifest", manifest)
    text = _replace_toml_string(text, "output_dir", output_root)
    config = tmp_path / "experiment.toml"
    config.write_text(text, encoding="utf-8")
    return config, output_root, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _write_completed_run(
    config_path: Path,
    run,
    manifest_sha256: str,
    *,
    step: int = 20_000,
    model_value: float = 1.0,
    optimizer_value: float = 0.0,
    marker_status: str = "completed",
) -> tuple[Path, Path]:
    run_dir = Path(run.output_dir)
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    checkpoint_config = {
        "output_dir": run.output_dir,
        "architecture": run.architecture,
        "tasks": tuple(run.tasks),
        "seed": run.seed,
        "max_steps": 20_000,
        "resume": "auto",
        "experiment_config_sha256": config_sha256,
    }
    checkpoint = {
        "format_version": 1,
        "model": {"weight": torch.tensor([model_value])},
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(20_000.0),
                    "exp_avg": torch.tensor([optimizer_value]),
                }
            },
            "param_groups": [{"params": [0]}],
        },
        "state": {"global_step": step},
        "config": checkpoint_config,
        "data_fingerprints": {
            "training_manifest_sha256": manifest_sha256,
            "validation_manifest_sha256": manifest_sha256,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    marker = {
        "status": marker_status,
        "run_id": run.run_id,
        "architecture": run.architecture,
        "tasks": list(run.tasks),
        "seed": run.seed,
        "global_step": step,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": training_config_sha256(checkpoint_config),
        "experiment_config_sha256": config_sha256,
        "training_manifest_sha256": manifest_sha256,
        "validation_manifest_sha256": manifest_sha256,
        "epoch": 3,
        "batches_in_epoch": 123,
        "last_loss": 0.25,
        "task_accounting": {
            task: {
                "examples": 10,
                "supervised_tokens": 20,
                "mean_example_loss": 0.5,
            }
            for task in run.tasks
        },
        "validation": {"length": {"loss": 0.75, "token_accuracy": 0.5}},
    }
    marker_path = run_dir / "completed.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    return checkpoint_path, marker_path


def test_all_thirty_valid_runs_pass_and_report_tensor_counts(tmp_path: Path) -> None:
    config, _, manifest_sha256 = _test_config(tmp_path)
    runs = build_matrix(config)
    assert len(runs) == 30
    for run in runs:
        _write_completed_run(config, run, manifest_sha256)

    summary = audit_experiment(config)

    assert summary["ok"] is True
    assert summary["status"] == "passed"
    assert summary["run_count"] == summary["passed_count"] == 30
    assert summary["incomplete_count"] == summary["failed_count"] == 0
    assert summary["partial_artifacts"] == []
    first = summary["runs"][0]
    assert first["status"] == "passed"
    assert first["checkpoint_global_step"] == 20_000
    assert first["model_tensors"]["tensor_count"] == 1
    assert first["optimizer_tensors"]["tensor_count"] == 2
    assert first["model_tensors"]["nonfinite_element_count"] == 0
    assert first["results"]["last_loss"] == 0.25
    assert first["results"]["validation"]["length"]["loss"] == 0.75


def test_missing_marker_is_incomplete_and_checkpoint_is_not_loaded(
    tmp_path: Path, monkeypatch
) -> None:
    config, _, _ = _test_config(tmp_path)
    run = build_matrix(config)[0]
    run_dir = Path(run.output_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_bytes(b"an active or incomplete checkpoint")

    def fail_load(*args, **kwargs):
        raise AssertionError("an unmarked checkpoint must not be loaded")

    monkeypatch.setattr(torch, "load", fail_load)
    summary = audit_experiment(config)

    first = summary["runs"][0]
    assert first["status"] == "incomplete"
    assert first["checkpoint_present"] is True
    assert first["model_tensors"]["status"] == "not_checked"
    assert {issue["code"] for issue in first["issues"]} == {
        "completion_marker_missing"
    }


def test_sha_mismatch_blocks_deserialization(tmp_path: Path, monkeypatch) -> None:
    config, _, manifest_sha256 = _test_config(tmp_path)
    run = build_matrix(config)[0]
    checkpoint, marker_path = _write_completed_run(config, run, manifest_sha256)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    def fail_load(*args, **kwargs):
        raise AssertionError("a checkpoint with a bad digest must not be loaded")

    monkeypatch.setattr(torch, "load", fail_load)
    summary = audit_experiment(config)

    first = summary["runs"][0]
    assert marker_path.is_file()
    assert first["status"] == "failed"
    assert "checkpoint_sha256_mismatch" in {
        issue["code"] for issue in first["issues"]
    }
    assert first["model_tensors"]["status"] == "not_checked"


def test_noncompleted_marker_status_blocks_deserialization(
    tmp_path: Path, monkeypatch
) -> None:
    config, _, manifest_sha256 = _test_config(tmp_path)
    run = build_matrix(config)[0]
    _write_completed_run(
        config, run, manifest_sha256, marker_status="stopped"
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("a stopped run checkpoint must not be loaded")

    monkeypatch.setattr(torch, "load", fail_load)
    summary = audit_experiment(config)

    first = summary["runs"][0]
    assert first["status"] == "failed"
    assert "marker_status_invalid" in {
        issue["code"] for issue in first["issues"]
    }


def test_nonfinite_model_optimizer_wrong_step_and_partial_are_reported(
    tmp_path: Path,
) -> None:
    config, output_root, manifest_sha256 = _test_config(tmp_path)
    run = build_matrix(config)[0]
    _write_completed_run(
        config,
        run,
        manifest_sha256,
        step=19_999,
        model_value=float("inf"),
        optimizer_value=float("nan"),
    )
    partial = Path(run.output_dir) / ".checkpoint.pt.deadbeef.tmp"
    partial.write_bytes(b"partial")

    summary = audit_experiment(config)
    first = summary["runs"][0]
    codes = {issue["code"] for issue in first["issues"]}

    assert summary["ok"] is False
    assert summary["partial_artifacts"] == [
        f"{run.run_id}/.checkpoint.pt.deadbeef.tmp"
    ]
    assert output_root.is_dir()
    assert first["status"] == "failed"
    assert {
        "marker_global_step_mismatch",
        "checkpoint_global_step_mismatch",
        "model_tensors_not_finite",
        "optimizer_tensors_not_finite",
        "partial_artifacts_present",
    } <= codes
    assert first["model_tensors"]["nonfinite_element_count"] == 1
    assert first["optimizer_tensors"]["nonfinite_element_count"] == 1


def test_cli_always_emits_machine_readable_json_and_failure_exit(
    tmp_path: Path, capsys
) -> None:
    config, _, _ = _test_config(tmp_path)

    exit_code = main(["--config", str(config), "--compact"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["incomplete_count"] == 30


def test_cli_config_error_is_json_and_uses_distinct_exit_code(
    tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "missing.toml"

    exit_code = main(["--config", str(missing), "--compact"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "FileNotFoundError"
