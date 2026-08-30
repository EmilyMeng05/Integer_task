"""Read-only completion audit for the frozen Henry experiment matrix.

The auditor intentionally does not repair, rename, or delete artifacts.  A
checkpoint is deserialized only after its bytes match the SHA-256 recorded in
``completed.json``; runs without a completion marker are reported as
incomplete without loading their possibly-active checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tomllib
from typing import Any

import torch
from torch import Tensor

from .experiments import ExperimentRun, build_matrix


DEFAULT_CONFIG = Path("configs/henry_permutation.toml")
AUDIT_FORMAT_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PARTIAL_COMPONENT_RE = re.compile(r"(?:^|[._-])(?:tmp|partial|part)(?:$|[._-])")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"code": code, "message": message}
    value.update(details)
    return value


def _sanitize_json_numbers(
    value: Any, *, path: str = "", nonfinite_paths: list[str] | None = None
) -> Any:
    """Return strict-JSON data, replacing NaN/Inf while recording their paths."""

    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite_paths is not None:
            nonfinite_paths.append(path or "<root>")
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_json_numbers(
                child,
                path=f"{path}.{key}" if path else str(key),
                nonfinite_paths=nonfinite_paths,
            )
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_json_numbers(
                child,
                path=f"{path}[{index}]",
                nonfinite_paths=nonfinite_paths,
            )
            for index, child in enumerate(value)
        ]
    return value


def _config_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(value)
    identity.pop("resume", None)
    return identity


def training_config_sha256(value: Mapping[str, Any]) -> str:
    """Reproduce the digest written by ``training.py`` without importing it."""

    payload = json.dumps(
        _config_identity(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_config_path(raw: str | os.PathLike[str], config_path: Path) -> Path:
    """Resolve paths using the runner's CWD semantics with a repo-root fallback."""

    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    project_relative = config_path.resolve().parent.parent / path
    return project_relative if project_relative.exists() else path


def _is_partial_artifact(path: Path) -> bool:
    return bool(_PARTIAL_COMPONENT_RE.search(path.name.lower()))


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _find_partial_artifacts(root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    if not root.exists():
        return [], []
    try:
        artifacts = sorted(
            (path for path in root.rglob("*") if _is_partial_artifact(path)),
            key=lambda path: str(path),
        )
        return artifacts, []
    except OSError as error:
        return [], [
            _issue(
                "partial_scan_failed",
                f"could not scan the output tree for partial artifacts: {error}",
                path=str(root),
            )
        ]


def _iter_tensors(value: Any, path: str) -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_tensors(child, child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_tensors(child, f"{path}[{index}]")


def _finite_tensor_summary(value: Any, *, root_name: str) -> dict[str, Any]:
    tensor_count = 0
    element_count = 0
    nonfinite_element_count = 0
    nonfinite_tensors: list[dict[str, Any]] = []
    check_errors: list[dict[str, Any]] = []

    for path, tensor in _iter_tensors(value, root_name):
        tensor_count += 1
        element_count += tensor.numel()
        try:
            finite = torch.isfinite(tensor)
            if finite.layout != torch.strided:
                finite = finite.to_dense()
            bad_count = int((~finite).sum().item())
        except (RuntimeError, TypeError, NotImplementedError) as error:
            check_errors.append(
                {
                    "path": path,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "error": str(error),
                }
            )
            continue
        if bad_count:
            nonfinite_element_count += bad_count
            nonfinite_tensors.append(
                {
                    "path": path,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "nonfinite_elements": bad_count,
                }
            )

    return {
        "status": (
            "passed"
            if tensor_count > 0 and not nonfinite_tensors and not check_errors
            else "failed"
        ),
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonfinite_element_count": nonfinite_element_count,
        "nonfinite_tensors": nonfinite_tensors,
        "check_errors": check_errors,
    }


def _checkpoint_reference_matches(
    raw: Any,
    *,
    expected: Path,
    marker_path: Path,
    project_root: Path,
) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    raw_path = Path(raw)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend((Path.cwd() / raw_path, project_root / raw_path))
        if raw_path.parent == Path("."):
            candidates.append(marker_path.parent / raw_path)
    try:
        expected_resolved = expected.resolve(strict=False)
        return any(
            candidate.resolve(strict=False) == expected_resolved
            for candidate in candidates
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _check_equal(
    issues: list[dict[str, Any]],
    *,
    code: str,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected or type(actual) is not type(expected):
        issues.append(
            _issue(
                code,
                f"{label} does not match the frozen experiment",
                expected=expected,
                actual=actual,
            )
        )


def audit_run(
    run: ExperimentRun,
    *,
    output_dir: Path,
    expected_steps: int,
    experiment_config_sha256: str,
    training_manifest_sha256: str | None,
    validation_manifest_sha256: str | None,
    project_root: Path,
    partial_artifacts: Sequence[Path] = (),
) -> dict[str, Any]:
    """Audit one expected run without mutating it."""

    marker_path = output_dir / "completed.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    issues: list[dict[str, Any]] = []
    run_partials = [
        _display_path(path, output_dir)
        for path in partial_artifacts
        if path == output_dir or output_dir in path.parents
    ]
    result: dict[str, Any] = {
        "run_id": run.run_id,
        "architecture": run.architecture,
        "task_count": run.task_count,
        "tasks": list(run.tasks),
        "seed": run.seed,
        "output_dir": str(output_dir),
        "marker_path": str(marker_path),
        "checkpoint_path": str(checkpoint_path),
        "marker_present": marker_path.is_file(),
        "checkpoint_present": checkpoint_path.is_file(),
        "expected_global_step": expected_steps,
        "marker_global_step": None,
        "checkpoint_global_step": None,
        "marker_checkpoint_sha256": None,
        "checkpoint_sha256": None,
        "checkpoint_size_bytes": None,
        "model_tensors": {"status": "not_checked"},
        "optimizer_tensors": {"status": "not_checked"},
        "results": {
            "epoch": None,
            "batches_in_epoch": None,
            "last_loss": None,
            "task_accounting": None,
            "validation": None,
        },
        "partial_artifacts": run_partials,
        "issues": issues,
    }
    if run_partials:
        issues.append(
            _issue(
                "partial_artifacts_present",
                "temporary or partial artifacts remain in the run directory",
                artifacts=run_partials,
            )
        )

    # An absent marker is the normal state of an active run.  Do not open its
    # checkpoint: this keeps post-hoc auditing safe to invoke around training.
    if not marker_path.is_file():
        issues.append(
            _issue("completion_marker_missing", "completed.json is missing")
        )
        result["status"] = "incomplete"
        return result

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        issues.append(
            _issue(
                "completion_marker_unreadable",
                f"completed.json could not be parsed: {error}",
            )
        )
        result["status"] = "failed"
        return result
    if not isinstance(marker, Mapping):
        issues.append(
            _issue("completion_marker_not_object", "completed.json is not an object")
        )
        result["status"] = "failed"
        return result

    nonfinite_result_paths: list[str] = []
    result["results"] = _sanitize_json_numbers(
        {
            "epoch": marker.get("epoch"),
            "batches_in_epoch": marker.get("batches_in_epoch"),
            "last_loss": marker.get("last_loss"),
            "task_accounting": marker.get("task_accounting"),
            "validation": marker.get("validation"),
        },
        path="results",
        nonfinite_paths=nonfinite_result_paths,
    )
    if nonfinite_result_paths:
        issues.append(
            _issue(
                "marker_results_not_finite",
                "completed.json contains NaN or infinity in reported results",
                paths=nonfinite_result_paths,
            )
        )

    result["marker_global_step"] = marker.get("global_step")
    result["marker_checkpoint_sha256"] = marker.get("checkpoint_sha256")
    _check_equal(
        issues,
        code="marker_status_invalid",
        label="completion status",
        actual=marker.get("status"),
        expected="completed",
    )
    _check_equal(
        issues,
        code="marker_global_step_mismatch",
        label="marker global_step",
        actual=marker.get("global_step"),
        expected=expected_steps,
    )
    _check_equal(
        issues,
        code="marker_run_id_mismatch",
        label="marker run_id",
        actual=marker.get("run_id"),
        expected=run.run_id,
    )
    _check_equal(
        issues,
        code="marker_architecture_mismatch",
        label="marker architecture",
        actual=marker.get("architecture"),
        expected=run.architecture,
    )
    _check_equal(
        issues,
        code="marker_tasks_mismatch",
        label="marker tasks",
        actual=marker.get("tasks"),
        expected=list(run.tasks),
    )
    _check_equal(
        issues,
        code="marker_seed_mismatch",
        label="marker seed",
        actual=marker.get("seed"),
        expected=run.seed,
    )
    _check_equal(
        issues,
        code="marker_experiment_config_sha256_mismatch",
        label="marker experiment config SHA-256",
        actual=marker.get("experiment_config_sha256"),
        expected=experiment_config_sha256,
    )
    if training_manifest_sha256 is not None:
        _check_equal(
            issues,
            code="marker_training_manifest_sha256_mismatch",
            label="marker training manifest SHA-256",
            actual=marker.get("training_manifest_sha256"),
            expected=training_manifest_sha256,
        )
    if validation_manifest_sha256 is not None:
        _check_equal(
            issues,
            code="marker_validation_manifest_sha256_mismatch",
            label="marker validation manifest SHA-256",
            actual=marker.get("validation_manifest_sha256"),
            expected=validation_manifest_sha256,
        )
    if not _checkpoint_reference_matches(
        marker.get("checkpoint"),
        expected=checkpoint_path,
        marker_path=marker_path,
        project_root=project_root,
    ):
        issues.append(
            _issue(
                "marker_checkpoint_path_mismatch",
                "marker does not reference the expected checkpoint",
                expected=str(checkpoint_path),
                actual=marker.get("checkpoint"),
            )
        )

    if marker.get("status") != "completed":
        # A stopped marker must never cause a possibly-live checkpoint load.
        result["status"] = "failed"
        return result
    if not checkpoint_path.is_file():
        issues.append(_issue("checkpoint_missing", "checkpoint.pt is missing"))
        result["status"] = "failed"
        return result

    marker_checkpoint_sha256 = marker.get("checkpoint_sha256")
    if not isinstance(marker_checkpoint_sha256, str) or not _SHA256_RE.fullmatch(
        marker_checkpoint_sha256
    ):
        issues.append(
            _issue(
                "marker_checkpoint_sha256_invalid",
                "marker checkpoint SHA-256 is not 64 lowercase hex characters",
                actual=marker_checkpoint_sha256,
            )
        )
        result["status"] = "failed"
        return result

    try:
        result["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
        checkpoint_sha256 = _sha256_file(checkpoint_path)
    except OSError as error:
        issues.append(
            _issue("checkpoint_unreadable", f"checkpoint could not be hashed: {error}")
        )
        result["status"] = "failed"
        return result
    result["checkpoint_sha256"] = checkpoint_sha256
    if checkpoint_sha256 != marker_checkpoint_sha256:
        issues.append(
            _issue(
                "checkpoint_sha256_mismatch",
                "checkpoint bytes do not match completed.json",
                expected=marker_checkpoint_sha256,
                actual=checkpoint_sha256,
            )
        )
        # Do not deserialize bytes that failed their recorded integrity check.
        result["status"] = "failed"
        return result

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as error:
        issues.append(
            _issue(
                "checkpoint_deserialization_failed",
                f"checkpoint could not be loaded: {type(error).__name__}: {error}",
            )
        )
        result["status"] = "failed"
        return result

    # A second digest makes concurrent replacement observable rather than
    # accidentally validating a mixture of marker and checkpoint versions.
    try:
        checkpoint_sha256_after_load = _sha256_file(checkpoint_path)
    except OSError as error:
        issues.append(
            _issue(
                "checkpoint_changed_during_audit",
                f"checkpoint became unreadable during audit: {error}",
            )
        )
        result["status"] = "failed"
        return result
    if checkpoint_sha256_after_load != checkpoint_sha256:
        issues.append(
            _issue(
                "checkpoint_changed_during_audit",
                "checkpoint bytes changed while they were being audited",
                before=checkpoint_sha256,
                after=checkpoint_sha256_after_load,
            )
        )

    if not isinstance(checkpoint, Mapping):
        issues.append(
            _issue("checkpoint_not_mapping", "deserialized checkpoint is not a mapping")
        )
        result["status"] = "failed"
        return result
    _check_equal(
        issues,
        code="checkpoint_format_version_mismatch",
        label="checkpoint format_version",
        actual=checkpoint.get("format_version"),
        expected=1,
    )

    state = checkpoint.get("state")
    if isinstance(state, Mapping):
        result["checkpoint_global_step"] = state.get("global_step")
        _check_equal(
            issues,
            code="checkpoint_global_step_mismatch",
            label="checkpoint global_step",
            actual=state.get("global_step"),
            expected=expected_steps,
        )
    else:
        issues.append(
            _issue("checkpoint_state_missing", "checkpoint state is not a mapping")
        )

    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, Mapping):
        try:
            checkpoint_config_sha256 = training_config_sha256(checkpoint_config)
        except (TypeError, ValueError) as error:
            issues.append(
                _issue(
                    "checkpoint_config_unhashable",
                    f"checkpoint training config is not JSON-serializable: {error}",
                )
            )
        else:
            _check_equal(
                issues,
                code="marker_training_config_sha256_mismatch",
                label="marker training config SHA-256",
                actual=marker.get("config_sha256"),
                expected=checkpoint_config_sha256,
            )
        for key, expected in (
            ("architecture", run.architecture),
            ("tasks", tuple(run.tasks)),
            ("seed", run.seed),
            ("max_steps", expected_steps),
            ("output_dir", run.output_dir),
            ("experiment_config_sha256", experiment_config_sha256),
        ):
            actual = checkpoint_config.get(key)
            if key == "tasks" and isinstance(actual, list):
                actual = tuple(actual)
            _check_equal(
                issues,
                code=f"checkpoint_config_{key}_mismatch",
                label=f"checkpoint config {key}",
                actual=actual,
                expected=expected,
            )
    else:
        issues.append(
            _issue("checkpoint_config_missing", "checkpoint config is not a mapping")
        )

    data_fingerprints = checkpoint.get("data_fingerprints")
    if isinstance(data_fingerprints, Mapping):
        for key, expected in (
            ("training_manifest_sha256", training_manifest_sha256),
            ("validation_manifest_sha256", validation_manifest_sha256),
        ):
            if expected is not None:
                _check_equal(
                    issues,
                    code=f"checkpoint_{key}_mismatch",
                    label=f"checkpoint {key}",
                    actual=data_fingerprints.get(key),
                    expected=expected,
                )
    else:
        issues.append(
            _issue(
                "checkpoint_data_fingerprints_missing",
                "checkpoint data_fingerprints is not a mapping",
            )
        )

    model = checkpoint.get("model")
    if isinstance(model, Mapping):
        result["model_tensors"] = _finite_tensor_summary(model, root_name="model")
        if result["model_tensors"]["status"] != "passed":
            issues.append(
                _issue(
                    "model_tensors_not_finite",
                    "model state has no tensors, non-finite values, or uncheckable tensors",
                )
            )
    else:
        issues.append(
            _issue("checkpoint_model_missing", "checkpoint model is not a mapping")
        )

    optimizer = checkpoint.get("optimizer")
    if isinstance(optimizer, Mapping):
        result["optimizer_tensors"] = _finite_tensor_summary(
            optimizer, root_name="optimizer"
        )
        if result["optimizer_tensors"]["status"] != "passed":
            issues.append(
                _issue(
                    "optimizer_tensors_not_finite",
                    "optimizer state has no tensors, non-finite values, or uncheckable tensors",
                )
            )
    else:
        issues.append(
            _issue(
                "checkpoint_optimizer_missing", "checkpoint optimizer is not a mapping"
            )
        )

    result["status"] = "passed" if not issues else "failed"
    return result


def audit_experiment(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return a complete JSON-compatible audit of all configured formal runs."""

    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    runs = build_matrix(config_path)
    project_root = config_path.resolve().parent.parent
    output_root = _resolve_config_path(config["output_dir"], config_path)
    expected_steps = int(config["training"]["max_steps"])
    global_issues: list[dict[str, Any]] = []

    if len(runs) != 30:
        global_issues.append(
            _issue(
                "expected_run_count_mismatch",
                "the frozen Henry matrix must contain exactly 30 runs",
                expected=30,
                actual=len(runs),
            )
        )

    manifest_hashes: dict[str, str | None] = {}
    manifest_paths: dict[str, str] = {}
    for label, key in (
        ("training", "dataset_manifest"),
        ("validation", "validation_manifest"),
    ):
        path = _resolve_config_path(config[key], config_path)
        manifest_paths[label] = str(path)
        try:
            manifest_hashes[label] = _sha256_file(path)
        except OSError as error:
            manifest_hashes[label] = None
            global_issues.append(
                _issue(
                    f"{label}_manifest_unreadable",
                    f"{label} manifest could not be hashed: {error}",
                    path=str(path),
                )
            )

    partial_artifacts, scan_issues = _find_partial_artifacts(output_root)
    global_issues.extend(scan_issues)
    if partial_artifacts:
        global_issues.append(
            _issue(
                "partial_artifacts_present",
                "temporary or partial artifacts remain below the experiment output root",
                artifacts=[
                    _display_path(path, output_root) for path in partial_artifacts
                ],
            )
        )

    run_results = [
        audit_run(
            run,
            output_dir=output_root / run.run_id,
            expected_steps=expected_steps,
            experiment_config_sha256=config_sha256,
            training_manifest_sha256=manifest_hashes["training"],
            validation_manifest_sha256=manifest_hashes["validation"],
            project_root=project_root,
            partial_artifacts=partial_artifacts,
        )
        for run in runs
    ]
    passed_count = sum(run["status"] == "passed" for run in run_results)
    incomplete_count = sum(run["status"] == "incomplete" for run in run_results)
    failed_count = sum(run["status"] == "failed" for run in run_results)
    ok = not global_issues and passed_count == len(runs) == 30

    summary = {
        "audit_format_version": AUDIT_FORMAT_VERSION,
        "status": "passed" if ok else "failed",
        "ok": ok,
        "protocol_version": config.get("protocol_version"),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "output_root": str(output_root),
        "expected_global_step": expected_steps,
        "expected_run_count": 30,
        "run_count": len(run_results),
        "passed_count": passed_count,
        "incomplete_count": incomplete_count,
        "failed_count": failed_count,
        "manifest_paths": manifest_paths,
        "manifest_sha256": manifest_hashes,
        "partial_artifacts": [
            _display_path(path, output_root) for path in partial_artifacts
        ],
        "issues": global_issues,
        "runs": run_results,
    }
    # ``json.loads`` accepts NaN/Infinity by default.  Keep the public API and
    # CLI strict-JSON even for a malformed marker, while retaining an issue at
    # the offending run.
    return _sanitize_json_numbers(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--compact", action="store_true", help="emit JSON on one line"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = audit_experiment(args.config)
    except Exception as error:
        summary = {
            "audit_format_version": AUDIT_FORMAT_VERSION,
            "status": "error",
            "ok": False,
            "config_path": str(args.config),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        exit_code = 2
    else:
        exit_code = 0 if summary["ok"] else 1
    print(
        json.dumps(
            summary,
            indent=None if args.compact else 2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
