"""Plan and run the formal integer T1/T2/T4/T8 training matrix.

This module is an integer-specific controller over the shared permutation
trainer. It does not duplicate model, data-loader, optimizer, checkpoint, or
validation code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any, Iterable, Mapping, Sequence

from neurips_permutations.training import TrainConfig, parse_shard_indices

from .generate import TASK_NAMES
from .training import automatic_device, train_integer_run


DEFAULT_CONFIG = Path("configs/henry_integer_production.toml")
EXPECTED_SUBSET_SIZES = (1, 2, 4, 8)
EXPECTED_ARCHITECTURES = ("transformer", "mlp")
EXPECTED_SEEDS = (17, 42, 314159)


@dataclass(frozen=True, slots=True)
class ProductionRun:
    architecture: str
    task_count: int
    tasks: tuple[str, ...]
    seed: int
    run_id: str
    output_dir: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    config = tomllib.loads(payload.decode("utf-8"))
    validate_protocol(config)
    return config, _sha256_bytes(payload)


def validate_protocol(config: Mapping[str, Any]) -> None:
    """Reject accidental changes to the frozen Stage 1 experiment design."""

    if tuple(config.get("task_order", ())) != TASK_NAMES:
        raise ValueError("task_order must match the frozen eight-task order")
    if tuple(config.get("task_subset_sizes", ())) != EXPECTED_SUBSET_SIZES:
        raise ValueError("task_subset_sizes must be 1,2,4,8")
    if tuple(config.get("architectures", ())) != EXPECTED_ARCHITECTURES:
        raise ValueError("architectures must be transformer and mlp")
    if tuple(config.get("model_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("model_seeds must be 17,42,314159")
    if int(config.get("total_records", -1)) != 4_000_000:
        raise ValueError("production corpus must contain 4,000,000 records")
    if int(config.get("records_per_task", -1)) != 500_000:
        raise ValueError("each task must contain 500,000 production records")
    if config.get("dataset_manifest") != "data/integer-4m-v1/manifest.json":
        raise ValueError("dataset_manifest must identify the verified 4M corpus")
    if not isinstance(config.get("output_dir"), str):
        raise ValueError("output_dir is required")
    if not isinstance(config.get("preflight_output_dir"), str):
        raise ValueError("preflight_output_dir is required")

    data = config.get("data")
    model = config.get("model")
    training = config.get("training")
    if not all(isinstance(value, Mapping) for value in (data, model, training)):
        raise ValueError("data, model, and training tables are required")
    assert isinstance(data, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(training, Mapping)
    if data.get("train_shards") != "000-097":
        raise ValueError("formal training must use shards 000-097")
    if data.get("validation_shards") != "098":
        raise ValueError("formal validation must use shard 098")
    if data.get("test_shards") != "099":
        raise ValueError("formal test must remain untouched as shard 099")
    if int(model.get("d_model", 0)) < 1:
        raise ValueError("d_model must be positive")
    if int(training.get("max_steps", 0)) < 1:
        raise ValueError("max_steps must be positive")
    if int(training.get("micro_batch_size", 0)) < 1:
        raise ValueError("micro_batch_size must be positive")
    if int(training.get("gradient_accumulation_steps", 0)) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")


def build_matrix(config_path: Path = DEFAULT_CONFIG) -> tuple[ProductionRun, ...]:
    config, _ = read_config(config_path)
    root = Path(str(config["output_dir"]))
    runs: list[ProductionRun] = []
    for task_count in EXPECTED_SUBSET_SIZES:
        tasks = TASK_NAMES[:task_count]
        for architecture in EXPECTED_ARCHITECTURES:
            for seed in EXPECTED_SEEDS:
                run_id = f"{architecture}-tasks{task_count:02d}-seed{seed}"
                runs.append(
                    ProductionRun(
                        architecture=architecture,
                        task_count=task_count,
                        tasks=tasks,
                        seed=seed,
                        run_id=run_id,
                        output_dir=str(root / run_id),
                    )
                )
    if len(runs) != 24 or len({run.run_id for run in runs}) != 24:
        raise RuntimeError("formal integer matrix must contain 24 unique runs")
    return tuple(runs)


def build_train_config(
    run: ProductionRun,
    *,
    config_path: Path = DEFAULT_CONFIG,
    preflight_steps: int | None = None,
    device: str | None = None,
) -> TrainConfig:
    """Construct one formal or isolated preflight training configuration."""

    config, config_sha256 = read_config(config_path)
    data = config["data"]
    model = config["model"]
    training = config["training"]
    assert isinstance(data, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(training, Mapping)

    if preflight_steps is not None:
        if isinstance(preflight_steps, bool) or preflight_steps < 1:
            raise ValueError("preflight_steps must be a positive integer")
        output_dir = str(
            Path(str(config["preflight_output_dir"]))
            / f"steps{preflight_steps:05d}"
            / run.run_id
        )
        max_steps = preflight_steps
        warmup_steps = max(1, min(preflight_steps, preflight_steps // 10 or 1))
        checkpoint_every = preflight_steps
        validate_every = preflight_steps
        validation_batches_per_task = 1
    else:
        output_dir = run.output_dir
        max_steps = int(training["max_steps"])
        warmup_steps = int(training["warmup_steps"])
        checkpoint_every = int(training["checkpoint_every_steps"])
        validate_every = int(training["validate_every_steps"])
        validation_batches_per_task = int(training["validation_batches_per_task"])

    precision = str(training.get("precision", "bf16")).lower()
    manifest = str(config["dataset_manifest"])
    return TrainConfig(
        manifest=manifest,
        validation_manifest=manifest,
        output_dir=output_dir,
        architecture=run.architecture,
        d_model=int(model["d_model"]),
        num_layers=int(
            model["transformer_layers"]
            if run.architecture == "transformer"
            else model["mlp_layers"]
        ),
        num_heads=int(model["num_heads"]),
        dropout=float(model["dropout"]),
        mlp_ratio=float(model["ff_multiplier"]),
        tie_embeddings=bool(model["tie_embeddings"]),
        tasks=run.tasks,
        validation_tasks=run.tasks,
        seed=run.seed,
        max_steps=max_steps,
        batch_size=int(training["micro_batch_size"]),
        validation_batch_size=int(training["micro_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_seq_len=int(data["max_sequence_length"]),
        max_tokens_per_batch=int(training["max_tokens_per_batch"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_steps=warmup_steps,
        min_lr_ratio=float(training["min_learning_rate_ratio"]),
        max_grad_norm=float(training["gradient_clip_norm"]),
        shuffle_buffer_size=int(data["shuffle_buffer"]),
        num_workers=int(training["num_workers"]),
        checkpoint_every=checkpoint_every,
        validate_every=validate_every,
        validation_batches_per_task=validation_batches_per_task,
        device=device or automatic_device(),
        amp=precision in {"bf16", "bfloat16", "fp16", "float16"},
        bf16=precision in {"bf16", "bfloat16"},
        resume="auto",
        shard_indices=parse_shard_indices(str(data["train_shards"])),
        validation_shard_indices=parse_shard_indices(
            str(data["validation_shards"])
        ),
        experiment_config=str(config_path),
        experiment_config_sha256=config_sha256,
    )


def _completion_is_valid(
    run: ProductionRun,
    *,
    config_path: Path,
) -> bool:
    try:
        config, config_sha256 = read_config(config_path)
        marker_path = Path(run.output_dir) / "completed.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checkpoint = Path(run.output_dir) / "checkpoint.pt"
        return bool(
            isinstance(marker, dict)
            and marker.get("status") == "completed"
            and marker.get("run_id") == run.run_id
            and marker.get("architecture") == run.architecture
            and marker.get("tasks") == list(run.tasks)
            and marker.get("seed") == run.seed
            and marker.get("global_step") == int(config["training"]["max_steps"])
            and marker.get("experiment_config_sha256") == config_sha256
            and marker.get("training_manifest_sha256")
            == _sha256_file(Path(str(config["dataset_manifest"])))
            and checkpoint.is_file()
            and marker.get("checkpoint_sha256") == _sha256_file(checkpoint)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def matrix_status(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    complete: list[str] = []
    incomplete: list[str] = []
    for run in build_matrix(config_path):
        target = complete if _completion_is_valid(run, config_path=config_path) else incomplete
        target.append(run.run_id)
    return {
        "run_count": 24,
        "complete_count": len(complete),
        "incomplete_count": len(incomplete),
        "complete": complete,
        "incomplete": incomplete,
    }


def _selected_runs(
    runs: Sequence[ProductionRun], only: Iterable[str]
) -> tuple[ProductionRun, ...]:
    requested = tuple(only)
    if not requested:
        return tuple(runs)
    known = {run.run_id: run for run in runs}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"unknown run IDs: {', '.join(unknown)}")
    selected = set(requested)
    return tuple(run for run in runs if run.run_id in selected)


def run_matrix(
    *,
    config_path: Path = DEFAULT_CONFIG,
    only: Iterable[str] = (),
    preflight_steps: int | None = None,
    dry_run: bool = False,
    device: str | None = None,
) -> int:
    runs = _selected_runs(build_matrix(config_path), only)
    for run in runs:
        if preflight_steps is None and _completion_is_valid(
            run, config_path=config_path
        ):
            print(f"Skipping verified completed run: {run.run_id}", flush=True)
            continue
        train_config = build_train_config(
            run,
            config_path=config_path,
            preflight_steps=preflight_steps,
            device=device,
        )
        if dry_run:
            print(
                json.dumps(
                    {"run": asdict(run), "train_config": asdict(train_config)},
                    indent=2,
                    sort_keys=True,
                )
            )
            continue
        label = "preflight" if preflight_steps is not None else "formal"
        print(f"Starting {label} run: {run.run_id}", flush=True)
        summary = train_integer_run(train_config)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        expected_steps = preflight_steps or int(read_config(config_path)[0]["training"]["max_steps"])
        if summary.get("status") != "completed" or summary.get("global_step") != expected_steps:
            raise RuntimeError(f"run did not complete safely: {run.run_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--steps", type=int, default=10, help="preflight steps")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan:
        print(json.dumps([asdict(run) for run in build_matrix(args.config)], indent=2))
        return 0
    if args.status:
        print(json.dumps(matrix_status(args.config), indent=2))
        return 0
    return run_matrix(
        config_path=args.config,
        only=args.only,
        preflight_steps=args.steps if args.preflight else None,
        dry_run=args.dry_run,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "ProductionRun",
    "build_matrix",
    "build_train_config",
    "main",
    "matrix_status",
    "read_config",
    "run_matrix",
    "validate_protocol",
]
