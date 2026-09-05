"""Autoregressive IID and task-aware OOD evaluation for T16 checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from neurips_permutations.training import TrainConfig

from . import evaluate as base_evaluate
from .evaluation_t16_data import (
    FACTORIAL_IID_DOMAIN,
    build_ood_records,
    legacy_diagnostic_stratum,
)
from .generate_t16 import TASK_NAMES
from .passage_t16 import ID_TO_TOKEN, TOKEN_TO_ID, decode_number
from .production_t16 import (
    DEFAULT_CONFIG,
    ProductionRun,
    build_matrix,
    completion_is_valid,
)
from .training_t16 import automatic_device, build_t16_model


DEFAULT_IID_MANIFEST = Path("data/integer-8m-t16-v1/manifest.json")
DEFAULT_OUTPUT_DIR = Path("results/integer-t16-v1")
FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["id"]).encode())
        digest.update(b"\0")
        digest.update(str(record["canonical_text"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _t16_evaluation_runtime() -> Iterator[None]:
    """Temporarily point the proven evaluator helpers at the T16 vocabulary."""

    original = (
        base_evaluate.TOKEN_TO_ID,
        base_evaluate.ID_TO_TOKEN,
        base_evaluate.decode_number,
    )
    base_evaluate.TOKEN_TO_ID = TOKEN_TO_ID
    base_evaluate.ID_TO_TOKEN = ID_TO_TOKEN
    base_evaluate.decode_number = decode_number
    try:
        yield
    finally:
        (
            base_evaluate.TOKEN_TO_ID,
            base_evaluate.ID_TO_TOKEN,
            base_evaluate.decode_number,
        ) = original


def _manifest_records(manifest_path: Path, shard_index: int) -> Iterator[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not 0 <= shard_index < len(entries):
        raise ValueError("invalid T16 manifest or shard index")
    entry = entries[shard_index]
    if not isinstance(entry, Mapping) or not isinstance(entry.get("filename"), str):
        raise ValueError("invalid T16 manifest shard entry")
    path = manifest_path.parent / str(entry["filename"])
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("evaluation shard contains a non-object record")
            yield record


def _iid_stratum(record: Mapping[str, Any]) -> str:
    value = record.get("sampling_stratum")
    if value == "legacy_random" or value is None:
        return legacy_diagnostic_stratum(record)
    return str(value)


def load_iid_records(
    manifest_path: Path,
    *,
    tasks: Sequence[str] = TASK_NAMES,
    examples_per_task_per_length: int = 1_000,
) -> list[dict[str, Any]]:
    if not 1 <= examples_per_task_per_length <= 1_000:
        raise ValueError("IID examples per task per length must be between 1 and 1000")
    unknown = set(tasks) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"unknown T16 tasks: {sorted(unknown)}")
    selected = set(tasks)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    factorial: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for raw in _manifest_records(manifest_path, 99):
        task = str(raw.get("task"))
        if task not in selected:
            continue
        if task == "factorial":
            primary = int(raw["inputs"]["primary"])
            if primary in FACTORIAL_IID_DOMAIN and primary not in factorial:
                record = dict(raw)
                record["evaluation_bucket"] = "0_100"
                record["evaluation_regime"] = "factorial_seen_domain"
                record["sampling_stratum"] = _iid_stratum(record)
                factorial[primary] = record
            continue
        bucket = int(raw.get("sampling_bucket", raw.get("n_digits")))
        key = (task, bucket)
        if bucket not in range(1, 6) or counts[key] >= examples_per_task_per_length:
            continue
        record = dict(raw)
        record["evaluation_bucket"] = str(bucket)
        record["evaluation_regime"] = "iid"
        record["sampling_stratum"] = _iid_stratum(record)
        records.append(record)
        counts[key] += 1

    missing = {
        (task, bucket): examples_per_task_per_length - counts[(task, bucket)]
        for task in tasks if task != "factorial"
        for bucket in range(1, 6)
        if counts[(task, bucket)] != examples_per_task_per_length
    }
    if missing:
        raise ValueError(f"IID test shard is missing balanced groups: {missing}")
    if "factorial" in selected:
        missing_factorials = set(FACTORIAL_IID_DOMAIN) - set(factorial)
        if missing_factorials:
            raise ValueError(f"IID test shard is missing factorial inputs: {sorted(missing_factorials)}")
        records.extend(factorial[value] for value in FACTORIAL_IID_DOMAIN)
    return records


def _metric_groups() -> dict[str, base_evaluate.MetricAccumulator]:
    return defaultdict(base_evaluate.MetricAccumulator)


def evaluate_records(
    model: torch.nn.Module,
    records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    max_seq_len: int,
    progress_every: int = 1_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups = _metric_groups()
    mistakes: list[dict[str, Any]] = []
    model.eval()
    processed = 0
    with _t16_evaluation_runtime():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompts: list[tuple[str, ...]] = []
            expected_rows: list[tuple[str, ...]] = []
            for record in batch:
                prompt, expected = base_evaluate.split_prompt_and_answer(record["tokens"])
                prompts.append(prompt)
                expected_rows.append(expected)
            teacher = base_evaluate.teacher_forced_batch(model, batch, device=device)
            generated = base_evaluate.greedy_generate_batch(
                model, prompts, device=device, max_new_tokens=max_new_tokens,
                max_seq_len=max_seq_len,
            )
            for record, expected, prediction, teacher_row in zip(
                batch, expected_rows, generated, teacher, strict=True
            ):
                task = str(record["task"])
                n_digits = int(record["n_digits"])
                bucket = str(record["evaluation_bucket"])
                regime = str(record["evaluation_regime"])
                stratum = str(record.get("sampling_stratum", "unspecified"))
                keys = (
                    "overall",
                    f"task::{task}",
                    f"length::{n_digits}",
                    f"task_length::{task}::{n_digits}",
                    f"bucket::{bucket}",
                    f"task_bucket::{task}::{bucket}",
                    f"regime::{regime}",
                    f"task_regime::{task}::{regime}",
                    f"task_stratum::{task}::{stratum}",
                )
                update = {
                    "expected_tokens": expected,
                    "generated_tokens": prediction,
                    "expected_value": int(record["answer"]),
                    "teacher_loss_sum": float(teacher_row["loss_sum"]),
                    "teacher_tokens": int(teacher_row["tokens"]),
                    "teacher_token_correct": int(teacher_row["token_correct"]),
                    "teacher_sequence_correct": bool(teacher_row["sequence_correct"]),
                }
                for key in keys:
                    groups[key].update(**update)
                parsed = base_evaluate.parse_generated_answer(prediction)
                if parsed != int(record["answer"]) and len(mistakes) < 20:
                    mistakes.append({
                        "id": record["id"], "task": task,
                        "n_digits": n_digits, "evaluation_bucket": bucket,
                        "evaluation_regime": regime, "sampling_stratum": stratum,
                        "inputs": record["inputs"], "expected": list(expected),
                        "generated": list(prediction), "parsed_generated": parsed,
                    })
            processed += len(batch)
            if progress_every > 0 and (
                processed == len(records)
                or processed // progress_every != (processed - len(batch)) // progress_every
            ):
                print(f"Evaluated {processed:,}/{len(records):,} examples", flush=True)

        by_task: dict[str, Any] = {}
        by_length: dict[str, Any] = {}
        by_task_length: dict[str, dict[str, Any]] = defaultdict(dict)
        by_bucket: dict[str, Any] = {}
        by_task_bucket: dict[str, dict[str, Any]] = defaultdict(dict)
        by_regime: dict[str, Any] = {}
        by_task_regime: dict[str, dict[str, Any]] = defaultdict(dict)
        by_task_stratum: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, accumulator in groups.items():
            if key == "overall":
                continue
            parts = key.split("::")
            destination = parts[0]
            if destination == "task": by_task[parts[1]] = accumulator.summary()
            elif destination == "length": by_length[parts[1]] = accumulator.summary()
            elif destination == "task_length": by_task_length[parts[1]][parts[2]] = accumulator.summary()
            elif destination == "bucket": by_bucket[parts[1]] = accumulator.summary()
            elif destination == "task_bucket": by_task_bucket[parts[1]][parts[2]] = accumulator.summary()
            elif destination == "regime": by_regime[parts[1]] = accumulator.summary()
            elif destination == "task_regime": by_task_regime[parts[1]][parts[2]] = accumulator.summary()
            elif destination == "task_stratum": by_task_stratum[parts[1]][parts[2]] = accumulator.summary()
            else: raise AssertionError(f"unknown metric group: {key}")

        macro_exact = sum(
            row.summary()["autoregressive_exact_accuracy"]
            for key, row in groups.items() if key.startswith("task::")
        ) / max(1, len(by_task))
        return ({
            "overall": groups["overall"].summary(),
            "macro_task_exact_accuracy": macro_exact,
            "by_task": dict(sorted(by_task.items())),
            "by_actual_decimal_length": dict(sorted(by_length.items(), key=lambda item: int(item[0]))),
            "by_task_and_actual_decimal_length": {
                task: dict(sorted(rows.items(), key=lambda item: int(item[0])))
                for task, rows in sorted(by_task_length.items())
            },
            "by_evaluation_bucket": dict(sorted(by_bucket.items())),
            "by_task_and_evaluation_bucket": {
                task: dict(sorted(rows.items())) for task, rows in sorted(by_task_bucket.items())
            },
            "by_generalization_regime": dict(sorted(by_regime.items())),
            "by_task_and_generalization_regime": {
                task: dict(sorted(rows.items())) for task, rows in sorted(by_task_regime.items())
            },
            "by_task_and_sampling_stratum": {
                task: dict(sorted(rows.items())) for task, rows in sorted(by_task_stratum.items())
            },
        }, mistakes)


def _selected_completed_runs(
    config_path: Path, only: Iterable[str]
) -> tuple[ProductionRun, ...]:
    runs = build_matrix(config_path)
    lookup = {run.run_id: run for run in runs}
    requested = tuple(only)
    if requested:
        unknown = sorted(set(requested) - set(lookup))
        if unknown:
            raise ValueError(f"unknown run IDs: {', '.join(unknown)}")
        candidates = tuple(run for run in runs if run.run_id in set(requested))
    else:
        candidates = runs
    completed = tuple(
        run for run in candidates if completion_is_valid(run, config_path=config_path)
    )
    if len(completed) != len(candidates):
        incomplete = sorted({run.run_id for run in candidates} - {run.run_id for run in completed})
        raise ValueError(f"runs are not verified complete: {', '.join(incomplete)}")
    return completed


def _result_is_current(
    path: Path, *, checkpoint_sha256: str, data_sha256: str,
    examples_per_group: int, max_new_tokens: int, tasks: Sequence[str],
) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            value.get("checkpoint_sha256") == checkpoint_sha256
            and value.get("format_version") == FORMAT_VERSION
            and value.get("evaluation_data_sha256") == data_sha256
            and value.get("examples_per_task_per_length") == examples_per_group
            and value.get("max_new_tokens") == max_new_tokens
            and value.get("evaluated_tasks") == list(tasks)
            and isinstance(value.get("metrics"), dict)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _print_summary(metrics: Mapping[str, Any]) -> None:
    print(f"Macro task exact accuracy: {metrics['macro_task_exact_accuracy']:.2%}")
    regimes = metrics["by_task_and_generalization_regime"]
    for task, row in metrics["by_task"].items():
        baseline = row["most_common_target_baseline"]
        print(
            f"  {task}: exact={row['autoregressive_exact_accuracy']:.2%} "
            f"token={row['autoregressive_token_accuracy']:.2%} "
            f"well_formed={row['well_formed_rate']:.2%} "
            f"baseline={baseline['accuracy']:.2%} ({baseline['value']})"
        )
        for regime, regime_row in regimes.get(task, {}).items():
            print(
                f"    {regime}: n={regime_row['examples']} "
                f"exact={regime_row['autoregressive_exact_accuracy']:.2%} "
                f"well_formed={regime_row['well_formed_rate']:.2%}"
            )


def evaluate_run_split(
    run: ProductionRun, *, split: str, iid_manifest: Path,
    config_path: Path, output_dir: Path, selected_tasks: Sequence[str],
    device_name: str, batch_size: int, max_new_tokens: int,
    examples_per_task_per_length: int, progress_every: int, force: bool,
) -> Path:
    checkpoint_path = Path(run.output_dir) / "checkpoint.pt"
    checkpoint_sha256 = _sha256(checkpoint_path)
    tasks = tuple(task for task in run.tasks if task in set(selected_tasks))
    if split == "iid":
        records = load_iid_records(
            iid_manifest, tasks=tasks,
            examples_per_task_per_length=examples_per_task_per_length,
        )
        data_sha256 = _sha256(iid_manifest)
        data_description = f"{iid_manifest} shard 099"
    else:
        records = build_ood_records(
            tasks=tasks,
            examples_per_task_per_length=examples_per_task_per_length,
        )
        data_sha256 = _records_sha256(records)
        data_description = "deterministic in-memory T16 OOD protocol"
    task_suffix = "" if tasks == run.tasks else "-tasks-" + "_".join(tasks)
    destination = output_dir / f"{run.run_id}-{split}{task_suffix}.json"
    if not force and _result_is_current(
        destination, checkpoint_sha256=checkpoint_sha256,
        data_sha256=data_sha256, examples_per_group=examples_per_task_per_length,
        max_new_tokens=max_new_tokens, tasks=tasks,
    ):
        print(f"Reusing current evaluation: {destination}")
        return destination

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_config = TrainConfig.from_value(checkpoint["config"])
    model = build_t16_model(train_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    device = torch.device(device_name)
    model.to(device)
    print(f"Evaluating {run.run_id} on {split}: {len(records):,} examples")
    metrics, mistakes = evaluate_records(
        model, records, device=device, batch_size=batch_size,
        max_new_tokens=max_new_tokens, max_seq_len=train_config.max_seq_len,
        progress_every=progress_every,
    )
    result = {
        "format_version": FORMAT_VERSION,
        "run": asdict(run), "split": split,
        "evaluated_tasks": list(tasks),
        "examples_per_task_per_length": examples_per_task_per_length,
        "factorial_iid_examples": len(FACTORIAL_IID_DOMAIN) if split == "iid" and "factorial" in tasks else 0,
        "factorial_ood_examples": 20 if split == "ood" and "factorial" in tasks else 0,
        "max_new_tokens": max_new_tokens,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_data": data_description,
        "evaluation_data_sha256": data_sha256,
        "metrics": metrics, "first_mistakes": mistakes,
        "definitions": {
            "iid": "held-out shard 099, training sampling buckets 1-5",
            "magnitude_ood_familiar_token_width": "six decimal digits; three base-100 input tokens, matching the maximum training token width",
            "token_length_ood": "seven-to-ten decimal digits; more base-100 input tokens than training",
            "factorial_seen_domain": "unique factorial inputs 0-100; not semantically held out because the finite domain appears during training",
            "factorial_value_ood": "unique unseen factorial inputs 101-120",
            "most_common_target_baseline": "always predict the most frequent target inside the reported group",
        },
    }
    _atomic_json(result, destination)
    print(f"Saved: {destination}")
    _print_summary(metrics)
    model.to("cpu")
    del model
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--split", choices=("iid", "ood", "both"), default="both")
    parser.add_argument("--iid-manifest", type=Path, default=DEFAULT_IID_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--examples-per-task-per-length", type=int, default=1_000)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--device", default=automatic_device())
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = tuple(args.task) if args.task else TASK_NAMES
    unknown = set(tasks) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"unknown T16 tasks: {sorted(unknown)}")
    runs = _selected_completed_runs(args.config, args.only)
    for run in runs:
        for split in (("iid", "ood") if args.split == "both" else (args.split,)):
            evaluate_run_split(
                run, split=split, iid_manifest=args.iid_manifest,
                config_path=args.config, output_dir=args.output_dir,
                selected_tasks=tasks, device_name=args.device,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                examples_per_task_per_length=args.examples_per_task_per_length,
                progress_every=args.progress_every, force=args.force,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluate_records", "load_iid_records", "main",
]
