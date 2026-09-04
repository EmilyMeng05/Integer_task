"""Deterministic pilot generation for the sixteen-task integer corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import math_ops as ops
from .generate import (
    PILOT_SCHEMA_VERSION as LEGACY_PILOT_SCHEMA_VERSION,
    TASK_NAMES as LEGACY_TASK_ORDER,
    build_record as build_legacy_record,
)
from .passage_t16 import NEW_TASK_SPECS, TASK_SPECS, passage_tokens
from .sampling_protocol import (
    DEFAULT_SEED as NEW_SAMPLING_SEED,
    SAMPLING_PROTOCOL_VERSION,
    answer as new_task_answer,
    sample_inputs,
    validate_sample,
)


PILOT_SCHEMA_VERSION = "integer-16/v1-pilot"
PRODUCTION_SCHEMA_VERSION = "integer-16/v1"
SUPPORTED_SCHEMA_VERSIONS = (PILOT_SCHEMA_VERSION, PRODUCTION_SCHEMA_VERSION)
TASK_ORDER_SEED = 20_260_830
LEGACY_SAMPLING_SEED = 20_260_830
TASK_NAMES: tuple[str, ...] = (
    "decimal_digit_sum",
    "greatest_common_divisor",
    "multiplication",
    "greater_than",
    "integer_list_sum",
    "modulo",
    "addition",
    "successor",
    "subtraction",
    "integer_division",
    "number_of_decimal_digits",
    "reverse_decimal_digits",
    "decimal_digit_occurrence_count",
    "even_odd",
    "divisibility",
    "factorial",
)
DEFAULT_COUNT = 16_000
DEFAULT_SHARD_SIZE = 160
DEFAULT_MIN_DIGITS = 1
DEFAULT_MAX_DIGITS = 5
DEFAULT_OUTPUT_DIR = Path("data/integer-t16-16k-pilot")

if tuple(TASK_NAMES[:8]) != tuple(LEGACY_TASK_ORDER):
    raise RuntimeError("T16 must preserve the frozen eight-task order")
if set(TASK_NAMES) != set(ops.T16_TASK_NAMES):
    raise RuntimeError("T16 generator and mathematical task registries disagree")
if set(TASK_NAMES[8:]) != set(NEW_TASK_SPECS):
    raise RuntimeError("T16 generator and new Passage task registries disagree")


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _input_size(inputs: Mapping[str, object]) -> int:
    values: list[int] = []
    if "values" in inputs:
        raw = inputs["values"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("values must be a nonempty sequence")
        values.extend(int(value) for value in raw)
    for name in ("primary", "operand", "digit"):
        if name in inputs:
            values.append(int(inputs[name]))
    if not values or any(value < 0 for value in values):
        raise ValueError("inputs must contain nonnegative integers")
    return max(len(str(value)) for value in values)


def build_record(
    record_id: int,
    *,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    legacy_seed: int = LEGACY_SAMPLING_SEED,
    new_seed: int = NEW_SAMPLING_SEED,
    schema_version: str = PILOT_SCHEMA_VERSION,
) -> dict[str, Any]:
    if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id < 0:
        raise ValueError("record_id must be a nonnegative integer")
    if not 1 <= min_digits <= max_digits:
        raise ValueError("digit range must satisfy 1 <= min_digits <= max_digits")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported T16 schema: {schema_version!r}")

    task_index = record_id % len(TASK_NAMES)
    task = TASK_NAMES[task_index]
    occurrence = record_id // len(TASK_NAMES)
    width_count = max_digits - min_digits + 1
    sampling_bucket = min_digits + occurrence % width_count

    if task_index < 8:
        legacy_id = occurrence * len(LEGACY_TASK_ORDER) + task_index
        legacy = build_legacy_record(
            legacy_id,
            min_digits=min_digits,
            max_digits=max_digits,
            seed=legacy_seed,
            schema_version=LEGACY_PILOT_SCHEMA_VERSION,
        )
        inputs = dict(legacy["inputs"])
        result = int(legacy["answer"])
        stratum = "legacy_random"
        sampling_version = "integer-8/v1"
    else:
        sample = sample_inputs(task, sampling_bucket, occurrence, seed=new_seed)
        validate_sample(task, sample)
        inputs = dict(sample["inputs"])
        result = new_task_answer(task, inputs)
        stratum = str(sample["stratum"])
        sampling_version = SAMPLING_PROTOCOL_VERSION

    n_digits = _input_size(inputs)
    tokens = passage_tokens(task, result, size=n_digits, **inputs)
    spec = TASK_SPECS[task]
    answer_kind = getattr(spec, "answer_kind")
    return {
        "schema_version": schema_version,
        "id": record_id,
        "task": task,
        "n_digits": n_digits,
        "sampling_bucket": sampling_bucket,
        "sampling_protocol_version": sampling_version,
        "sampling_stratum": stratum,
        "inputs": _json_value(inputs),
        "answer": result,
        "answer_kind": answer_kind,
        "tokens": list(tokens),
        "canonical_text": " ".join(tokens),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    count: int = DEFAULT_COUNT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    legacy_seed: int = LEGACY_SAMPLING_SEED,
    new_seed: int = NEW_SAMPLING_SEED,
    schema_version: str = PILOT_SCHEMA_VERSION,
) -> dict[str, Any]:
    cells = len(TASK_NAMES) * (max_digits - min_digits + 1)
    if count <= 0 or count % cells:
        raise ValueError("count must balance every task-by-sampling-bucket cell")
    if shard_size <= 0 or count % shard_size:
        raise ValueError("shard_size must be positive and divide count")
    if count // shard_size != 100:
        raise ValueError("T16 data must use exactly 100 shards")

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    stratum_counts: dict[str, Counter[str]] = {
        task: Counter() for task in TASK_NAMES
    }
    for shard_index in range(100):
        first_id = shard_index * shard_size
        path = output_dir / f"part-{shard_index:05d}.jsonl.gz"
        shard_tasks: Counter[str] = Counter()
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                for record_id in range(first_id, first_id + shard_size):
                    record = build_record(
                        record_id,
                        min_digits=min_digits,
                        max_digits=max_digits,
                        legacy_seed=legacy_seed,
                        new_seed=new_seed,
                        schema_version=schema_version,
                    )
                    task = record["task"]
                    shard_tasks[task] += 1
                    task_counts[task] += 1
                    stratum_counts[task][record["sampling_stratum"]] += 1
                    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
                    handle.write(line.encode("utf-8") + b"\n")
        entries.append({
            "index": shard_index,
            "filename": path.name,
            "first_id": first_id,
            "last_id": first_id + shard_size - 1,
            "record_count": shard_size,
            "byte_size": path.stat().st_size,
            "sha256": _sha256(path),
            "task_counts": {task: shard_tasks[task] for task in TASK_NAMES},
        })

    manifest = {
        "schema_version": schema_version,
        "count": count,
        "base": 100,
        "min_sampling_bucket": min_digits,
        "max_sampling_bucket": max_digits,
        "legacy_sampling_seed": legacy_seed,
        "new_sampling_seed": new_seed,
        "task_order_seed": TASK_ORDER_SEED,
        "tasks": list(TASK_NAMES),
        "records_per_task": count // len(TASK_NAMES),
        "records_per_task_per_sampling_bucket": count // cells,
        "factorial_input_domain": [0, 100],
        "factorial_length_ood_domain": [101, 120],
        "shard_size": shard_size,
        "shard_count": 100,
        "splits": {"train": [0, 97], "validation": [98, 98], "test": [99, 99]},
        "task_counts": {task: task_counts[task] for task in TASK_NAMES},
        "sampling_stratum_counts": {
            task: dict(sorted(stratum_counts[task].items()))
            for task in TASK_NAMES
        },
        "shards": entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--min-digits", type=int, default=DEFAULT_MIN_DIGITS)
    parser.add_argument("--max-digits", type=int, default=DEFAULT_MAX_DIGITS)
    parser.add_argument("--legacy-seed", type=int, default=LEGACY_SAMPLING_SEED)
    parser.add_argument("--new-seed", type=int, default=NEW_SAMPLING_SEED)
    args = parser.parse_args(argv)
    result = generate(
        args.output_dir,
        count=args.count,
        shard_size=args.shard_size,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        legacy_seed=args.legacy_seed,
        new_seed=args.new_seed,
    )
    print(f"Generated {result['count']:,} T16 records in 100 shards")
    print(f"Task counts: {result['task_counts']}")
    print(f"Saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
