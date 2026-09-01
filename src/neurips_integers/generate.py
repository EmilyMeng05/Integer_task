"""Deterministic generation of the eight-task integer pilot corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

from . import math_ops as ops
from .passage import TASK_SPECS, passage_tokens


PILOT_SCHEMA_VERSION = "integer-8/v1-pilot"
PRODUCTION_SCHEMA_VERSION = "integer-8/v1"
OOD_SCHEMA_VERSION = "integer-8/v1-length-ood"
SUPPORTED_SCHEMA_VERSIONS = (
    PILOT_SCHEMA_VERSION,
    PRODUCTION_SCHEMA_VERSION,
    OOD_SCHEMA_VERSION,
)
SCHEMA_VERSION = PILOT_SCHEMA_VERSION
TASK_ORDER_SEED = 20_260_830
TASK_NAMES: tuple[str, ...] = (
    "decimal_digit_sum",
    "greatest_common_divisor",
    "multiplication",
    "greater_than",
    "integer_list_sum",
    "modulo",
    "addition",
    "successor",
)
DEFAULT_COUNT = 20_000
DEFAULT_MIN_DIGITS = 1
DEFAULT_MAX_DIGITS = 5
DEFAULT_SEED = 20_260_830
DEFAULT_SHARD_SIZE = 200
DEFAULT_OUTPUT_DIR = Path("data/integer-20k-pilot")
_MASK_64 = (1 << 64) - 1

if set(TASK_NAMES) != set(ops.TRAINING_TASKS):
    raise RuntimeError("frozen Phase 1 order does not match training tasks")


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return value ^ (value >> 31)


def _record_rng(seed: int, record_id: int) -> Random:
    return Random(_splitmix64(seed ^ _splitmix64(record_id)))


def _exact_width_integer(digits: int, rng: Random) -> int:
    """Sample an integer with exactly ``digits`` ordinary decimal digits."""

    if digits == 1:
        return rng.randrange(0, 10)
    return rng.randrange(10 ** (digits - 1), 10**digits)


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _answer(task: str, inputs: Mapping[str, object]) -> int:
    primary = inputs.get("primary")
    operand = inputs.get("operand")
    values = inputs.get("values")
    if task == "successor":
        return ops.successor(primary)  # type: ignore[arg-type]
    if task == "addition":
        return ops.addition(primary, operand)  # type: ignore[arg-type]
    if task == "multiplication":
        return ops.multiplication(primary, operand)  # type: ignore[arg-type]
    if task == "modulo":
        return ops.modulo(primary, operand)  # type: ignore[arg-type]
    if task == "greater_than":
        return int(ops.greater_than(primary, operand))  # type: ignore[arg-type]
    if task == "decimal_digit_sum":
        return ops.decimal_digit_sum(primary)  # type: ignore[arg-type]
    if task == "greatest_common_divisor":
        return ops.greatest_common_divisor(primary, operand)  # type: ignore[arg-type]
    if task == "integer_list_sum":
        return ops.integer_list_sum(values)  # type: ignore[arg-type]
    raise AssertionError(f"unhandled task {task!r}")


def build_record(
    record_id: int,
    *,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    seed: int = DEFAULT_SEED,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id < 0:
        raise ValueError("record_id must be a nonnegative integer")
    if not 1 <= min_digits <= max_digits:
        raise ValueError("digit range must satisfy 1 <= min_digits <= max_digits")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported integer schema: {schema_version!r}")

    task = TASK_NAMES[record_id % len(TASK_NAMES)]
    occurrence = record_id // len(TASK_NAMES)
    width_count = max_digits - min_digits + 1
    digits = min_digits + (occurrence % width_count)
    rng = _record_rng(seed, record_id)
    inputs: dict[str, object]

    if task == "integer_list_sum":
        length = 2 + rng.randrange(7)
        inputs = {
            "values": tuple(_exact_width_integer(digits, rng) for _ in range(length))
        }
    elif task in {"addition", "multiplication", "modulo", "greatest_common_divisor"}:
        primary = _exact_width_integer(digits, rng)
        operand = _exact_width_integer(digits, rng)
        if task == "modulo" and operand == 0:
            operand = 1
        if task == "greatest_common_divisor" and primary == operand == 0:
            operand = 1
        inputs = {"primary": primary, "operand": operand}
    elif task == "greater_than":
        left = _exact_width_integer(digits, rng)
        right = _exact_width_integer(digits, rng)
        if left == right:
            right = right - 1 if right else 1
        should_be_true = occurrence % 2 == 0
        high, low = max(left, right), min(left, right)
        inputs = (
            {"primary": high, "operand": low}
            if should_be_true
            else {"primary": low, "operand": high}
        )
    else:
        inputs = {"primary": _exact_width_integer(digits, rng)}

    answer = _answer(task, inputs)
    render_kwargs = dict(inputs)
    tokens = passage_tokens(task, answer, size=digits, **render_kwargs)  # type: ignore[arg-type]
    return {
        "schema_version": schema_version,
        "id": record_id,
        "task": task,
        "n_digits": digits,
        "inputs": _json_value(inputs),
        "answer": answer,
        "answer_kind": TASK_SPECS[task].answer_kind,
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
    seed: int = DEFAULT_SEED,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    if count <= 0 or count % len(TASK_NAMES):
        raise ValueError("count must be positive and divisible by eight")
    if shard_size <= 0 or count % shard_size:
        raise ValueError("shard_size must be positive and divide count")
    if count // shard_size != 100:
        raise ValueError("pilot must use exactly 100 shards for a 98/1/1 split")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_entries: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    shard_count = count // shard_size

    for shard_index in range(shard_count):
        start = shard_index * shard_size
        filename = f"part-{shard_index:05d}.jsonl.gz"
        path = output_dir / filename
        counts: Counter[str] = Counter()
        with path.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as handle:
                for record_id in range(start, start + shard_size):
                    record = build_record(
                        record_id,
                        min_digits=min_digits,
                        max_digits=max_digits,
                        seed=seed,
                        schema_version=schema_version,
                    )
                    counts[record["task"]] += 1
                    line = json.dumps(
                        record, sort_keys=True, separators=(",", ":")
                    )
                    handle.write(line.encode("utf-8") + b"\n")
        total_counts.update(counts)
        shard_entries.append(
            {
                "index": shard_index,
                "filename": filename,
                "first_id": start,
                "last_id": start + shard_size - 1,
                "record_count": shard_size,
                "byte_size": path.stat().st_size,
                "sha256": _sha256(path),
                "task_counts": {task: counts[task] for task in TASK_NAMES},
            }
        )

    manifest = {
        "schema_version": schema_version,
        "count": count,
        "base": 100,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "seed": seed,
        "task_order_seed": TASK_ORDER_SEED,
        "tasks": list(TASK_NAMES),
        "shard_size": shard_size,
        "shard_count": shard_count,
        "splits": {"train": [0, 97], "validation": [98, 98], "test": [99, 99]},
        "task_counts": {task: total_counts[task] for task in TASK_NAMES},
        "shards": shard_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--min-digits", type=int, default=DEFAULT_MIN_DIGITS)
    parser.add_argument("--max-digits", type=int, default=DEFAULT_MAX_DIGITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    manifest = generate(
        args.output_dir,
        count=args.count,
        shard_size=args.shard_size,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        seed=args.seed,
    )
    print(f"Generated {manifest['count']:,} records in {manifest['shard_count']} shards")
    print(f"Task counts: {manifest['task_counts']}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
