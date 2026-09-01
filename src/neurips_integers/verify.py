"""Full mathematical and encoding verification for the integer pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .generate import SUPPORTED_SCHEMA_VERSIONS, TASK_NAMES
from . import math_ops as ops
from .passage import TASK_SPECS, VOCABULARY, passage_tokens


class VerificationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise VerificationError(message)


def _decimal_digits(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{name} must be a nonnegative integer")
    return len(str(value))


def _input_size(inputs: Mapping[str, object]) -> int:
    if "values" in inputs:
        values = inputs["values"]
        if not isinstance(values, list) or not values:
            _fail("values must be a nonempty list")
        return max(
            _decimal_digits(value, name=f"values[{index}]")
            for index, value in enumerate(values)
        )
    widths = [_decimal_digits(inputs.get("primary"), name="primary")]
    if "operand" in inputs:
        widths.append(_decimal_digits(inputs["operand"], name="operand"))
    return max(widths)


def _mathematical_answer(task: str, inputs: Mapping[str, object]) -> int:
    """Independently recompute the target rather than calling the generator."""

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


def verify_record(record: object, expected_id: int, *, schema_version: str) -> str:
    if not isinstance(record, dict):
        _fail("record must be an object")
    task = TASK_NAMES[expected_id % len(TASK_NAMES)]
    if record.get("schema_version") != schema_version:
        _fail("wrong schema version")
    if record.get("id") != expected_id or record.get("task") != task:
        _fail("record id/task schedule mismatch")
    digits = record.get("n_digits")
    if isinstance(digits, bool) or not isinstance(digits, int) or digits < 1:
        _fail("invalid n_digits")
    inputs = record.get("inputs")
    if not isinstance(inputs, dict):
        _fail("inputs must be an object")
    if digits != _input_size(inputs):
        _fail("n_digits does not match the longest input integer")
    try:
        truth = _mathematical_answer(task, inputs)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"cannot recompute answer: {exc}") from exc
    if record.get("answer") != truth:
        _fail("stored answer disagrees with recomputed answer")
    if record.get("answer_kind") != TASK_SPECS[task].answer_kind:
        _fail("wrong answer kind")
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(x, str) for x in tokens):
        _fail("tokens must be strings")
    if any(token not in VOCABULARY for token in tokens):
        _fail("token outside integer vocabulary")
    try:
        expected_tokens = passage_tokens(task, truth, size=digits, **inputs)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"cannot rerender record: {exc}") from exc
    if tokens != list(expected_tokens):
        _fail("tokens are not canonical")
    if record.get("canonical_text") != " ".join(expected_tokens):
        _fail("canonical_text does not match tokens")
    return task


def verify(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        _fail("manifest uses an unsupported integer schema")
    if manifest.get("tasks") != list(TASK_NAMES):
        _fail("manifest task order mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != 100:
        _fail("manifest must contain exactly 100 shards")
    expected_id = 0
    counts: Counter[str] = Counter()
    for entry in shards:
        if not isinstance(entry, Mapping):
            _fail("invalid shard entry")
        path = directory / str(entry["filename"])
        if path.stat().st_size != entry["byte_size"] or _sha256(path) != entry["sha256"]:
            _fail(f"hash/size mismatch for {path.name}")
        shard_counts: Counter[str] = Counter()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                task = verify_record(
                    json.loads(line), expected_id, schema_version=schema_version
                )
                counts[task] += 1
                shard_counts[task] += 1
                expected_id += 1
        if dict(entry["task_counts"]) != {task: shard_counts[task] for task in TASK_NAMES}:
            _fail(f"task counts mismatch for {path.name}")
    if expected_id != manifest.get("count"):
        _fail("record count mismatch")
    expected_counts = {task: counts[task] for task in TASK_NAMES}
    if manifest.get("task_counts") != expected_counts:
        _fail("global task counts mismatch")
    return {"records": expected_id, "task_counts": expected_counts, "status": "passed"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path("data/integer-20k-pilot"))
    args = parser.parse_args(argv)
    result = verify(args.directory)
    print(f"Full verification passed: {result['records']:,} records")
    print(f"Task counts: {result['task_counts']}")


if __name__ == "__main__":
    main()
