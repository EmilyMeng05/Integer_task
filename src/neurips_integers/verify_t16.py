"""Full mathematical, sampling, and encoding verification for T16 data."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from . import math_ops as ops
from .generate_t16 import SUPPORTED_SCHEMA_VERSIONS, TASK_NAMES
from .passage_t16 import NEW_TASK_SPECS, TASK_SPECS, VOCABULARY, passage_tokens
from .sampling_protocol import SAMPLING_PROTOCOL_VERSION, validate_sample


class VerificationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_size(inputs: Mapping[str, object]) -> int:
    integers: list[int] = []
    if "values" in inputs:
        values = inputs["values"]
        if not isinstance(values, list) or not values:
            _fail("values must be a nonempty list")
        integers.extend(values)  # type: ignore[arg-type]
    for name in ("primary", "operand", "digit"):
        if name in inputs:
            value = inputs[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"{name} must be a nonnegative integer")
            integers.append(value)
    if not integers or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers):
        _fail("all inputs must be nonnegative integers")
    return max(len(str(value)) for value in integers)


def _answer(task: str, inputs: Mapping[str, object]) -> int:
    primary = inputs.get("primary")
    operand = inputs.get("operand")
    digit = inputs.get("digit")
    values = inputs.get("values")
    functions = {
        "successor": lambda: ops.successor(primary),
        "addition": lambda: ops.addition(primary, operand),
        "multiplication": lambda: ops.multiplication(primary, operand),
        "modulo": lambda: ops.modulo(primary, operand),
        "greater_than": lambda: int(ops.greater_than(primary, operand)),
        "decimal_digit_sum": lambda: ops.decimal_digit_sum(primary),
        "greatest_common_divisor": lambda: ops.greatest_common_divisor(primary, operand),
        "integer_list_sum": lambda: ops.integer_list_sum(values),
        "subtraction": lambda: ops.subtraction(primary, operand),
        "integer_division": lambda: ops.integer_division(primary, operand),
        "number_of_decimal_digits": lambda: ops.number_of_decimal_digits(primary),
        "reverse_decimal_digits": lambda: ops.reverse_decimal_digits(primary),
        "decimal_digit_occurrence_count": lambda: ops.decimal_digit_occurrence_count(primary, digit),
        "even_odd": lambda: ops.even_odd(primary),
        "divisibility": lambda: ops.divisibility(primary, operand),
        "factorial": lambda: ops.factorial(primary),
    }
    return int(functions[task]())  # type: ignore[arg-type]


def verify_record(record: object, expected_id: int, *, schema_version: str) -> str:
    if not isinstance(record, dict):
        _fail("record must be an object")
    task = TASK_NAMES[expected_id % len(TASK_NAMES)]
    if record.get("schema_version") != schema_version:
        _fail("wrong schema version")
    if record.get("id") != expected_id or record.get("task") != task:
        _fail("record id/task schedule mismatch")
    inputs = record.get("inputs")
    if not isinstance(inputs, dict):
        _fail("inputs must be an object")
    n_digits = _input_size(inputs)
    if record.get("n_digits") != n_digits:
        _fail("n_digits does not match the longest input integer")
    bucket = record.get("sampling_bucket")
    if isinstance(bucket, bool) or not isinstance(bucket, int) or not 1 <= bucket <= 5:
        _fail("invalid sampling bucket")
    if task != "factorial" and bucket != n_digits:
        _fail("non-factorial sampling bucket must equal n_digits")

    try:
        truth = _answer(task, inputs)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"cannot recompute answer: {exc}") from exc
    if record.get("answer") != truth:
        _fail("stored answer disagrees with recomputed answer")
    spec = TASK_SPECS[task]
    if record.get("answer_kind") != getattr(spec, "answer_kind"):
        _fail("wrong answer kind")

    is_new = task in NEW_TASK_SPECS
    expected_protocol = SAMPLING_PROTOCOL_VERSION if is_new else "integer-8/v1"
    if record.get("sampling_protocol_version") != expected_protocol:
        _fail("wrong sampling protocol version")
    stratum = record.get("sampling_stratum")
    if not isinstance(stratum, str):
        _fail("missing sampling stratum")
    if is_new:
        try:
            validate_sample(task, {"inputs": inputs, "stratum": stratum})
        except (AssertionError, TypeError, ValueError) as exc:
            raise VerificationError(f"invalid task-aware stratum: {exc}") from exc
    elif stratum != "legacy_random":
        _fail("legacy task must use legacy_random stratum")

    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        _fail("tokens must be strings")
    if any(token not in VOCABULARY for token in tokens):
        _fail("token outside T16 vocabulary")
    try:
        expected_tokens = passage_tokens(task, truth, size=n_digits, **inputs)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"cannot rerender record: {exc}") from exc
    if tokens != list(expected_tokens):
        _fail("tokens are not canonical")
    if record.get("canonical_text") != " ".join(expected_tokens):
        _fail("canonical_text does not match tokens")
    return task


def verify(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        _fail("manifest uses an unsupported T16 schema")
    if manifest.get("tasks") != list(TASK_NAMES):
        _fail("manifest task order mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != 100:
        _fail("manifest must contain exactly 100 shards")

    expected_id = 0
    task_counts: Counter[str] = Counter()
    stratum_counts: dict[str, Counter[str]] = {
        task: Counter() for task in TASK_NAMES
    }
    for entry in shards:
        if not isinstance(entry, Mapping):
            _fail("invalid shard entry")
        path = directory / str(entry["filename"])
        if path.stat().st_size != entry["byte_size"] or _sha256(path) != entry["sha256"]:
            _fail(f"hash/size mismatch for {path.name}")
        shard_counts: Counter[str] = Counter()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                task = verify_record(record, expected_id, schema_version=str(schema_version))
                task_counts[task] += 1
                shard_counts[task] += 1
                stratum_counts[task][record["sampling_stratum"]] += 1
                expected_id += 1
        if dict(entry["task_counts"]) != {
            task: shard_counts[task] for task in TASK_NAMES
        }:
            _fail(f"task counts mismatch for {path.name}")

    if expected_id != manifest.get("count"):
        _fail("record count mismatch")
    expected_tasks = {task: task_counts[task] for task in TASK_NAMES}
    if manifest.get("task_counts") != expected_tasks:
        _fail("global task counts mismatch")
    expected_strata = {
        task: dict(sorted(stratum_counts[task].items()))
        for task in TASK_NAMES
    }
    if manifest.get("sampling_stratum_counts") != expected_strata:
        _fail("global sampling-stratum counts mismatch")
    return {
        "records": expected_id,
        "task_counts": expected_tasks,
        "sampling_stratum_counts": expected_strata,
        "status": "passed",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("data/integer-t16-16k-pilot"))
    args = parser.parse_args(argv)
    result = verify(args.directory)
    print(f"Full T16 verification passed: {result['records']:,} records")
    print(f"Task counts: {result['task_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
