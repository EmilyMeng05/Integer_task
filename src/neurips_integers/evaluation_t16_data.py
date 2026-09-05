"""Deterministic, in-memory evaluation records for the T16 checkpoints."""

from __future__ import annotations

import hashlib
import math
from random import Random
from typing import Any, Mapping, Sequence

from . import math_ops as ops
from .generate import (
    OOD_SCHEMA_VERSION,
    TASK_NAMES as LEGACY_TASKS,
    build_record as build_legacy_record,
)
from .generate_t16 import TASK_NAMES
from .passage_t16 import TASK_SPECS, passage_tokens


OOD_SEED = 20_260_905
OOD_LENGTHS = (6, 7, 8, 9, 10)
FACTORIAL_IID_DOMAIN = tuple(range(0, 101))
FACTORIAL_OOD_DOMAIN = tuple(range(101, 121))


def _rng(task: str, digits: int, occurrence: int) -> Random:
    payload = f"{OOD_SEED}:{task}:{digits}:{occurrence}".encode()
    return Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def _bounds(digits: int) -> tuple[int, int]:
    if digits not in OOD_LENGTHS:
        raise ValueError("OOD decimal length must be between 6 and 10")
    return 10 ** (digits - 1), 10**digits - 1


def _exact_width(digits: int, rng: Random) -> int:
    low, high = _bounds(digits)
    return rng.randint(low, high)


def decimal_carry_count(left: int, right: int) -> int:
    carries = 0
    incoming = 0
    while left or right or incoming:
        total = left % 10 + right % 10 + incoming
        incoming = int(total >= 10)
        carries += incoming
        left //= 10
        right //= 10
    return carries


def decimal_borrow_count(left: int, right: int) -> int:
    if right > left:
        raise ValueError("borrow count requires left >= right")
    borrows = 0
    incoming = 0
    while left or right or incoming:
        needs_borrow = left % 10 - incoming < right % 10
        incoming = int(needs_borrow)
        borrows += incoming
        left //= 10
        right //= 10
    return borrows


def legacy_diagnostic_stratum(record: Mapping[str, Any]) -> str:
    task = str(record["task"])
    inputs = record["inputs"]
    answer = int(record["answer"])
    if task == "addition":
        carries = decimal_carry_count(inputs["primary"], inputs["operand"])
        return "carry_0" if carries == 0 else ("carry_1" if carries == 1 else "carry_2_plus")
    if task == "successor":
        value = int(inputs["primary"])
        trailing_nines = len(str(value)) - len(str(value).rstrip("9"))
        return "no_cascade" if trailing_nines == 0 else "trailing_nine_cascade"
    if task == "greatest_common_divisor":
        return "target_1" if answer == 1 else "target_gt_1"
    if task == "multiplication":
        input_width = max(len(str(inputs["primary"])), len(str(inputs["operand"])))
        return "output_expands" if len(str(answer)) > input_width else "same_output_width"
    if task == "greater_than":
        return "true" if answer == 1 else "false"
    if task == "integer_list_sum":
        return f"list_length_{len(inputs['values'])}"
    if task == "modulo":
        return "remainder_0" if answer == 0 else "remainder_positive"
    if task == "decimal_digit_sum":
        return "sum_0_18" if answer <= 18 else ("sum_19_36" if answer <= 36 else "sum_37_plus")
    return "legacy_random"


def _new_inputs(task: str, digits: int, occurrence: int) -> tuple[dict[str, int], str]:
    rng = _rng(task, digits, occurrence)
    low, high = _bounds(digits)
    if task == "subtraction":
        category = "no_borrow" if occurrence % 2 == 0 else "borrow"
        if category == "no_borrow":
            left_digits = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(digits - 1)]
            right_digits = [rng.randint(0, value) for value in left_digits]
            return {
                "primary": int("".join(map(str, left_digits))),
                "operand": int("".join(map(str, right_digits))),
            }, category
        for _ in range(10_000):
            left = rng.randint(low, high)
            right = rng.randint(0, left)
            if decimal_borrow_count(left, right):
                return {"primary": left, "operand": right}, category
        raise RuntimeError("could not sample a subtraction borrow case")

    if task == "integer_division":
        category = ("quotient_zero", "quotient_one", "exact_multi", "remainder_multi")[occurrence % 4]
        if category == "quotient_zero":
            divisor = rng.randint(low, high)
            return {"primary": rng.randint(0, divisor - 1), "operand": divisor}, category
        if category == "quotient_one":
            value = rng.randint(low, high)
            return {"primary": value, "operand": value}, category
        for _ in range(10_000):
            quotient = rng.randint(2, 9)
            divisor_low = max(1, math.ceil(low / quotient))
            divisor_high = high // quotient
            if divisor_low > divisor_high:
                continue
            divisor = rng.randint(divisor_low, divisor_high)
            remainder = 0
            if category == "remainder_multi":
                maximum = min(divisor - 1, high - divisor * quotient)
                if maximum < 1:
                    continue
                remainder = rng.randint(1, maximum)
            dividend = divisor * quotient + remainder
            if low <= dividend <= high:
                return {"primary": dividend, "operand": divisor}, category
        raise RuntimeError("could not sample an integer-division case")

    if task == "number_of_decimal_digits":
        return {"primary": _exact_width(digits, rng)}, f"{digits}_digits"

    if task == "reverse_decimal_digits":
        category = "trailing_zero" if occurrence % 2 == 0 else "no_trailing_zero"
        if category == "trailing_zero":
            value = rng.randint(10 ** (digits - 2), 10 ** (digits - 1) - 1) * 10
        else:
            value = _exact_width(digits, rng)
            while value % 10 == 0:
                value = _exact_width(digits, rng)
        return {"primary": value}, category

    if task == "decimal_digit_occurrence_count":
        category = ("zero", "one", "multiple")[occurrence % 3]
        digit = rng.randint(0, 9)
        allowed = list(range(digits)) if digit else list(range(1, digits))
        count = 0 if category == "zero" else (1 if category == "one" else rng.randint(2, 4))
        positions = set(rng.sample(allowed, count))
        chars: list[str] = []
        for index in range(digits):
            if index in positions:
                chars.append(str(digit))
            else:
                choices = list(range(1, 10)) if index == 0 else list(range(10))
                chars.append(str(rng.choice([value for value in choices if value != digit])))
        return {"primary": int("".join(chars)), "digit": digit}, category

    if task == "even_odd":
        desired = occurrence % 2
        value = _exact_width(digits, rng)
        if value % 2 != desired:
            value = value + 1 if value < high else value - 1
        return {"primary": value}, "odd" if desired else "even"

    if task == "divisibility":
        divisible = occurrence % 2 == 0
        divisor = rng.randint(2, 9)
        if divisible:
            value = divisor * rng.randint(math.ceil(low / divisor), high // divisor)
        else:
            value = _exact_width(digits, rng)
            while value % divisor == 0:
                value = _exact_width(digits, rng)
        return {"primary": value, "operand": divisor}, "divisible" if divisible else "not_divisible"

    raise ValueError(f"task {task!r} does not use length-OOD sampling")


def _new_answer(task: str, inputs: Mapping[str, int]) -> int:
    primary = inputs["primary"]
    if task == "subtraction":
        return ops.subtraction(primary, inputs["operand"])
    if task == "integer_division":
        return ops.integer_division(primary, inputs["operand"])
    if task == "number_of_decimal_digits":
        return ops.number_of_decimal_digits(primary)
    if task == "reverse_decimal_digits":
        return ops.reverse_decimal_digits(primary)
    if task == "decimal_digit_occurrence_count":
        return ops.decimal_digit_occurrence_count(primary, inputs["digit"])
    if task == "even_odd":
        return int(ops.even_odd(primary))
    if task == "divisibility":
        return int(ops.divisibility(primary, inputs["operand"]))
    if task == "factorial":
        return ops.factorial(primary)
    raise AssertionError(task)


def _render_record(
    *, task: str, inputs: Mapping[str, int], answer: int, record_id: str,
    n_digits: int, evaluation_bucket: str, regime: str, stratum: str,
) -> dict[str, Any]:
    tokens = passage_tokens(task, answer, size=n_digits, **inputs)
    return {
        "schema_version": "integer-16/v1-evaluation",
        "id": record_id,
        "task": task,
        "n_digits": n_digits,
        "evaluation_bucket": evaluation_bucket,
        "evaluation_regime": regime,
        "sampling_stratum": stratum,
        "inputs": dict(inputs),
        "answer": answer,
        "answer_kind": getattr(TASK_SPECS[task], "answer_kind"),
        "tokens": list(tokens),
        "canonical_text": " ".join(tokens),
    }


def build_ood_records(
    *, tasks: Sequence[str] = TASK_NAMES, examples_per_task_per_length: int = 1_000
) -> list[dict[str, Any]]:
    if examples_per_task_per_length < 1:
        raise ValueError("examples_per_task_per_length must be positive")
    unknown = set(tasks) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"unknown T16 tasks: {sorted(unknown)}")
    records: list[dict[str, Any]] = []
    for task in TASK_NAMES:
        if task not in tasks:
            continue
        if task == "factorial":
            for value in FACTORIAL_OOD_DOMAIN:
                inputs = {"primary": value}
                answer = _new_answer(task, inputs)
                records.append(_render_record(
                    task=task, inputs=inputs, answer=answer,
                    record_id=f"ood:{task}:{value}", n_digits=len(str(value)),
                    evaluation_bucket="101_120", regime="factorial_value_ood",
                    stratum="input_101_120",
                ))
            continue
        for digits in OOD_LENGTHS:
            for occurrence in range(examples_per_task_per_length):
                if task in LEGACY_TASKS:
                    task_index = LEGACY_TASKS.index(task)
                    internal_occurrence = occurrence * len(OOD_LENGTHS) + (digits - OOD_LENGTHS[0])
                    legacy_id = internal_occurrence * len(LEGACY_TASKS) + task_index
                    record = build_legacy_record(
                        legacy_id, min_digits=6, max_digits=10,
                        seed=OOD_SEED, schema_version=OOD_SCHEMA_VERSION,
                    )
                    record["id"] = f"ood:{task}:{digits}:{occurrence}"
                    record["evaluation_bucket"] = str(digits)
                    record["evaluation_regime"] = (
                        "magnitude_ood_familiar_token_width" if digits == 6 else "token_length_ood"
                    )
                    record["sampling_stratum"] = legacy_diagnostic_stratum(record)
                    records.append(record)
                else:
                    inputs, stratum = _new_inputs(task, digits, occurrence)
                    answer = _new_answer(task, inputs)
                    records.append(_render_record(
                        task=task, inputs=inputs, answer=answer,
                        record_id=f"ood:{task}:{digits}:{occurrence}",
                        n_digits=digits, evaluation_bucket=str(digits),
                        regime="magnitude_ood_familiar_token_width" if digits == 6 else "token_length_ood",
                        stratum=stratum,
                    ))
    return records


__all__ = [
    "FACTORIAL_IID_DOMAIN",
    "FACTORIAL_OOD_DOMAIN",
    "OOD_LENGTHS",
    "OOD_SEED",
    "build_ood_records",
    "decimal_borrow_count",
    "decimal_carry_count",
    "legacy_diagnostic_stratum",
]
