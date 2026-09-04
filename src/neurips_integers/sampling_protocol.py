"""Deterministic, task-aware sampling audit for the eight new T16 tasks.

This module does not generate the production corpus.  It freezes and verifies
the sampling rules first, before the expensive eight-million-record build.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

from . import math_ops as ops


SAMPLING_PROTOCOL_VERSION = "integer-t16/sampling-v1"
DEFAULT_SEED = 20_260_904
DEFAULT_MIN_DIGITS = 1
DEFAULT_MAX_DIGITS = 5
DEFAULT_SAMPLES_PER_TASK_PER_LENGTH = 200
DEFAULT_REPORT = Path("results/integer-sampling/t16-sampling-audit.json")
FACTORIAL_MAX_INPUT = 100
FACTORIAL_MAX_BASE100_CHUNKS = 80
NEW_TASKS = ops.T16_CANDIDATE_TASKS


def _rng(task: str, digits: int, occurrence: int, seed: int) -> Random:
    payload = f"{seed}:{task}:{digits}:{occurrence}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return Random(value)


def _bounds(digits: int) -> tuple[int, int]:
    if not 1 <= digits <= 5:
        raise ValueError("digits must be between 1 and 5")
    return (0 if digits == 1 else 10 ** (digits - 1), 10**digits - 1)


def _exact_width_integer(digits: int, rng: Random) -> int:
    low, high = _bounds(digits)
    return rng.randint(low, high)


def subtraction_borrow_count(left: int, right: int) -> int:
    """Count decimal columns that borrow while computing left-right."""
    if right > left:
        raise ValueError("borrow count requires left >= right")
    borrows = 0
    incoming = 0
    while left or right or incoming:
        left_digit = left % 10
        right_digit = right % 10
        needs_borrow = left_digit - incoming < right_digit
        borrows += int(needs_borrow)
        incoming = int(needs_borrow)
        left //= 10
        right //= 10
    return borrows


def _sample_subtraction(digits: int, occurrence: int, rng: Random) -> tuple[dict[str, int], str]:
    category = "no_borrow" if digits == 1 or occurrence % 2 == 0 else "borrow"
    if category == "no_borrow":
        left_digits = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(digits - 1)]
        right_digits = [rng.randint(0, digit) for digit in left_digits]
        left = int("".join(map(str, left_digits)))
        right = int("".join(map(str, right_digits)))
    else:
        low, high = _bounds(digits)
        for _ in range(10_000):
            left = rng.randint(low, high)
            right = rng.randint(0, left)
            if subtraction_borrow_count(left, right) > 0:
                break
        else:
            raise RuntimeError("could not sample subtraction borrow case")
    return {"primary": left, "operand": right}, category


def _sample_integer_division(digits: int, occurrence: int, rng: Random) -> tuple[dict[str, int], str]:
    category = ("quotient_zero", "quotient_one", "exact_multi", "remainder_multi")[occurrence % 4]
    low, high = _bounds(digits)
    if category == "quotient_zero":
        divisor = rng.randint(max(1, low), high)
        dividend = rng.randint(0, divisor - 1)
    elif category == "quotient_one":
        divisor = rng.randint(max(1, low), high)
        dividend = divisor
    else:
        for _ in range(10_000):
            quotient = rng.randint(2, 9)
            minimum_divisor = max(1, math.ceil(low / quotient))
            maximum_divisor = max(1, high // quotient)
            if minimum_divisor > maximum_divisor:
                continue
            divisor = rng.randint(minimum_divisor, maximum_divisor)
            remainder = 0
            if category == "remainder_multi":
                if divisor < 2 or divisor * quotient + 1 > high:
                    continue
                remainder = rng.randint(1, min(divisor - 1, high - divisor * quotient))
            dividend = divisor * quotient + remainder
            if low <= dividend <= high:
                break
        else:
            raise RuntimeError("could not sample integer-division case")
    return {"primary": dividend, "operand": divisor}, category


def _sample_reverse(digits: int, occurrence: int, rng: Random) -> tuple[dict[str, int], str]:
    category = "trailing_zero" if occurrence % 2 == 0 else "no_trailing_zero"
    if category == "trailing_zero":
        if digits == 1:
            value = 0
        else:
            prefix = rng.randint(10 ** (digits - 2), 10 ** (digits - 1) - 1)
            value = prefix * 10
    else:
        low, high = _bounds(digits)
        for _ in range(100):
            value = rng.randint(low, high)
            if value % 10:
                break
    return {"primary": value}, category


def _sample_occurrence(digits: int, occurrence: int, rng: Random) -> tuple[dict[str, int], str]:
    categories = ("zero", "one") if digits == 1 else ("zero", "one", "multiple")
    category = categories[occurrence % len(categories)]
    digit = rng.randint(0, 9)
    if category == "multiple" and digit == 0 and digits == 2:
        digit = rng.randint(1, 9)
    chars: list[str] = []
    if category == "zero":
        positions: set[int] = set()
    elif category == "one":
        allowed = list(range(digits)) if digit else list(range(1, digits))
        if not allowed:  # the one-digit number zero
            return {"primary": 0, "digit": 0}, "one"
        positions = {rng.choice(allowed)}
    else:
        allowed = list(range(digits)) if digit else list(range(1, digits))
        count = rng.randint(2, min(len(allowed), 4))
        positions = set(rng.sample(allowed, count))
    for index in range(digits):
        if index in positions:
            chars.append(str(digit))
            continue
        choices = list(range(1, 10)) if index == 0 else list(range(10))
        choices = [candidate for candidate in choices if candidate != digit]
        chars.append(str(rng.choice(choices)))
    value = int("".join(chars))
    return {"primary": value, "digit": digit}, category


def _sample_parity(digits: int, occurrence: int, rng: Random) -> tuple[dict[str, int], str]:
    desired = occurrence % 2
    low, high = _bounds(digits)
    value = rng.randint(low, high)
    if value % 2 != desired:
        value = value + 1 if value < high else value - 1
    return {"primary": value}, "odd" if desired else "even"


def _sample_divisibility(digits: int, occurrence: int, rng: Random) -> tuple[dict[str, int], str]:
    divisible = occurrence % 2 == 0
    low, high = _bounds(digits)
    divisor = rng.randint(2, 9)
    if divisible:
        minimum_q = math.ceil(low / divisor)
        maximum_q = high // divisor
        value = divisor * rng.randint(minimum_q, maximum_q)
    else:
        value = rng.randint(low, high)
        while value % divisor == 0:
            value = rng.randint(low, high)
    return {"primary": value, "operand": divisor}, "divisible" if divisible else "not_divisible"


def _sample_factorial(occurrence: int, rng: Random) -> tuple[dict[str, int], str]:
    ranges = ((0, 20, "small_0_20"), (21, 50, "medium_21_50"), (51, 100, "large_51_100"))
    low, high, category = ranges[occurrence % len(ranges)]
    return {"primary": rng.randint(low, high)}, category


def sample_inputs(task: str, digits: int, occurrence: int, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Return one deterministic input specification and its sampling stratum."""
    if task not in NEW_TASKS:
        raise ValueError(f"not a new T16 task: {task!r}")
    if occurrence < 0:
        raise ValueError("occurrence must be nonnegative")
    rng = _rng(task, digits, occurrence, seed)
    if task == "subtraction":
        inputs, stratum = _sample_subtraction(digits, occurrence, rng)
    elif task == "integer_division":
        inputs, stratum = _sample_integer_division(digits, occurrence, rng)
    elif task == "number_of_decimal_digits":
        inputs, stratum = {"primary": _exact_width_integer(digits, rng)}, f"{digits}_digits"
    elif task == "reverse_decimal_digits":
        inputs, stratum = _sample_reverse(digits, occurrence, rng)
    elif task == "decimal_digit_occurrence_count":
        inputs, stratum = _sample_occurrence(digits, occurrence, rng)
    elif task == "even_odd":
        inputs, stratum = _sample_parity(digits, occurrence, rng)
    elif task == "divisibility":
        inputs, stratum = _sample_divisibility(digits, occurrence, rng)
    else:
        inputs, stratum = _sample_factorial(occurrence, rng)
    return {"inputs": inputs, "stratum": stratum}


def answer(task: str, inputs: Mapping[str, int]) -> int:
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
        return ops.even_odd(primary)
    if task == "divisibility":
        return ops.divisibility(primary, inputs["operand"])
    if task == "factorial":
        return ops.factorial(primary)
    raise AssertionError(task)


def validate_sample(task: str, sample: Mapping[str, Any]) -> None:
    inputs = sample["inputs"]
    stratum = sample["stratum"]
    result = answer(task, inputs)
    if task == "subtraction":
        borrow_count = subtraction_borrow_count(inputs["primary"], inputs["operand"])
        assert result >= 0
        assert (borrow_count > 0) == (stratum == "borrow")
    elif task == "integer_division":
        dividend, divisor = inputs["primary"], inputs["operand"]
        assert divisor > 0 and result == dividend // divisor
        if stratum == "quotient_zero": assert result == 0
        if stratum == "quotient_one": assert result == 1 and dividend % divisor == 0
        if stratum == "exact_multi": assert result >= 2 and dividend % divisor == 0
        if stratum == "remainder_multi": assert result >= 2 and dividend % divisor > 0
    elif task == "number_of_decimal_digits":
        assert result == len(str(inputs["primary"]))
    elif task == "reverse_decimal_digits":
        assert result == int(str(inputs["primary"])[::-1])
    elif task == "decimal_digit_occurrence_count":
        assert result == str(inputs["primary"]).count(str(inputs["digit"]))
        if stratum == "zero": assert result == 0
        if stratum == "one": assert result == 1
        if stratum == "multiple": assert result >= 2
    elif task == "even_odd":
        assert result in (0, 1)
        assert (result == 1) == (stratum == "odd")
    elif task == "divisibility":
        assert result in (0, 1)
        assert (result == 1) == (stratum == "divisible")
    elif task == "factorial":
        assert 0 <= inputs["primary"] <= FACTORIAL_MAX_INPUT
        chunks = max(1, math.ceil(len(str(result)) / 2))
        assert chunks <= FACTORIAL_MAX_BASE100_CHUNKS


def audit_sampling(*, samples_per_task_per_length: int = DEFAULT_SAMPLES_PER_TASK_PER_LENGTH, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    if samples_per_task_per_length <= 0:
        raise ValueError("samples_per_task_per_length must be positive")
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: dict[str, Counter[str]] = defaultdict(Counter)
    examples = 0
    for task in NEW_TASKS:
        for digits in range(DEFAULT_MIN_DIGITS, DEFAULT_MAX_DIGITS + 1):
            for occurrence in range(samples_per_task_per_length):
                sample = sample_inputs(task, digits, occurrence, seed=seed)
                validate_sample(task, sample)
                integer_inputs = [
                    value for value in sample["inputs"].values()
                    if isinstance(value, int) and not isinstance(value, bool)
                ]
                actual_digits = max(len(str(value)) for value in integer_inputs)
                strata[task][sample["stratum"]] += 1
                lengths[task][str(actual_digits)] += 1
                examples += 1
    return {
        "status": "passed",
        "protocol_version": SAMPLING_PROTOCOL_VERSION,
        "seed": seed,
        "samples_per_task_per_length": samples_per_task_per_length,
        "examples_checked": examples,
        "original_eight_sampling": "unchanged",
        "factorial_domain": [0, FACTORIAL_MAX_INPUT],
        "tasks": {
            task: {
                "strata": dict(sorted(strata[task].items())),
                "actual_max_input_digit_lengths": dict(sorted(lengths[task].items(), key=lambda item: int(item[0]))),
            }
            for task in NEW_TASKS
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-task-per-length", type=int, default=DEFAULT_SAMPLES_PER_TASK_PER_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = audit_sampling(samples_per_task_per_length=args.samples_per_task_per_length, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
