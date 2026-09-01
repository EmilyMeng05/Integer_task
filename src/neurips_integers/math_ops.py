"""Authoritative mathematical operations for the integer multitask corpus.

This module deliberately contains no tokenization, sampling, or model code.
The data generator and full verifier should both call these functions so that
the mathematical conventions are explicit and testable.
"""

from __future__ import annotations

from collections.abc import Iterable
import math


TRAINING_TASKS: tuple[str, ...] = (
    "successor",
    "addition",
    "multiplication",
    "modulo",
    "greater_than",
    "decimal_digit_sum",
    "greatest_common_divisor",
    "integer_list_sum",
)

HOLDOUT_TASKS: tuple[str, ...] = (
    "predecessor",
    "least_common_multiple",
    "modular_addition",
    "sort_ascending",
)

TASK_NAMES: tuple[str, ...] = TRAINING_TASKS + HOLDOUT_TASKS


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer_tuple(values: Iterable[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    if not result:
        raise ValueError(f"{name} must contain at least one integer")
    for index, value in enumerate(result):
        _nonnegative_integer(value, f"{name}[{index}]")
    return result


# Phase 1 training operations


def successor(value: int) -> int:
    """Return ``value + 1`` for a nonnegative integer."""

    return _nonnegative_integer(value, "value") + 1


def addition(left: int, right: int) -> int:
    """Return the sum of two nonnegative integers."""

    return _nonnegative_integer(left, "left") + _nonnegative_integer(
        right, "right"
    )


def multiplication(left: int, right: int) -> int:
    """Return the product of two nonnegative integers."""

    return _nonnegative_integer(left, "left") * _nonnegative_integer(
        right, "right"
    )


def modulo(dividend: int, divisor: int) -> int:
    """Return the nonnegative remainder ``dividend % divisor``."""

    return _nonnegative_integer(dividend, "dividend") % _positive_integer(
        divisor, "divisor"
    )


def greater_than(left: int, right: int) -> bool:
    """Return whether ``left`` is strictly greater than ``right``."""

    return _nonnegative_integer(left, "left") > _nonnegative_integer(
        right, "right"
    )


def decimal_digit_sum(value: int) -> int:
    """Return the sum of the ordinary base-10 digits of ``value``."""

    number = _nonnegative_integer(value, "value")
    return sum(int(digit) for digit in str(number))


def greatest_common_divisor(left: int, right: int) -> int:
    """Return ``gcd(left, right)``; the pair ``(0, 0)`` is undefined."""

    left_value = _nonnegative_integer(left, "left")
    right_value = _nonnegative_integer(right, "right")
    if left_value == 0 and right_value == 0:
        raise ValueError("greatest_common_divisor is undefined for (0, 0)")
    return math.gcd(left_value, right_value)


def integer_list_sum(values: Iterable[int]) -> int:
    """Return the sum of a nonempty sequence of nonnegative integers."""

    return sum(_integer_tuple(values, "values"))


# Fixed holdout operations


def predecessor(value: int) -> int:
    """Return ``value - 1`` for a positive integer."""

    return _positive_integer(value, "value") - 1


def least_common_multiple(left: int, right: int) -> int:
    """Return the LCM of two positive integers."""

    left_value = _positive_integer(left, "left")
    right_value = _positive_integer(right, "right")
    return math.lcm(left_value, right_value)


def modular_addition(left: int, right: int, modulus: int) -> int:
    """Return ``(left + right) % modulus`` for a positive modulus."""

    return (
        _nonnegative_integer(left, "left")
        + _nonnegative_integer(right, "right")
    ) % _positive_integer(modulus, "modulus")


def sort_ascending(values: Iterable[int]) -> tuple[int, ...]:
    """Return a nonempty integer sequence in nondecreasing order."""

    return tuple(sorted(_integer_tuple(values, "values")))

