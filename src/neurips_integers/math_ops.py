"""Authoritative operations for the integer multitask corpus.

The original eight training tasks remain frozen.  The eight T16 candidates are
listed separately until the T16 generator is installed, so the existing
eight-task generator and checkpoints remain valid.
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

T16_CANDIDATE_TASKS: tuple[str, ...] = (
    "subtraction",
    "integer_division",
    "number_of_decimal_digits",
    "reverse_decimal_digits",
    "decimal_digit_occurrence_count",
    "even_odd",
    "divisibility",
    "factorial",
)

HOLDOUT_TASKS: tuple[str, ...] = (
    "predecessor",
    "least_common_multiple",
    "modular_addition",
    "sort_ascending",
)
# Preserve compatibility with the existing eight-task pipeline.
TASK_NAMES: tuple[str, ...] = TRAINING_TASKS + HOLDOUT_TASKS

# Registries for the upcoming T16 pipeline.
T16_TASK_NAMES: tuple[str, ...] = (
    TRAINING_TASKS + T16_CANDIDATE_TASKS
)

ALL_TASK_NAMES: tuple[str, ...] = (
    T16_TASK_NAMES + HOLDOUT_TASKS
)


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


# Original eight training operations

def successor(value: int) -> int:
    return _nonnegative_integer(value, "value") + 1


def addition(left: int, right: int) -> int:
    return _nonnegative_integer(left, "left") + _nonnegative_integer(right, "right")


def multiplication(left: int, right: int) -> int:
    return _nonnegative_integer(left, "left") * _nonnegative_integer(right, "right")


def modulo(dividend: int, divisor: int) -> int:
    return _nonnegative_integer(dividend, "dividend") % _positive_integer(divisor, "divisor")


def greater_than(left: int, right: int) -> bool:
    return _nonnegative_integer(left, "left") > _nonnegative_integer(right, "right")


def decimal_digit_sum(value: int) -> int:
    number = _nonnegative_integer(value, "value")
    return sum(int(digit) for digit in str(number))


def greatest_common_divisor(left: int, right: int) -> int:
    left_value = _nonnegative_integer(left, "left")
    right_value = _nonnegative_integer(right, "right")
    if left_value == 0 and right_value == 0:
        raise ValueError("greatest_common_divisor is undefined for (0, 0)")
    return math.gcd(left_value, right_value)


def integer_list_sum(values: Iterable[int]) -> int:
    return sum(_integer_tuple(values, "values"))


# Eight candidate T16 operations

def subtraction(left: int, right: int) -> int:
    """Return left-right, restricted to nonnegative answers."""
    left_value = _nonnegative_integer(left, "left")
    right_value = _nonnegative_integer(right, "right")
    if right_value > left_value:
        raise ValueError("subtraction requires left >= right")
    return left_value - right_value


def integer_division(dividend: int, divisor: int) -> int:
    """Return floor division for a nonnegative dividend and positive divisor."""
    return _nonnegative_integer(dividend, "dividend") // _positive_integer(divisor, "divisor")


def number_of_decimal_digits(value: int) -> int:
    """Count ordinary base-10 digits; zero has one digit."""
    return len(str(_nonnegative_integer(value, "value")))


def reverse_decimal_digits(value: int) -> int:
    """Reverse ordinary decimal digits; leading zeros in the result are dropped."""
    number = _nonnegative_integer(value, "value")
    return int(str(number)[::-1])


def decimal_digit_occurrence_count(value: int, digit: int) -> int:
    """Count an ordinary decimal digit, independently of base-100 tokenization."""
    number = _nonnegative_integer(value, "value")
    digit_value = _nonnegative_integer(digit, "digit")
    if digit_value > 9:
        raise ValueError("digit must be between 0 and 9")
    return str(number).count(str(digit_value))


def even_odd(value: int) -> int:
    """Return 0 for even and 1 for odd."""
    return _nonnegative_integer(value, "value") % 2


def divisibility(dividend: int, divisor: int) -> int:
    """Return 1 when divisor divides dividend, otherwise 0."""
    number = _nonnegative_integer(dividend, "dividend")
    divisor_value = _positive_integer(divisor, "divisor")
    return int(number % divisor_value == 0)


def factorial(value: int) -> int:
    return math.factorial(_nonnegative_integer(value, "value"))


# Fixed holdout operations

def predecessor(value: int) -> int:
    return _positive_integer(value, "value") - 1


def least_common_multiple(left: int, right: int) -> int:
    return math.lcm(_positive_integer(left, "left"), _positive_integer(right, "right"))


def modular_addition(left: int, right: int, modulus: int) -> int:
    return (
        _nonnegative_integer(left, "left")
        + _nonnegative_integer(right, "right")
    ) % _positive_integer(modulus, "modulus")


def sort_ascending(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(_integer_tuple(values, "values")))
