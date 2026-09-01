"""Tests for the authoritative Phase 1 integer operations."""

from __future__ import annotations

import math
import random

import pytest

from neurips_integers import math_ops as ops


def test_task_registry_is_complete_and_disjoint() -> None:
    assert len(ops.TRAINING_TASKS) == 8
    assert len(ops.HOLDOUT_TASKS) == 4
    assert len(ops.TASK_NAMES) == 12
    assert set(ops.TRAINING_TASKS).isdisjoint(ops.HOLDOUT_TASKS)
    assert len(set(ops.TASK_NAMES)) == len(ops.TASK_NAMES)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 1), (7, 8), (99, 100), (1299, 1300)],
)
def test_successor(value: int, expected: int) -> None:
    assert ops.successor(value) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(0, 0, 0), (247, 85, 332), (999, 1, 1000)],
)
def test_addition(left: int, right: int, expected: int) -> None:
    assert ops.addition(left, right) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(0, 999, 0), (23, 14, 322), (99, 99, 9801)],
)
def test_multiplication(left: int, right: int, expected: int) -> None:
    assert ops.multiplication(left, right) == expected


@pytest.mark.parametrize(
    ("dividend", "divisor", "expected"),
    [(247, 12, 7), (100, 10, 0), (3, 5, 3)],
)
def test_modulo(dividend: int, divisor: int, expected: int) -> None:
    assert ops.modulo(dividend, divisor) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(247, 85, True), (85, 247, False), (7, 7, False)],
)
def test_greater_than(left: int, right: int, expected: bool) -> None:
    assert ops.greater_than(left, right) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (7, 7), (137, 11), (90909, 27)],
)
def test_decimal_digit_sum(value: int, expected: int) -> None:
    assert ops.decimal_digit_sum(value) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(84, 30, 6), (0, 9, 9), (17, 13, 1)],
)
def test_greatest_common_divisor(left: int, right: int, expected: int) -> None:
    assert ops.greatest_common_divisor(left, right) == expected


def test_integer_list_sum() -> None:
    assert ops.integer_list_sum([31, 4, 18]) == 53
    assert ops.integer_list_sum((0, 0, 7)) == 7


@pytest.mark.parametrize(
    ("value", "expected"), [(1, 0), (8, 7), (1300, 1299)]
)
def test_predecessor(value: int, expected: int) -> None:
    assert ops.predecessor(value) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(12, 18, 36), (7, 5, 35), (9, 9, 9)],
)
def test_least_common_multiple(left: int, right: int, expected: int) -> None:
    assert ops.least_common_multiple(left, right) == expected


@pytest.mark.parametrize(
    ("left", "right", "modulus", "expected"),
    [(47, 28, 10, 5), (9, 8, 7, 3), (0, 0, 5, 0)],
)
def test_modular_addition(
    left: int, right: int, modulus: int, expected: int
) -> None:
    assert ops.modular_addition(left, right, modulus) == expected


def test_sort_ascending_preserves_duplicates() -> None:
    assert ops.sort_ascending([31, 4, 18, 4]) == (4, 4, 18, 31)


def test_invalid_domains_are_rejected() -> None:
    with pytest.raises(ValueError):
        ops.successor(-1)
    with pytest.raises(ValueError):
        ops.predecessor(0)
    with pytest.raises(ValueError):
        ops.modulo(10, 0)
    with pytest.raises(ValueError):
        ops.greatest_common_divisor(0, 0)
    with pytest.raises(ValueError):
        ops.least_common_multiple(0, 4)
    with pytest.raises(ValueError):
        ops.modular_addition(1, 2, 0)
    with pytest.raises(ValueError):
        ops.integer_list_sum([])
    with pytest.raises(ValueError):
        ops.sort_ascending([])
    with pytest.raises(TypeError):
        ops.addition(True, 1)


def test_random_arithmetic_properties() -> None:
    rng = random.Random(20260830)
    for _ in range(500):
        left = rng.randrange(0, 1_000_000)
        right = rng.randrange(0, 1_000_000)
        divisor = rng.randrange(1, 10_000)
        assert ops.addition(left, right) == left + right
        assert ops.multiplication(left, right) == left * right
        assert ops.modulo(left, divisor) == left % divisor
        if left or right:
            assert ops.greatest_common_divisor(left, right) == math.gcd(left, right)

