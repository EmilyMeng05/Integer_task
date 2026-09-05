from __future__ import annotations

from collections import Counter

from neurips_integers.evaluation_t16_data import (
    FACTORIAL_OOD_DOMAIN,
    build_ood_records,
    decimal_borrow_count,
    decimal_carry_count,
)
from neurips_integers.generate_t16 import TASK_NAMES


def test_small_ood_protocol_is_deterministic_and_balanced() -> None:
    first = build_ood_records(examples_per_task_per_length=1)
    second = build_ood_records(examples_per_task_per_length=1)
    assert first == second
    assert len(first) == 15 * 5 + len(FACTORIAL_OOD_DOMAIN)
    counts = Counter(record["task"] for record in first)
    assert counts["factorial"] == 20
    assert all(counts[task] == 5 for task in TASK_NAMES if task != "factorial")


def test_ood_regime_boundary_follows_base100_token_width() -> None:
    records = build_ood_records(
        tasks=("successor",), examples_per_task_per_length=2
    )
    by_length = {record["n_digits"]: record["evaluation_regime"] for record in records}
    assert by_length[6] == "magnitude_ood_familiar_token_width"
    assert all(by_length[length] == "token_length_ood" for length in range(7, 11))


def test_factorial_ood_is_the_unique_domain_101_through_120() -> None:
    records = build_ood_records(
        tasks=("factorial",), examples_per_task_per_length=7
    )
    assert [record["inputs"]["primary"] for record in records] == list(range(101, 121))
    assert all(record["evaluation_regime"] == "factorial_value_ood" for record in records)


def test_arithmetic_diagnostic_counts() -> None:
    assert decimal_carry_count(1299, 1) == 2
    assert decimal_carry_count(123, 100) == 0
    assert decimal_borrow_count(1000, 1) == 3
    assert decimal_borrow_count(987, 123) == 0


def test_new_task_records_are_semantically_correct() -> None:
    records = build_ood_records(examples_per_task_per_length=2)
    for record in records:
        task = record["task"]
        inputs = record["inputs"]
        answer = record["answer"]
        if task == "subtraction":
            assert answer == inputs["primary"] - inputs["operand"]
        elif task == "integer_division":
            assert answer == inputs["primary"] // inputs["operand"]
        elif task == "number_of_decimal_digits":
            assert answer == len(str(inputs["primary"]))
        elif task == "reverse_decimal_digits":
            assert answer == int(str(inputs["primary"])[::-1])
        elif task == "decimal_digit_occurrence_count":
            assert answer == str(inputs["primary"]).count(str(inputs["digit"]))
        elif task == "even_odd":
            assert answer == inputs["primary"] % 2
        elif task == "divisibility":
            assert answer == int(inputs["primary"] % inputs["operand"] == 0)
