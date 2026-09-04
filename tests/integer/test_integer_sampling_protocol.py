from __future__ import annotations

import pytest

from neurips_integers import math_ops as ops
from neurips_integers import sampling_protocol as sampling


def test_original_training_task_list_is_not_changed() -> None:
    assert len(ops.TRAINING_TASKS) == 8
    assert len(ops.T16_CANDIDATE_TASKS) == 8
    assert not set(ops.TRAINING_TASKS) & set(ops.T16_CANDIDATE_TASKS)


@pytest.mark.parametrize("task", sampling.NEW_TASKS)
@pytest.mark.parametrize("digits", range(1, 6))
def test_sampling_is_deterministic_and_valid(task: str, digits: int) -> None:
    for occurrence in range(40):
        first = sampling.sample_inputs(task, digits, occurrence)
        second = sampling.sample_inputs(task, digits, occurrence)
        assert first == second
        sampling.validate_sample(task, first)


def test_subtraction_covers_borrow_and_no_borrow() -> None:
    strata = {
        sampling.sample_inputs("subtraction", 5, index)["stratum"]
        for index in range(20)
    }
    assert strata == {"borrow", "no_borrow"}


def test_integer_division_covers_four_behaviors() -> None:
    strata = {
        sampling.sample_inputs("integer_division", 5, index)["stratum"]
        for index in range(40)
    }
    assert strata == {"quotient_zero", "quotient_one", "exact_multi", "remainder_multi"}


def test_digit_occurrence_covers_zero_one_and_multiple() -> None:
    strata = {
        sampling.sample_inputs("decimal_digit_occurrence_count", 5, index)["stratum"]
        for index in range(30)
    }
    assert strata == {"zero", "one", "multiple"}


def test_binary_tasks_are_exactly_balanced() -> None:
    for task in ("even_odd", "divisibility"):
        counts: dict[str, int] = {}
        for index in range(100):
            stratum = sampling.sample_inputs(task, 5, index)["stratum"]
            counts[stratum] = counts.get(stratum, 0) + 1
        assert sorted(counts.values()) == [50, 50]


def test_factorial_is_bounded_for_context_length() -> None:
    for index in range(300):
        sample = sampling.sample_inputs("factorial", 5, index)
        value = sample["inputs"]["primary"]
        assert 0 <= value <= sampling.FACTORIAL_MAX_INPUT
        sampling.validate_sample("factorial", sample)


def test_complete_audit_passes() -> None:
    report = sampling.audit_sampling(samples_per_task_per_length=10)
    assert report["status"] == "passed"
    assert report["examples_checked"] == 8 * 5 * 10
    assert report["original_eight_sampling"] == "unchanged"
