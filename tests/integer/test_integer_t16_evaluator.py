from __future__ import annotations

import pytest

pytest.importorskip("torch")

from neurips_integers import evaluate as base_evaluate
from neurips_integers.evaluate_t16 import _t16_evaluation_runtime
from neurips_integers.evaluation_t16_data import build_ood_records
from neurips_integers.generate_t16 import TASK_NAMES
from neurips_integers.passage import TOKEN_TO_ID as LEGACY_TOKEN_TO_ID
from neurips_integers.passage_t16 import TOKEN_TO_ID as T16_TOKEN_TO_ID


def test_runtime_switches_and_restores_vocabulary() -> None:
    assert base_evaluate.TOKEN_TO_ID is LEGACY_TOKEN_TO_ID
    with _t16_evaluation_runtime():
        assert base_evaluate.TOKEN_TO_ID is T16_TOKEN_TO_ID
        assert base_evaluate.parse_generated_answer(("01", "<EOS>")) == 1
    assert base_evaluate.TOKEN_TO_ID is LEGACY_TOKEN_TO_ID


def test_all_generated_records_have_prompt_only_before_equals() -> None:
    records = build_ood_records(examples_per_task_per_length=1)
    assert set(record["task"] for record in records) == set(TASK_NAMES)
    with _t16_evaluation_runtime():
        for record in records:
            prompt, answer = base_evaluate.split_prompt_and_answer(record["tokens"])
            assert prompt[-1] == "="
            assert answer[-1] == "<EOS>"
            assert len(answer) >= 2
