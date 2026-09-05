from __future__ import annotations

from neurips_integers import transfer_related_holdouts as transfer
from neurips_integers.passage import TOKEN_TO_ID as BASE_TOKEN_TO_ID


def test_each_holdout_token_is_appended_without_changing_base_ids() -> None:
    for task, token in transfer.TASK_TOKENS.items():
        mapping, inverse = transfer.task_vocabulary(task)
        assert all(mapping[name] == index for name, index in BASE_TOKEN_TO_ID.items())
        assert mapping[token] == len(BASE_TOKEN_TO_ID)
        assert inverse[mapping[token]] == token


def test_lcm_record_uses_base100_binary_grammar() -> None:
    record = transfer.build_record(
        "example", "least_common_multiple", {"primary": 12, "operand": 18}
    )
    assert record["answer"] == 36
    assert record["tokens"] == [
        "<BOS>", "<SIZE>", "02", "12", "<ARG_START>", "18", "<ARG_END>",
        "<LEAST_COMMON_MULTIPLE>", "=", "36", "<EOS>",
    ]


def test_sort_record_and_answer_parser() -> None:
    record = transfer.build_record(
        "example", "sort_ascending", {"values": [247, 85, 103]}
    )
    assert record["answer"] == [85, 103, 247]
    equals = record["tokens"].index("=")
    assert transfer.parse_answer("sort_ascending", record["tokens"][equals + 1 :]) == [85, 103, 247]


def test_all_splits_are_deterministic_and_semantically_disjoint() -> None:
    for task in transfer.TASK_TOKENS:
        first = transfer.build_splits(
            task, shots=20, validation_per_length=2, test_per_length=3
        )
        second = transfer.build_splits(
            task, shots=20, validation_per_length=2, test_per_length=3
        )
        assert first == second
        assert len(first["train"]) == 20
        seen: set[tuple[object, ...]] = set()
        for records in first.values():
            for record in records:
                inputs = record["inputs"]
                if task == "least_common_multiple":
                    signature = tuple(
                        sorted((inputs["primary"], inputs["operand"]))
                    )
                elif task == "modular_addition":
                    signature = (
                        *sorted((inputs["primary"], inputs["operand"])),
                        inputs["modulus"],
                    )
                elif task == "sort_ascending":
                    signature = tuple(sorted(inputs["values"]))
                else:
                    raise AssertionError(f"Unhandled transfer task: {task}")
                assert signature not in seen
                seen.add(signature)


def test_scalar_parser_rejects_extra_tokens() -> None:
    assert transfer.parse_answer("least_common_multiple", ("36", "<EOS>")) == 36
    assert transfer.parse_answer("least_common_multiple", ("36", "37", "<EOS>")) is None

def test_modular_addition_record_uses_three_argument_grammar() -> None:
    record = transfer.build_record(
        "example",
        "modular_addition",
        {"primary": 17, "operand": 9, "modulus": 5},
    )

    assert record["answer"] == 1
    assert record["tokens"] == [
        "<BOS>",
        "<SIZE>",
        "02",
        "17",
        "<ARG_START>",
        "09",
        "<ARG_END>",
        "<ARG_START>",
        "05",
        "<ARG_END>",
        "<MODULAR_ADDITION>",
        "=",
        "01",
        "<EOS>",
    ]

    equals = record["tokens"].index("=")
    assert (
        transfer.parse_answer(
            "modular_addition",
            record["tokens"][equals + 1 :],
        )
        == 1
    ) 