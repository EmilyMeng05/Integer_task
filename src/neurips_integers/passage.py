"""Passage Math rendering for the Phase 1 integer multitask corpus.

The number encoder and original fixed-token order are imported from the
permutation package.  Integer tasks only append domain-specific tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from neurips_permutations.passage import (
    NUMBER_TOKENS,
    ORIGINAL_FIXED_TOKENS,
    decode_number,
    encode_number,
)

from .math_ops import TRAINING_TASKS


INTEGER_FIXED_TOKENS: tuple[str, ...] = (
    "<SUCCESSOR>",
    "<ADDITION>",
    "<MULTIPLICATION>",
    "<MODULO>",
    "<GREATER_THAN>",
    "<DECIMAL_DIGIT_SUM>",
    "<GREATEST_COMMON_DIVISOR>",
    "<INTEGER_LIST_SUM>",
    "<ARG_START>",
    "<ARG_END>",
    "<LIST_START>",
    "<LIST_END>",
    "%",
    ">",
)

FIXED_TOKENS: tuple[str, ...] = tuple(
    dict.fromkeys((*ORIGINAL_FIXED_TOKENS, *INTEGER_FIXED_TOKENS))
)
VOCABULARY: tuple[str, ...] = NUMBER_TOKENS + FIXED_TOKENS
TOKEN_TO_ID: Mapping[str, int] = {
    token: index for index, token in enumerate(VOCABULARY)
}
ID_TO_TOKEN: Mapping[int, str] = {
    index: token for token, index in TOKEN_TO_ID.items()
}


@dataclass(frozen=True)
class TaskSpec:
    token: str
    input_kind: str
    operator: str | None
    answer_kind: str = "integer"


TASK_SPECS: Mapping[str, TaskSpec] = {
    "successor": TaskSpec("<SUCCESSOR>", "unary", None),
    "addition": TaskSpec("<ADDITION>", "binary", "+"),
    "multiplication": TaskSpec("<MULTIPLICATION>", "binary", "*"),
    "modulo": TaskSpec("<MODULO>", "binary", "%"),
    "greater_than": TaskSpec(
        "<GREATER_THAN>", "binary", ">", answer_kind="boolean"
    ),
    "decimal_digit_sum": TaskSpec("<DECIMAL_DIGIT_SUM>", "unary", None),
    "greatest_common_divisor": TaskSpec(
        "<GREATEST_COMMON_DIVISOR>", "binary", None
    ),
    "integer_list_sum": TaskSpec("<INTEGER_LIST_SUM>", "list", None),
}

if tuple(TASK_SPECS) != TRAINING_TASKS:
    raise RuntimeError("integer math and Passage task registries disagree")


def _require_nonnegative(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _number(value: object, *, name: str) -> tuple[str, ...]:
    return encode_number(_require_nonnegative(value, name=name))


def _list_tokens(values: Sequence[int]) -> tuple[str, ...]:
    if not values:
        raise ValueError("values must be nonempty")
    result: list[str] = ["<LIST_START>"]
    for index, value in enumerate(values):
        if index:
            result.append(",")
        result.extend(_number(value, name=f"values[{index}]"))
    result.append("<LIST_END>")
    return tuple(result)


def passage_tokens(
    task: str,
    answer: int,
    *,
    size: int,
    primary: int | None = None,
    operand: int | None = None,
    values: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Return one canonical answer-supervised integer sequence."""

    try:
        spec = TASK_SPECS[task]
    except KeyError:
        raise ValueError(f"unknown integer task {task!r}") from None
    size_value = _require_nonnegative(size, name="size")
    if size_value < 1:
        raise ValueError("size must be positive")

    if spec.input_kind == "list":
        if primary is not None or operand is not None or values is None:
            raise ValueError("list task requires only values")
        input_tokens = _list_tokens(values)
    else:
        if primary is None or values is not None:
            raise ValueError(f"{spec.input_kind} task requires primary")
        input_parts = list(_number(primary, name="primary"))
        if spec.input_kind == "binary":
            if operand is None:
                raise ValueError("binary task requires operand")
            if spec.operator is not None:
                input_parts.append(spec.operator)
            input_parts.extend(("<ARG_START>", *_number(operand, name="operand"), "<ARG_END>"))
        elif operand is not None:
            raise ValueError("unary task cannot receive operand")
        input_tokens = tuple(input_parts)

    answer_value = _require_nonnegative(answer, name="answer")
    if spec.answer_kind == "boolean" and answer_value not in {0, 1}:
        raise ValueError("boolean answer must be 0 or 1")
    return (
        "<BOS>",
        "<SIZE>",
        *encode_number(size_value),
        *input_tokens,
        spec.token,
        "=",
        *encode_number(answer_value),
        "<EOS>",
    )


def render_passage(task: str, answer: int, **kwargs: object) -> str:
    return " ".join(passage_tokens(task, answer, **kwargs))  # type: ignore[arg-type]


__all__ = [
    "FIXED_TOKENS",
    "ID_TO_TOKEN",
    "INTEGER_FIXED_TOKENS",
    "TASK_SPECS",
    "TOKEN_TO_ID",
    "TaskSpec",
    "VOCABULARY",
    "decode_number",
    "encode_number",
    "passage_tokens",
    "render_passage",
]
