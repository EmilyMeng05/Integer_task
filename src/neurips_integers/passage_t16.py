"""Base-100 Passage rendering for the frozen sixteen-task integer corpus.

The legacy vocabulary is kept as an exact prefix, so every existing token ID
is unchanged. New task and operator tokens are appended at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .passage import (
    FIXED_TOKENS as LEGACY_FIXED_TOKENS,
    TASK_SPECS as LEGACY_TASK_SPECS,
    VOCABULARY as LEGACY_VOCABULARY,
    decode_number,
    encode_number,
    passage_tokens as legacy_passage_tokens,
)


T16_NEW_FIXED_TOKENS: tuple[str, ...] = (
    "<SUBTRACTION>",
    "<INTEGER_DIVISION>",
    "<NUMBER_OF_DECIMAL_DIGITS>",
    "<REVERSE_DECIMAL_DIGITS>",
    "<DECIMAL_DIGIT_OCCURRENCE_COUNT>",
    "<EVEN_ODD>",
    "<DIVISIBILITY>",
    "<FACTORIAL>",
    "-",
    "/",
    "!",
)

FIXED_TOKENS: tuple[str, ...] = tuple(
    dict.fromkeys((*LEGACY_FIXED_TOKENS, *T16_NEW_FIXED_TOKENS))
)
VOCABULARY: tuple[str, ...] = tuple(
    dict.fromkeys((*LEGACY_VOCABULARY, *T16_NEW_FIXED_TOKENS))
)
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


NEW_TASK_SPECS: Mapping[str, TaskSpec] = {
    "subtraction": TaskSpec("<SUBTRACTION>", "binary", "-"),
    "integer_division": TaskSpec("<INTEGER_DIVISION>", "binary", "/"),
    "number_of_decimal_digits": TaskSpec(
        "<NUMBER_OF_DECIMAL_DIGITS>", "unary", None
    ),
    "reverse_decimal_digits": TaskSpec(
        "<REVERSE_DECIMAL_DIGITS>", "unary", None
    ),
    "decimal_digit_occurrence_count": TaskSpec(
        "<DECIMAL_DIGIT_OCCURRENCE_COUNT>", "digit_argument", None
    ),
    "even_odd": TaskSpec("<EVEN_ODD>", "unary", None, "boolean"),
    "divisibility": TaskSpec(
        "<DIVISIBILITY>", "binary", None, "boolean"
    ),
    "factorial": TaskSpec("<FACTORIAL>", "postfix_unary", "!"),
}

TASK_SPECS: Mapping[str, object] = {
    **LEGACY_TASK_SPECS,
    **NEW_TASK_SPECS,
}


def _require_nonnegative(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _number(value: object, *, name: str) -> tuple[str, ...]:
    return encode_number(_require_nonnegative(value, name=name))


def passage_tokens(
    task: str,
    answer: int,
    *,
    size: int,
    primary: int | None = None,
    operand: int | None = None,
    digit: int | None = None,
    values: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Return one canonical T16 answer-supervised sequence."""

    if task in LEGACY_TASK_SPECS:
        if digit is not None:
            raise ValueError("legacy task cannot receive digit")
        return legacy_passage_tokens(
            task,
            answer,
            size=size,
            primary=primary,
            operand=operand,
            values=values,
        )

    try:
        spec = NEW_TASK_SPECS[task]
    except KeyError:
        raise ValueError(f"unknown T16 integer task {task!r}") from None

    size_value = _require_nonnegative(size, name="size")
    if size_value < 1:
        raise ValueError("size must be positive")
    if primary is None or values is not None:
        raise ValueError("new T16 task requires primary and no values")

    input_parts = list(_number(primary, name="primary"))
    if spec.input_kind == "binary":
        if operand is None or digit is not None:
            raise ValueError("binary task requires operand and no digit")
        if spec.operator is not None:
            input_parts.append(spec.operator)
        input_parts.extend(
            ("<ARG_START>", *_number(operand, name="operand"), "<ARG_END>")
        )
    elif spec.input_kind == "digit_argument":
        if digit is None or operand is not None:
            raise ValueError("digit task requires digit and no operand")
        digit_value = _require_nonnegative(digit, name="digit")
        if digit_value > 9:
            raise ValueError("digit must be between 0 and 9")
        input_parts.extend(
            ("<ARG_START>", *_number(digit_value, name="digit"), "<ARG_END>")
        )
    elif spec.input_kind == "postfix_unary":
        if operand is not None or digit is not None:
            raise ValueError("postfix unary task accepts only primary")
        assert spec.operator is not None
        input_parts.append(spec.operator)
    elif operand is not None or digit is not None:
        raise ValueError("unary task accepts only primary")

    answer_value = _require_nonnegative(answer, name="answer")
    if spec.answer_kind == "boolean" and answer_value not in {0, 1}:
        raise ValueError("boolean answer must be 0 or 1")
    return (
        "<BOS>",
        "<SIZE>",
        *encode_number(size_value),
        *input_parts,
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
    "NEW_TASK_SPECS",
    "TASK_SPECS",
    "T16_NEW_FIXED_TOKENS",
    "TOKEN_TO_ID",
    "VOCABULARY",
    "decode_number",
    "encode_number",
    "passage_tokens",
    "render_passage",
]
