"""Autoregressive IID and length-OOD evaluation for integer checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from neurips_permutations.training import TrainConfig

from .passage import ID_TO_TOKEN, TOKEN_TO_ID, decode_number
from .production import (
    DEFAULT_CONFIG,
    ProductionRun,
    build_matrix,
    completion_is_valid,
)
from .training import automatic_device, build_integer_model


DEFAULT_IID_MANIFEST = Path("data/integer-4m-v1/manifest.json")
DEFAULT_OOD_MANIFEST = Path("data/integer-ood-40k-v1/manifest.json")
DEFAULT_OUTPUT_DIR = Path("results/integer-v1")
IID_LENGTHS = (1, 2, 3, 4, 5)
OOD_LENGTHS = (6, 7, 8, 9, 10)
FORMAT_VERSION = 2
TRAINING_MAX_DECIMAL_DIGITS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def split_prompt_and_answer(tokens: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a canonical record after `=`, without exposing answer tokens."""

    if tokens.count("=") != 1:
        raise ValueError("record must contain exactly one equals token")
    equals = tokens.index("=")
    prompt = tuple(tokens[: equals + 1])
    answer = tuple(tokens[equals + 1 :])
    if not answer or answer[-1] != "<EOS>":
        raise ValueError("record answer must end with EOS")
    return prompt, answer


def parse_generated_answer(tokens: Sequence[str]) -> int | None:
    """Parse one complete canonical integer answer followed by EOS."""

    if not tokens or tokens[-1] != "<EOS>" or tokens.count("<EOS>") != 1:
        return None
    try:
        value, next_index = decode_number(tokens[:-1])
    except (TypeError, ValueError):
        return None
    return value if next_index == len(tokens) - 1 else None


def input_base100_width(n_digits: int) -> int:
    """Return the number of base-100 tokens required by an n-digit operand."""

    if isinstance(n_digits, bool) or not isinstance(n_digits, int) or n_digits < 1:
        raise ValueError("n_digits must be a positive integer")
    return (n_digits + 1) // 2


def generalization_regime(n_digits: int) -> str:
    """Name the IID, magnitude-OOD, or true token-length-OOD regime."""

    width = input_base100_width(n_digits)
    training_width = input_base100_width(TRAINING_MAX_DECIMAL_DIGITS)
    if n_digits <= TRAINING_MAX_DECIMAL_DIGITS:
        return "iid"
    if width <= training_width:
        return "magnitude_ood_familiar_token_width"
    return "token_length_ood"


def _manifest_records(
    manifest_path: Path,
    *,
    shard_indices: Sequence[int] | None = None,
) -> Iterator[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("shards")
    if not isinstance(entries, list):
        raise ValueError(f"manifest {manifest_path} has no shard list")
    selected = tuple(range(len(entries))) if shard_indices is None else tuple(shard_indices)
    for index in selected:
        if not 0 <= index < len(entries):
            raise IndexError(f"shard index {index} is out of range")
        entry = entries[index]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("filename"), str):
            raise ValueError(f"invalid shard entry {index}")
        path = manifest_path.parent / str(entry["filename"])
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"non-object record in {path}")
                yield record


def load_balanced_records(
    manifest_path: Path,
    *,
    tasks: Sequence[str],
    lengths: Sequence[int],
    examples_per_task_per_length: int,
    shard_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    if examples_per_task_per_length < 1:
        raise ValueError("examples_per_task_per_length must be positive")
    task_set = set(tasks)
    length_set = set(lengths)
    wanted = {
        (task, length): examples_per_task_per_length
        for task in tasks
        for length in lengths
    }
    counts: dict[tuple[str, int], int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for record in _manifest_records(manifest_path, shard_indices=shard_indices):
        task = record.get("task")
        length = record.get("n_digits")
        if task not in task_set or length not in length_set:
            continue
        key = (str(task), int(length))
        if counts[key] >= wanted[key]:
            continue
        records.append(record)
        counts[key] += 1
        if counts == wanted:
            break
    missing = {key: wanted[key] - counts[key] for key in wanted if counts[key] != wanted[key]}
    if missing:
        raise ValueError(f"evaluation manifest is missing balanced groups: {missing}")
    return records


@dataclass
class MetricAccumulator:
    examples: int = 0
    exact: int = 0
    well_formed: int = 0
    generated_token_matches: int = 0
    generated_token_slots: int = 0
    teacher_loss_sum: float = 0.0
    teacher_tokens: int = 0
    teacher_token_correct: int = 0
    teacher_sequence_correct: int = 0
    target_counts: Counter[int] = field(default_factory=Counter)
    generated_value_counts: Counter[int] = field(default_factory=Counter)
    target_equal_one_examples: int = 0
    target_equal_one_exact: int = 0
    target_greater_than_one_examples: int = 0
    target_greater_than_one_exact: int = 0

    def update(
        self,
        *,
        expected_tokens: Sequence[str],
        generated_tokens: Sequence[str],
        expected_value: int,
        teacher_loss_sum: float,
        teacher_tokens: int,
        teacher_token_correct: int,
        teacher_sequence_correct: bool,
    ) -> None:
        parsed = parse_generated_answer(generated_tokens)
        width = max(len(expected_tokens), len(generated_tokens))
        self.examples += 1
        self.exact += int(parsed == expected_value)
        self.well_formed += int(parsed is not None)
        self.target_counts[expected_value] += 1
        if parsed is not None:
            self.generated_value_counts[parsed] += 1
        if expected_value == 1:
            self.target_equal_one_examples += 1
            self.target_equal_one_exact += int(parsed == expected_value)
        elif expected_value > 1:
            self.target_greater_than_one_examples += 1
            self.target_greater_than_one_exact += int(parsed == expected_value)
        self.generated_token_matches += sum(
            left == right for left, right in zip(expected_tokens, generated_tokens)
        )
        self.generated_token_slots += width
        self.teacher_loss_sum += teacher_loss_sum
        self.teacher_tokens += teacher_tokens
        self.teacher_token_correct += teacher_token_correct
        self.teacher_sequence_correct += int(teacher_sequence_correct)

    def summary(self) -> dict[str, Any]:
        most_common_target = sorted(
            self.target_counts.items(), key=lambda item: (-item[1], item[0])
        )[0] if self.target_counts else (None, 0)
        generated = sorted(
            self.generated_value_counts.items(), key=lambda item: (-item[1], item[0])
        )
        return {
            "examples": self.examples,
            "autoregressive_exact": self.exact,
            "autoregressive_exact_accuracy": self.exact / max(1, self.examples),
            "well_formed": self.well_formed,
            "well_formed_rate": self.well_formed / max(1, self.examples),
            "autoregressive_token_accuracy": self.generated_token_matches
            / max(1, self.generated_token_slots),
            "teacher_forced_loss": self.teacher_loss_sum
            / max(1, self.teacher_tokens),
            "teacher_forced_token_accuracy": self.teacher_token_correct
            / max(1, self.teacher_tokens),
            "teacher_forced_sequence_accuracy": self.teacher_sequence_correct
            / max(1, self.examples),
            "supervised_tokens": self.teacher_tokens,
            "most_common_target_baseline": {
                "value": most_common_target[0],
                "count": most_common_target[1],
                "accuracy": most_common_target[1] / max(1, self.examples),
            },
            "generated_value_distribution": {
                "parsed_examples": sum(self.generated_value_counts.values()),
                "malformed_examples": self.examples - sum(self.generated_value_counts.values()),
                "top_values": [
                    {
                        "value": value,
                        "count": count,
                        "rate_over_all_examples": count / max(1, self.examples),
                        "rate_over_parsed_examples": count
                        / max(1, sum(self.generated_value_counts.values())),
                    }
                    for value, count in generated[:10]
                ],
            },
            "target_value_strata": {
                "target_equals_1": {
                    "examples": self.target_equal_one_examples,
                    "exact": self.target_equal_one_exact,
                    "exact_accuracy": self.target_equal_one_exact
                    / max(1, self.target_equal_one_examples),
                },
                "target_greater_than_1": {
                    "examples": self.target_greater_than_one_examples,
                    "exact": self.target_greater_than_one_exact,
                    "exact_accuracy": self.target_greater_than_one_exact
                    / max(1, self.target_greater_than_one_examples),
                },
            },
        }


def _padded_ids(
    rows: Sequence[Sequence[int]], *, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    if not rows:
        raise ValueError("cannot pad an empty batch")
    width = max(len(row) for row in rows)
    pad_id = TOKEN_TO_ID["<PAD>"]
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    lengths = torch.empty(len(rows), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        if not row:
            raise ValueError("token row cannot be empty")
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention[index, : len(row)] = True
        lengths[index] = len(row)
    return input_ids, attention, lengths


@torch.inference_mode()
def greedy_generate_batch(
    model: torch.nn.Module,
    prompts: Sequence[Sequence[str]],
    *,
    device: torch.device,
    max_new_tokens: int,
    max_seq_len: int,
) -> list[tuple[str, ...]]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    sequences = [[TOKEN_TO_ID[token] for token in prompt] for prompt in prompts]
    prompt_lengths = [len(sequence) for sequence in sequences]
    finished = [False] * len(sequences)
    eos_id = TOKEN_TO_ID["<EOS>"]
    for _ in range(max_new_tokens):
        active = [index for index, done in enumerate(finished) if not done]
        if not active:
            break
        if any(len(sequences[index]) >= max_seq_len for index in active):
            break
        input_ids, attention, lengths = _padded_ids(sequences, device=device)
        logits = model(input_ids=input_ids, attention_mask=attention)
        row_indices = torch.arange(len(sequences), device=device)
        next_ids = logits[row_indices, lengths - 1].argmax(dim=-1).tolist()
        for index in active:
            token_id = int(next_ids[index])
            sequences[index].append(token_id)
            if token_id == eos_id:
                finished[index] = True
    return [
        tuple(ID_TO_TOKEN[token_id] for token_id in sequence[prompt_length:])
        for sequence, prompt_length in zip(sequences, prompt_lengths, strict=True)
    ]


@torch.inference_mode()
def teacher_forced_batch(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> list[dict[str, float | int | bool]]:
    token_rows: list[list[int]] = []
    answer_starts: list[int] = []
    for record in records:
        tokens = record.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("record tokens must be a list of strings")
        prompt, _ = split_prompt_and_answer(tokens)
        token_rows.append([TOKEN_TO_ID[token] for token in tokens])
        answer_starts.append(len(prompt))
    input_ids, attention, _ = _padded_ids(token_rows, device=device)
    labels = torch.full_like(input_ids, -100)
    for index, (row, answer_start) in enumerate(zip(token_rows, answer_starts, strict=True)):
        labels[index, answer_start : len(row)] = input_ids[index, answer_start : len(row)]
    logits = model(input_ids=input_ids, attention_mask=attention)
    targets = labels[:, 1:]
    predictions = logits[:, :-1].argmax(dim=-1)
    supervised = targets.ne(-100)
    losses = F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        targets.contiguous().view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(targets.shape)
    results: list[dict[str, float | int | bool]] = []
    for index in range(len(records)):
        mask = supervised[index]
        correct = predictions[index].eq(targets[index]) & mask
        results.append(
            {
                "loss_sum": float(losses[index][mask].sum()),
                "tokens": int(mask.sum()),
                "token_correct": int(correct.sum()),
                "sequence_correct": bool(
                    mask.any() and predictions[index][mask].eq(targets[index][mask]).all()
                ),
            }
        )
    return results


def _metric_groups() -> dict[str, MetricAccumulator]:
    return defaultdict(MetricAccumulator)


def evaluate_records(
    model: torch.nn.Module,
    records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    max_seq_len: int,
    progress_every: int = 1_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups = _metric_groups()
    mistakes: list[dict[str, Any]] = []
    model.eval()
    processed = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts: list[tuple[str, ...]] = []
        expected_rows: list[tuple[str, ...]] = []
        for record in batch:
            prompt, expected = split_prompt_and_answer(record["tokens"])
            prompts.append(prompt)
            expected_rows.append(expected)
        teacher = teacher_forced_batch(model, batch, device=device)
        generated = greedy_generate_batch(
            model,
            prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
        )
        for record, expected, prediction, teacher_row in zip(
            batch, expected_rows, generated, teacher, strict=True
        ):
            task = str(record["task"])
            length = int(record["n_digits"])
            base100_width = input_base100_width(length)
            regime = generalization_regime(length)
            keys = (
                "overall",
                f"task::{task}",
                f"length::{length}",
                f"task_length::{task}::{length}",
                f"base100_width::{base100_width}",
                f"task_base100_width::{task}::{base100_width}",
                f"regime::{regime}",
                f"task_regime::{task}::{regime}",
            )
            update = {
                "expected_tokens": expected,
                "generated_tokens": prediction,
                "expected_value": int(record["answer"]),
                "teacher_loss_sum": float(teacher_row["loss_sum"]),
                "teacher_tokens": int(teacher_row["tokens"]),
                "teacher_token_correct": int(teacher_row["token_correct"]),
                "teacher_sequence_correct": bool(teacher_row["sequence_correct"]),
            }
            for key in keys:
                groups[key].update(**update)
            if parse_generated_answer(prediction) != int(record["answer"]) and len(mistakes) < 20:
                mistakes.append(
                    {
                        "id": record["id"],
                        "task": task,
                        "n_digits": length,
                        "inputs": record["inputs"],
                        "expected": list(expected),
                        "generated": list(prediction),
                        "parsed_generated": parse_generated_answer(prediction),
                    }
                )
        processed += len(batch)
        if progress_every > 0 and (
            processed == len(records) or processed // progress_every != (processed - len(batch)) // progress_every
        ):
            print(f"Evaluated {processed:,}/{len(records):,} examples", flush=True)

    by_task: dict[str, Any] = {}
    by_length: dict[str, Any] = {}
    by_task_length: dict[str, dict[str, Any]] = defaultdict(dict)
    by_base100_width: dict[str, Any] = {}
    by_task_base100_width: dict[str, dict[str, Any]] = defaultdict(dict)
    by_regime: dict[str, Any] = {}
    by_task_regime: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, accumulator in groups.items():
        if key == "overall":
            continue
        parts = key.split("::")
        if parts[0] == "task":
            by_task[parts[1]] = accumulator.summary()
        elif parts[0] == "length":
            by_length[parts[1]] = accumulator.summary()
        elif parts[0] == "task_length":
            by_task_length[parts[1]][parts[2]] = accumulator.summary()
        elif parts[0] == "base100_width":
            by_base100_width[parts[1]] = accumulator.summary()
        elif parts[0] == "task_base100_width":
            by_task_base100_width[parts[1]][parts[2]] = accumulator.summary()
        elif parts[0] == "regime":
            by_regime[parts[1]] = accumulator.summary()
        elif parts[0] == "task_regime":
            by_task_regime[parts[1]][parts[2]] = accumulator.summary()
        else:
            raise AssertionError(f"unknown metric group: {key}")
    return (
        {
            "overall": groups["overall"].summary(),
            "by_task": dict(sorted(by_task.items())),
            "by_digit_length": dict(sorted(by_length.items(), key=lambda item: int(item[0]))),
            "by_task_and_digit_length": {
                task: dict(sorted(values.items(), key=lambda item: int(item[0])))
                for task, values in sorted(by_task_length.items())
            },
            "by_input_base100_width": dict(
                sorted(by_base100_width.items(), key=lambda item: int(item[0]))
            ),
            "by_task_and_input_base100_width": {
                task: dict(sorted(values.items(), key=lambda item: int(item[0])))
                for task, values in sorted(by_task_base100_width.items())
            },
            "by_generalization_regime": dict(sorted(by_regime.items())),
            "by_task_and_generalization_regime": {
                task: dict(sorted(values.items()))
                for task, values in sorted(by_task_regime.items())
            },
        },
        mistakes,
    )


def _selected_completed_runs(
    config_path: Path, only: Iterable[str]
) -> tuple[ProductionRun, ...]:
    runs = build_matrix(config_path)
    lookup = {run.run_id: run for run in runs}
    requested = tuple(only)
    if requested:
        unknown = sorted(set(requested) - set(lookup))
        if unknown:
            raise ValueError(f"unknown run IDs: {', '.join(unknown)}")
        candidates = tuple(run for run in runs if run.run_id in set(requested))
    else:
        candidates = runs
    completed = tuple(
        run for run in candidates if completion_is_valid(run, config_path=config_path)
    )
    if requested and len(completed) != len(set(requested)):
        incomplete = sorted(set(requested) - {run.run_id for run in completed})
        raise ValueError(f"requested runs are not complete: {', '.join(incomplete)}")
    if not completed:
        raise ValueError("no completed runs are available for evaluation")
    return completed


def _result_is_current(
    path: Path,
    *,
    checkpoint_sha256: str,
    manifest_sha256: str,
    examples_per_group: int,
    max_new_tokens: int,
) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            value.get("checkpoint_sha256") == checkpoint_sha256
            and value.get("format_version") == FORMAT_VERSION
            and value.get("evaluation_manifest_sha256") == manifest_sha256
            and value.get("examples_per_task_per_length") == examples_per_group
            and value.get("max_new_tokens") == max_new_tokens
            and isinstance(value.get("metrics"), dict)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _print_diagnostic_summary(metrics: Mapping[str, Any]) -> None:
    """Print the compact task/regime view used for experiment handoff."""

    print("Diagnostic summary", flush=True)
    task_metrics = metrics.get("by_task", {})
    task_regimes = metrics.get("by_task_and_generalization_regime", {})
    for task in sorted(task_metrics):
        task_row = task_metrics[task]
        baseline = task_row["most_common_target_baseline"]
        print(
            f"  {task}: exact={task_row['autoregressive_exact_accuracy']:.2%} "
            f"well_formed={task_row['well_formed_rate']:.2%} "
            f"most_common_target={baseline['value']} "
            f"baseline={baseline['accuracy']:.2%}",
            flush=True,
        )
        for regime, row in sorted(task_regimes.get(task, {}).items()):
            print(
                f"    {regime}: examples={row['examples']} "
                f"exact={row['autoregressive_exact_accuracy']:.2%} "
                f"well_formed={row['well_formed_rate']:.2%}",
                flush=True,
            )
        if task == "greatest_common_divisor":
            strata = task_row["target_value_strata"]
            equals_one = strata["target_equals_1"]
            greater_one = strata["target_greater_than_1"]
            print(
                "    GCD strata: "
                f"target=1 exact={equals_one['exact_accuracy']:.2%} "
                f"({equals_one['exact']}/{equals_one['examples']}), "
                f"target>1 exact={greater_one['exact_accuracy']:.2%} "
                f"({greater_one['exact']}/{greater_one['examples']})",
                flush=True,
            )
            top = task_row["generated_value_distribution"]["top_values"]
            rendered = ", ".join(
                f"{entry['value']}:{entry['rate_over_all_examples']:.2%}"
                for entry in top[:5]
            )
            print(f"    top generated values (all-example rate): {rendered}", flush=True)


def evaluate_run_split(
    run: ProductionRun,
    *,
    split: str,
    manifest_path: Path,
    lengths: Sequence[int],
    shard_indices: Sequence[int] | None,
    config_path: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
    max_new_tokens: int,
    examples_per_task_per_length: int,
    progress_every: int,
    force: bool,
) -> Path:
    checkpoint_path = Path(run.output_dir) / "checkpoint.pt"
    checkpoint_sha256 = _sha256(checkpoint_path)
    manifest_sha256 = _sha256(manifest_path)
    destination = output_dir / f"{run.run_id}-{split}.json"
    if not force and _result_is_current(
        destination,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha256,
        examples_per_group=examples_per_task_per_length,
        max_new_tokens=max_new_tokens,
    ):
        print(f"Reusing current evaluation: {destination}", flush=True)
        return destination

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_config = TrainConfig.from_value(checkpoint["config"])
    model = build_integer_model(train_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    device = torch.device(device_name)
    model.to(device)
    records = load_balanced_records(
        manifest_path,
        tasks=run.tasks,
        lengths=lengths,
        examples_per_task_per_length=examples_per_task_per_length,
        shard_indices=shard_indices,
    )
    print(
        f"Evaluating {run.run_id} on {split}: {len(records):,} examples",
        flush=True,
    )
    metrics, mistakes = evaluate_records(
        model,
        records,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_seq_len=train_config.max_seq_len,
        progress_every=progress_every,
    )
    result = {
        "format_version": FORMAT_VERSION,
        "run": asdict(run),
        "split": split,
        "lengths": list(lengths),
        "examples_per_task_per_length": examples_per_task_per_length,
        "max_new_tokens": max_new_tokens,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": manifest_sha256,
        "metrics": metrics,
        "first_mistakes": mistakes,
        "diagnostic_definitions": {
            "magnitude_ood_familiar_token_width": (
                "outside the 1-5 decimal-digit training range but no wider than "
                "the three-token base-100 training maximum"
            ),
            "token_length_ood": (
                "requires more base-100 input tokens than any 1-5-digit operand"
            ),
            "most_common_target_baseline": (
                "accuracy obtained by always predicting the most frequent target "
                "inside the reported group"
            ),
        },
    }
    _atomic_json(result, destination)
    print(f"Saved: {destination}", flush=True)
    _print_diagnostic_summary(metrics)
    model.to("cpu")
    del model
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--split", choices=("iid", "ood", "both"), default="both")
    parser.add_argument("--iid-manifest", type=Path, default=DEFAULT_IID_MANIFEST)
    parser.add_argument("--ood-manifest", type=Path, default=DEFAULT_OOD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--examples-per-task-per-length", type=int, default=1_000)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--device", default=automatic_device())
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute matching evaluations instead of reusing their JSON files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = _selected_completed_runs(args.config, args.only)
    for run in runs:
        if args.split in {"iid", "both"}:
            evaluate_run_split(
                run,
                split="iid",
                manifest_path=args.iid_manifest,
                lengths=IID_LENGTHS,
                shard_indices=(99,),
                config_path=args.config,
                output_dir=args.output_dir,
                device_name=args.device,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                examples_per_task_per_length=args.examples_per_task_per_length,
                progress_every=args.progress_every,
                force=args.force,
            )
        if args.split in {"ood", "both"}:
            evaluate_run_split(
                run,
                split="ood",
                manifest_path=args.ood_manifest,
                lengths=OOD_LENGTHS,
                shard_indices=None,
                config_path=args.config,
                output_dir=args.output_dir,
                device_name=args.device,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                examples_per_task_per_length=args.examples_per_task_per_length,
                progress_every=args.progress_every,
                force=args.force,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MetricAccumulator",
    "evaluate_records",
    "greedy_generate_batch",
    "generalization_regime",
    "input_base100_width",
    "load_balanced_records",
    "main",
    "parse_generated_answer",
    "split_prompt_and_answer",
    "teacher_forced_batch",
]
