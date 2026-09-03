"""Few-shot transfer for the LCM, modular-addition, and sorting holdouts.

This experiment compares T1, T2, T4, T8, and a matched randomly initialized
Transformer. Every source receives the same 20 examples and optimization
schedule. A task token is appended without changing any pretrained token ID.
The checkpoint with the lowest validation loss is restored for final testing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW

from neurips_permutations.models import build_model
from neurips_permutations.training import TrainConfig

from .passage import TOKEN_TO_ID as BASE_TOKEN_TO_ID
from .passage import decode_number, encode_number
from . import math_ops as ops
from .training import automatic_device


FORMAT_VERSION = 1
SOURCE_TASK_COUNTS = (1, 2, 4, 8)
TASK_TOKENS = {
    "least_common_multiple": "<LEAST_COMMON_MULTIPLE>",
    "modular_addition": "<MODULAR_ADDITION>",
    "sort_ascending": "<SORT_ASCENDING>",
}
DEFAULT_RUN_ROOT = Path("runs/henry-integer-v1")
DEFAULT_OUTPUT_ROOT = Path("runs/henry-integer-transfer-v1")
DEFAULT_RESULT_ROOT = Path("results/integer-transfer-v1")
DATA_SEED = 20_260_902


def task_vocabulary(task: str) -> tuple[dict[str, int], dict[int, str]]:
    try:
        task_token = TASK_TOKENS[task]
    except KeyError:
        raise ValueError(f"unsupported transfer task {task!r}") from None
    token_to_id = dict(BASE_TOKEN_TO_ID)
    if task_token in token_to_id:
        raise RuntimeError(f"{task_token} unexpectedly exists in base vocabulary")
    token_to_id[task_token] = len(token_to_id)
    return token_to_id, {index: token for token, index in token_to_id.items()}


def _number_list(values: Sequence[int]) -> tuple[str, ...]:
    if not values:
        raise ValueError("integer list must be nonempty")
    tokens: list[str] = ["<LIST_START>"]
    for index, value in enumerate(values):
        if index:
            tokens.append(",")
        tokens.extend(encode_number(value))
    tokens.append("<LIST_END>")
    return tuple(tokens)


def build_record(record_id: str, task: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    task_token = TASK_TOKENS[task]
    if task == "least_common_multiple":
        left, right = int(inputs["primary"]), int(inputs["operand"])
        if left <= 0 or right <= 0:
            raise ValueError("LCM operands must be positive")
        digits = max(len(str(left)), len(str(right)))
        answer: int | list[int] = math.lcm(left, right)
        input_tokens = (
            *encode_number(left),
            "<ARG_START>",
            *encode_number(right),
            "<ARG_END>",
        )
        answer_tokens = encode_number(answer)
    elif task == "modular_addition":
        left = int(inputs["primary"])
        right = int(inputs["operand"])
        modulus = int(inputs["modulus"])
        if left < 0 or right < 0:
            raise ValueError("modular-addition operands must be nonnegative")
        if modulus <= 0:
            raise ValueError("modular-addition modulus must be positive")
        digits = max(len(str(left)), len(str(right)), len(str(modulus)))
        answer = ops.modular_addition(left, right, modulus)
        input_tokens = (
            *encode_number(left),
            "<ARG_START>",
            *encode_number(right),
            "<ARG_END>",
            "<ARG_START>",
            *encode_number(modulus),
            "<ARG_END>",
        )
        answer_tokens = encode_number(answer)
    elif task == "sort_ascending":
        values = tuple(int(value) for value in inputs["values"])
        if not values or min(values) < 0:
            raise ValueError("sort values must be nonempty and nonnegative")
        digits = max(len(str(value)) for value in values)
        answer = sorted(values)
        input_tokens = _number_list(values)
        answer_tokens = _number_list(answer)
    else:
        raise ValueError(f"unsupported transfer task {task!r}")

    tokens = (
        "<BOS>",
        "<SIZE>",
        *encode_number(digits),
        *input_tokens,
        task_token,
        "=",
        *answer_tokens,
        "<EOS>",
    )
    return {
        "schema_version": f"integer-transfer/{task}-v1",
        "id": record_id,
        "task": task,
        "n_digits": digits,
        "inputs": dict(inputs),
        "answer": answer,
        "answer_kind": "integer" if task != "sort_ascending" else "integer_list",
        "tokens": list(tokens),
        "canonical_text": " ".join(tokens),
    }


def _range_for_digits(digits: int, *, positive: bool) -> tuple[int, int]:
    if digits < 1:
        raise ValueError("digits must be positive")
    if digits == 1:
        return (1 if positive else 0, 9)
    return 10 ** (digits - 1), 10**digits - 1


def _sample_case(
    task: str,
    *,
    digits: int,
    rng: random.Random,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if task == "least_common_multiple":
        low, high = _range_for_digits(digits, positive=True)
        left, right = rng.randint(low, high), rng.randint(low, high)
        # LCM is symmetric, so reversed operands share one leakage signature.
        return {"primary": left, "operand": right}, tuple(sorted((left, right)))
    if task == "modular_addition":
        low, high = _range_for_digits(digits, positive=False)
        positive_low, _ = _range_for_digits(digits, positive=True)
        left, right = rng.randint(low, high), rng.randint(low, high)
        modulus = rng.randint(positive_low, high)
        # Modular addition is symmetric in its two addends, but not its modulus.
        addends = tuple(sorted((left, right)))
        return (
            {"primary": left, "operand": right, "modulus": modulus},
            (*addends, modulus),
        )
    if task == "sort_ascending":
        low, high = _range_for_digits(digits, positive=False)
        length = rng.randint(3, 6)
        values = [rng.randint(low, high) for _ in range(length)]
        # Different permutations of the same multiset share one signature.
        return {"values": values}, tuple(sorted(values))
    raise ValueError(f"unsupported transfer task {task!r}")


def _sample_unique_cases(
    task: str,
    *,
    split: str,
    digits: int,
    count: int,
    rng: random.Random,
    used: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(10_000, count * 1_000)
    while len(records) < count and attempts < max_attempts:
        attempts += 1
        inputs, signature = _sample_case(task, digits=digits, rng=rng)
        key = (task, digits, *signature)
        if key in used:
            continue
        used.add(key)
        records.append(build_record(f"{split}-{digits}d-{len(records):06d}", task, inputs))
    if len(records) != count:
        raise ValueError(
            f"could generate only {len(records)}/{count} unique {task} cases "
            f"for {digits}-digit {split}"
        )
    return records


def build_splits(
    task: str,
    *,
    shots: int = 20,
    validation_per_length: int = 20,
    test_per_length: int = 100,
    seed: int = DATA_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic train, validation, IID, and OOD data."""

    if shots < 5:
        raise ValueError("shots must be at least five for length stratification")
    rng = random.Random(seed)
    used: set[tuple[Any, ...]] = set()
    result = {"train": [], "validation": [], "iid": [], "ood": []}

    for index, digits in enumerate(range(1, 6)):
        count = shots // 5 + int(index < shots % 5)
        result["train"].extend(
            _sample_unique_cases(
                task, split="train", digits=digits, count=count, rng=rng, used=used
            )
        )

    for split, count in (("validation", validation_per_length), ("iid", test_per_length)):
        for digits in range(1, 6):
            requested = count
            if task == "least_common_multiple" and digits == 1:
                # There are only 45 unordered positive one-digit pairs.
                requested = 10 if split == "validation" else 31
            result[split].extend(
                _sample_unique_cases(
                    task,
                    split=split,
                    digits=digits,
                    count=requested,
                    rng=rng,
                    used=used,
                )
            )

    for digits in range(6, 11):
        result["ood"].extend(
            _sample_unique_cases(
                task,
                split="ood",
                digits=digits,
                count=test_per_length,
                rng=rng,
                used=used,
            )
        )
    return result


def _build_model(config: TrainConfig, vocab_size: int) -> torch.nn.Module:
    values: dict[str, Any] = {
        "model_type": config.architecture,
        "vocab_size": vocab_size,
        "max_seq_len": config.max_seq_len,
        "d_model": config.d_model,
        "layers": config.num_layers or (4 if config.architecture == "transformer" else 1),
        "dropout": config.dropout,
        "mlp_ratio": config.mlp_ratio,
        "tie_embeddings": config.tie_embeddings,
    }
    if config.architecture == "transformer":
        values["n_heads"] = config.num_heads
    return build_model(**values)


def copy_pretrained_weights(
    model: torch.nn.Module,
    pretrained_state: Mapping[str, Tensor],
) -> None:
    target = model.state_dict()
    expandable = {"token_embedding.weight", "lm_head.weight"}
    for name, source_value in pretrained_state.items():
        if name not in target:
            raise ValueError(f"checkpoint contains unknown parameter {name!r}")
        target_value = target[name]
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
        elif name in expandable and source_value.ndim == 2:
            if source_value.shape[0] != len(BASE_TOKEN_TO_ID):
                raise ValueError(f"unexpected base-vocabulary size for {name}")
            if source_value.shape[1:] != target_value.shape[1:]:
                raise ValueError(f"incompatible embedding width for {name}")
            target_value[: source_value.shape[0]].copy_(source_value)
        else:
            raise ValueError(
                f"shape mismatch for {name}: checkpoint={tuple(source_value.shape)}, "
                f"transfer={tuple(target_value.shape)}"
            )
    model.load_state_dict(target, strict=True)


def load_source_model(
    source: str,
    *,
    seed: int,
    run_root: Path,
    vocab_size: int,
) -> tuple[torch.nn.Module, TrainConfig, str | None]:
    if source == "random":
        path = run_root / f"transformer-tasks08-seed{seed}" / "checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = TrainConfig.from_value(checkpoint["config"])
        torch.manual_seed(seed)
        return _build_model(config, vocab_size), config, None

    task_count = int(source.removeprefix("t"))
    if task_count not in SOURCE_TASK_COUNTS:
        raise ValueError("source must be t1, t2, t4, t8, or random")
    path = run_root / f"transformer-tasks{task_count:02d}-seed{seed}" / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = TrainConfig.from_value(checkpoint["config"])
    # Reset before construction so every source receives the same new-token row
    # and the same fine-tuning dropout random stream.
    torch.manual_seed(seed)
    model = _build_model(config, vocab_size)
    copy_pretrained_weights(model, checkpoint["model"])
    return model, config, str(path)


def _encode_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    token_to_id: Mapping[str, int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    rows = [[token_to_id[token] for token in record["tokens"]] for record in records]
    width = max(map(len, rows))
    inputs = torch.full(
        (len(rows), width), token_to_id["<PAD>"], dtype=torch.long, device=device
    )
    attention = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    labels = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        inputs[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention[index, : len(row)] = True
        equals = records[index]["tokens"].index("=")
        labels[index, equals + 1 : len(row)] = inputs[index, equals + 1 : len(row)]
    return inputs, attention, labels


def _loss(model: torch.nn.Module, batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    inputs, attention, labels = batch
    logits = model(input_ids=inputs, attention_mask=attention)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _parse_integer(tokens: Sequence[str]) -> int | None:
    if not tokens or tokens[-1] != "<EOS>":
        return None
    try:
        value, index = decode_number(tokens[:-1])
    except (TypeError, ValueError):
        return None
    return value if index == len(tokens) - 1 else None


def _parse_integer_list(tokens: Sequence[str]) -> list[int] | None:
    if (
        len(tokens) < 3
        or tokens[0] != "<LIST_START>"
        or tuple(tokens[-2:]) != ("<LIST_END>", "<EOS>")
    ):
        return None
    values: list[int] = []
    index = 1
    end = len(tokens) - 2
    try:
        while index < end:
            value, index = decode_number(tokens, index)
            values.append(value)
            if index < end:
                if tokens[index] != ",":
                    return None
                index += 1
    except (TypeError, ValueError):
        return None
    return values if values and index == end else None


def parse_answer(task: str, tokens: Sequence[str]) -> int | list[int] | None:
    if task in {"least_common_multiple", "modular_addition"}:
        return _parse_integer(tokens)
    if task == "sort_ascending":
        return _parse_integer_list(tokens)
    raise ValueError(f"unsupported transfer task {task!r}")


@torch.inference_mode()
def _generate(
    model: torch.nn.Module,
    prompt: Sequence[str],
    *,
    token_to_id: Mapping[str, int],
    id_to_token: Mapping[int, str],
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, ...]:
    ids = [token_to_id[token] for token in prompt]
    prompt_length = len(ids)
    eos = token_to_id["<EOS>"]
    for _ in range(max_new_tokens):
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        attention = torch.ones_like(inputs, dtype=torch.bool)
        logits = model(input_ids=inputs, attention_mask=attention)
        next_id = int(logits[0, -1].argmax())
        ids.append(next_id)
        if next_id == eos:
            break
    return tuple(id_to_token[index] for index in ids[prompt_length:])


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    token_to_id: Mapping[str, int],
    id_to_token: Mapping[int, str],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    model.eval()
    exact = well_formed = token_matches = token_slots = 0
    loss_sum = 0.0
    supervised_tokens = 0
    by_length: dict[int, list[int]] = {}
    for record in records:
        tokens = list(record["tokens"])
        equals = tokens.index("=")
        expected = tokens[equals + 1 :]
        generated = _generate(
            model,
            tokens[: equals + 1],
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        parsed = parse_answer(task, generated)
        answer = record["answer"]
        correct = parsed == answer
        exact += int(correct)
        well_formed += int(parsed is not None)
        token_matches += sum(left == right for left, right in zip(expected, generated))
        token_slots += max(len(expected), len(generated))
        digits = int(record["n_digits"])
        counts = by_length.setdefault(digits, [0, 0])
        counts[0] += int(correct)
        counts[1] += 1

    for start in range(0, len(records), 32):
        batch = _encode_batch(
            records[start : start + 32], token_to_id=token_to_id, device=device
        )
        count = int(batch[2][:, 1:].ne(-100).sum())
        loss_sum += float(_loss(model, batch)) * count
        supervised_tokens += count
    return {
        "examples": len(records),
        "exact_accuracy": exact / max(1, len(records)),
        "token_accuracy": token_matches / max(1, token_slots),
        "well_formed_rate": well_formed / max(1, len(records)),
        "teacher_forced_loss": loss_sum / max(1, supervised_tokens),
        "exact_by_digit_length": {
            str(length): {
                "exact": counts[0],
                "examples": counts[1],
                "accuracy": counts[0] / counts[1],
            }
            for length, counts in sorted(by_length.items())
        },
    }


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def run_source(
    task: str,
    source: str,
    *,
    seed: int,
    shots: int,
    steps: int,
    learning_rate: float,
    batch_size: int,
    evaluation_every: int,
    max_new_tokens: int,
    run_root: Path,
    output_root: Path,
    result_root: Path,
    device_name: str,
) -> Path:
    splits = build_splits(task, shots=shots)
    token_to_id, id_to_token = task_vocabulary(task)
    model, config, source_checkpoint = load_source_model(
        source, seed=seed, run_root=run_root, vocab_size=len(token_to_id)
    )
    device = torch.device(device_name)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = random.Random(seed)
    curve: list[dict[str, Any]] = []
    output_dir = output_root / task / "transformer" / f"seed{seed}" / source
    output_dir.mkdir(parents=True, exist_ok=True)
    best_state_path = output_dir / "best_model_state.pt"
    best_validation_loss = float("inf")
    best_step = -1

    for step in range(steps + 1):
        if step % evaluation_every == 0 or step == steps:
            validation = evaluate(
                model,
                splits["validation"],
                task=task,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            curve.append({"step": step, **validation})
            print(
                f"{source} step={step} val_loss={validation['teacher_forced_loss']:.6f} "
                f"val_exact={validation['exact_accuracy']:.2%}",
                flush=True,
            )
            if validation["teacher_forced_loss"] < best_validation_loss:
                best_validation_loss = float(validation["teacher_forced_loss"])
                best_step = step
                torch.save(model.state_dict(), best_state_path)
        if step == steps:
            break
        model.train()
        chosen = [
            splits["train"][rng.randrange(len(splits["train"]))]
            for _ in range(batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(
            model,
            _encode_batch(chosen, token_to_id=token_to_id, device=device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.load_state_dict(
        torch.load(best_state_path, map_location="cpu", weights_only=True), strict=True
    )
    iid = evaluate(
        model,
        splits["iid"],
        task=task,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    ood = evaluate(
        model,
        splits["ood"],
        task=task,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model": model.state_dict(),
            "task": task,
            "source": source,
            "source_checkpoint": source_checkpoint,
            "seed": seed,
            "shots": shots,
            "steps": steps,
            "best_step": best_step,
            "best_validation_loss": best_validation_loss,
            "learning_rate": learning_rate,
            "vocabulary": token_to_id,
            "base_config": asdict(config),
        },
        checkpoint_path,
    )
    best_state_path.unlink(missing_ok=True)
    result = {
        "format_version": FORMAT_VERSION,
        "task": task,
        "architecture": "transformer",
        "source": source,
        "source_checkpoint": source_checkpoint,
        "seed": seed,
        "shots": shots,
        "steps": steps,
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "selection_rule": "lowest validation teacher-forced loss",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "split_examples": {name: len(rows) for name, rows in splits.items()},
        "validation_curve": curve,
        "iid": iid,
        "ood": ood,
    }
    destination = result_root / task / f"transformer-seed{seed}-{source}.json"
    _atomic_json(result, destination)
    print(
        f"{source}: best_step={best_step}; IID exact={iid['exact_accuracy']:.2%}; "
        f"OOD exact={ood['exact_accuracy']:.2%}; saved {destination}",
        flush=True,
    )
    model.to("cpu")
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_TOKENS), required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--source",
        action="append",
        choices=("t1", "t2", "t4", "t8", "random"),
        default=[],
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-every", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default=automatic_device())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = tuple(args.source) or ("t1", "t2", "t4", "t8", "random")
    plan = {
        "task": args.task,
        "architecture": "transformer",
        "sources": sources,
        "seed": args.seed,
        "shots": args.shots,
        "steps": args.steps,
        "selection_rule": "lowest validation teacher-forced loss",
        "device": args.device,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        print("Dry run only. Add --run to execute.")
        return 0
    for source in sources:
        run_source(
            args.task,
            source,
            seed=args.seed,
            shots=args.shots,
            steps=args.steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            evaluation_every=args.evaluation_every,
            max_new_tokens=args.max_new_tokens,
            run_root=args.run_root,
            output_root=args.output_root,
            result_root=args.result_root,
            device_name=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
