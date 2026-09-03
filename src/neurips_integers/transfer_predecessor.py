"""Few-shot predecessor transfer pilot for T1/T2/T4/T8 integer checkpoints.

The held-out ``<PREDECESSOR>`` token is appended to the existing integer
vocabulary. Existing token IDs and pretrained parameter rows are preserved.
The new row is randomly initialized and learned from the same few-shot data
for every source model. A randomly initialized model is trained as a control.
The checkpoint with the lowest validation loss is restored for final testing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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

from .evaluate import parse_generated_answer
from .passage import TOKEN_TO_ID as BASE_TOKEN_TO_ID
from .passage import encode_number
from .training import automatic_device


TRANSFER_FORMAT_VERSION = 1
TRANSFER_TASK = "predecessor"
TRANSFER_TASK_TOKEN = "<PREDECESSOR>"
SOURCE_TASK_COUNTS = (1, 2, 4, 8)
DEFAULT_RUN_ROOT = Path("runs/henry-integer-v1")
DEFAULT_OUTPUT_ROOT = Path("runs/henry-integer-transfer-v1")
DEFAULT_RESULT_ROOT = Path("results/integer-transfer-v1")


def transfer_vocabulary() -> tuple[dict[str, int], dict[int, str]]:
    """Append the held-out token without changing any existing token ID."""

    token_to_id = dict(BASE_TOKEN_TO_ID)
    if TRANSFER_TASK_TOKEN in token_to_id:
        raise RuntimeError("predecessor token unexpectedly exists in base vocabulary")
    token_to_id[TRANSFER_TASK_TOKEN] = len(token_to_id)
    id_to_token = {index: token for token, index in token_to_id.items()}
    return token_to_id, id_to_token


TOKEN_TO_ID, ID_TO_TOKEN = transfer_vocabulary()


def decimal_digits(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("value must be a nonnegative integer")
    return len(str(value))


def predecessor_tokens(value: int) -> tuple[str, ...]:
    """Render one answer-supervised predecessor example."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("predecessor input must be a positive integer")
    size = decimal_digits(value)
    return (
        "<BOS>",
        "<SIZE>",
        *encode_number(size),
        *encode_number(value),
        TRANSFER_TASK_TOKEN,
        "=",
        *encode_number(value - 1),
        "<EOS>",
    )


def predecessor_record(record_id: str, value: int) -> dict[str, Any]:
    return {
        "schema_version": "integer-transfer/predecessor-v1",
        "id": record_id,
        "task": TRANSFER_TASK,
        "n_digits": decimal_digits(value),
        "inputs": {"primary": value},
        "answer": value - 1,
        "answer_kind": "integer",
        "tokens": list(predecessor_tokens(value)),
        "canonical_text": " ".join(predecessor_tokens(value)),
    }


def _range_for_digits(digits: int) -> tuple[int, int]:
    if digits < 1:
        raise ValueError("digits must be positive")
    return (1, 9) if digits == 1 else (10 ** (digits - 1), 10**digits - 1)


def _sample_unique(
    *,
    digits: int,
    count: int,
    rng: random.Random,
    excluded: set[int],
) -> list[int]:
    low, high = _range_for_digits(digits)
    available = high - low + 1 - sum(low <= value <= high for value in excluded)
    count = min(count, available)
    if count <= 0:
        return []
    # Small domains are easiest and safest to sample exhaustively.
    if high - low + 1 <= 100_000:
        choices = [value for value in range(low, high + 1) if value not in excluded]
        return rng.sample(choices, count)
    result: set[int] = set()
    while len(result) < count:
        candidate = rng.randint(low, high)
        if candidate not in excluded:
            result.add(candidate)
    return sorted(result)


def build_splits(
    *,
    shots: int = 20,
    validation_per_length: int = 20,
    test_per_length: int = 100,
    seed: int = 20_260_902,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic, input-disjoint train/validation/IID/OOD splits."""

    if shots < 1:
        raise ValueError("shots must be positive")
    if validation_per_length < 1 or test_per_length < 1:
        raise ValueError("evaluation counts must be positive")
    rng = random.Random(seed)
    used: set[int] = set()
    split_values: dict[str, list[int]] = {"train": [], "validation": [], "iid": [], "ood": []}

    # The positive one-digit domain has only nine distinct values. Reserve
    # enough of it for disjoint validation and testing instead of consuming
    # almost the whole domain during few-shot training.
    allocation = {digits: 0 for digits in range(1, 6)}
    allocation[1] = min(1, shots)
    remaining = shots - allocation[1]
    for index, digits in enumerate(range(2, 6)):
        allocation[digits] = remaining // 4 + int(index < remaining % 4)
    for digits in range(1, 6):
        count = allocation[digits]
        values = _sample_unique(digits=digits, count=count, rng=rng, excluded=used)
        split_values["train"].extend(values)
        used.update(values)

    for split, count in (("validation", validation_per_length), ("iid", test_per_length)):
        for digits in range(1, 6):
            # After the single one-digit training example, reserve two of the
            # eight remaining values for validation and six for IID testing.
            # Larger domains use the requested balanced count.
            requested = (2 if split == "validation" else 6) if digits == 1 else count
            values = _sample_unique(
                digits=digits,
                count=requested,
                rng=rng,
                excluded=used,
            )
            split_values[split].extend(values)
            used.update(values)

    for digits in range(6, 11):
        values = _sample_unique(
            digits=digits,
            count=test_per_length,
            rng=rng,
            excluded=used,
        )
        split_values["ood"].extend(values)
        used.update(values)

    return {
        split: [predecessor_record(f"{split}-{index:06d}", value) for index, value in enumerate(values)]
        for split, values in split_values.items()
    }


def _build_model(config: TrainConfig) -> torch.nn.Module:
    values: dict[str, Any] = {
        "model_type": config.architecture,
        "vocab_size": len(TOKEN_TO_ID),
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
    """Copy a base-vocabulary checkpoint into a one-row-expanded model."""

    target = model.state_dict()
    expanded = {"token_embedding.weight", "lm_head.weight"}
    for name, source_value in pretrained_state.items():
        if name not in target:
            raise ValueError(f"checkpoint contains unknown parameter {name!r}")
        target_value = target[name]
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
        elif name in expanded and source_value.ndim == 2:
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
) -> tuple[torch.nn.Module, TrainConfig, str | None]:
    """Load T1/T2/T4/T8 or create the matched random control."""

    if source == "random":
        reference = run_root / f"transformer-tasks08-seed{seed}" / "checkpoint.pt"
        checkpoint = torch.load(reference, map_location="cpu", weights_only=True)
        config = TrainConfig.from_value(checkpoint["config"])
        torch.manual_seed(seed)
        return _build_model(config), config, None

    task_count = int(source.removeprefix("t"))
    if task_count not in SOURCE_TASK_COUNTS:
        raise ValueError("source must be one of t1, t2, t4, t8, or random")
    path = run_root / f"transformer-tasks{task_count:02d}-seed{seed}" / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = TrainConfig.from_value(checkpoint["config"])
    # Give every pretrained source the same newly initialized task-token row
    # and the same fine-tuning dropout random stream.
    torch.manual_seed(seed)
    model = _build_model(config)
    copy_pretrained_weights(model, checkpoint["model"])
    return model, config, str(path)


def _encode_batch(
    records: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    rows = [[TOKEN_TO_ID[token] for token in record["tokens"]] for record in records]
    width = max(map(len, rows))
    inputs = torch.full(
        (len(rows), width), TOKEN_TO_ID["<PAD>"], dtype=torch.long, device=device
    )
    attention = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    labels = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        inputs[index, : len(row)] = torch.tensor(row, device=device)
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


@torch.inference_mode()
def _generate(
    model: torch.nn.Module,
    prompt: Sequence[str],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, ...]:
    ids = [TOKEN_TO_ID[token] for token in prompt]
    eos = TOKEN_TO_ID["<EOS>"]
    for _ in range(max_new_tokens):
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        attention = torch.ones_like(inputs, dtype=torch.bool)
        logits = model(input_ids=inputs, attention_mask=attention)
        next_id = int(logits[0, -1].argmax())
        ids.append(next_id)
        if next_id == eos:
            break
    return tuple(ID_TO_TOKEN[index] for index in ids[len(prompt) :])


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    model.eval()
    exact = 0
    well_formed = 0
    loss_sum = 0.0
    by_length: dict[int, list[int]] = {}
    for record in records:
        tokens = list(record["tokens"])
        equals = tokens.index("=")
        generated = _generate(
            model,
            tokens[: equals + 1],
            device=device,
            max_new_tokens=max_new_tokens,
        )
        parsed = parse_generated_answer(generated)
        correct = parsed == int(record["answer"])
        exact += int(correct)
        well_formed += int(parsed is not None)
        digits = int(record["n_digits"])
        counts = by_length.setdefault(digits, [0, 0])
        counts[0] += int(correct)
        counts[1] += 1
    # Teacher-forced loss is reported separately from generation accuracy.
    for start in range(0, len(records), 32):
        batch_records = records[start : start + 32]
        batch = _encode_batch(batch_records, device)
        supervised = int(batch[2][:, 1:].ne(-100).sum())
        loss_sum += float(_loss(model, batch)) * supervised
    tokens = sum(
        len(record["tokens"]) - record["tokens"].index("=") - 1
        for record in records
    )
    return {
        "examples": len(records),
        "exact_accuracy": exact / max(1, len(records)),
        "well_formed_rate": well_formed / max(1, len(records)),
        "teacher_forced_loss": loss_sum / max(1, tokens),
        "exact_by_digit_length": {
            str(length): {"exact": counts[0], "examples": counts[1], "accuracy": counts[0] / counts[1]}
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
    source: str,
    *,
    seed: int,
    shots: int,
    steps: int,
    learning_rate: float,
    batch_size: int,
    evaluation_every: int,
    run_root: Path,
    output_root: Path,
    result_root: Path,
    device_name: str,
) -> Path:
    splits = build_splits(shots=shots, seed=20_260_902)
    model, config, source_checkpoint = load_source_model(source, seed=seed, run_root=run_root)
    device = torch.device(device_name)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = random.Random(seed)
    curve: list[dict[str, Any]] = []
    output_dir = output_root / TRANSFER_TASK / "transformer" / f"seed{seed}" / source
    output_dir.mkdir(parents=True, exist_ok=True)
    best_state_path = output_dir / "best_model_state.pt"
    best_validation_loss = float("inf")
    best_step = -1

    for step in range(steps + 1):
        if step % evaluation_every == 0 or step == steps:
            validation = evaluate(model, splits["validation"], device=device)
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
        chosen = [splits["train"][rng.randrange(len(splits["train"]))] for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, _encode_batch(chosen, device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.load_state_dict(
        torch.load(best_state_path, map_location="cpu", weights_only=True), strict=True
    )
    iid = evaluate(model, splits["iid"], device=device)
    ood = evaluate(model, splits["ood"], device=device)
    torch.save(
        {
            "format_version": TRANSFER_FORMAT_VERSION,
            "model": model.state_dict(),
            "source": source,
            "source_checkpoint": source_checkpoint,
            "seed": seed,
            "shots": shots,
            "steps": steps,
            "best_step": best_step,
            "best_validation_loss": best_validation_loss,
            "learning_rate": learning_rate,
            "vocabulary": TOKEN_TO_ID,
            "base_config": asdict(config),
        },
        output_dir / "checkpoint.pt",
    )
    best_state_path.unlink(missing_ok=True)
    result = {
        "format_version": TRANSFER_FORMAT_VERSION,
        "task": TRANSFER_TASK,
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
    destination = result_root / TRANSFER_TASK / f"transformer-seed{seed}-{source}.json"
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
    parser.add_argument("--run", action="store_true", help="execute the selected transfer runs")
    parser.add_argument(
        "--source",
        action="append",
        choices=("t1", "t2", "t4", "t8", "random"),
        default=[],
        help="starting checkpoint; repeat to select multiple (default: all five)",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-every", type=int, default=50)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default=automatic_device())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = tuple(args.source) or ("t1", "t2", "t4", "t8", "random")
    plan = {
        "task": TRANSFER_TASK,
        "architecture": "transformer",
        "sources": sources,
        "seed": args.seed,
        "shots": args.shots,
        "steps": args.steps,
        "selection_rule": "lowest validation teacher-forced loss",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        print("Dry run only. Add --run to execute.")
        return 0
    for source in sources:
        run_source(
            source,
            seed=args.seed,
            shots=args.shots,
            steps=args.steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            evaluation_every=args.evaluation_every,
            run_root=args.run_root,
            output_root=args.output_root,
            result_root=args.result_root,
            device_name=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
