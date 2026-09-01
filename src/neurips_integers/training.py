"""Integer adapter for Xuanyu's shared streaming training implementation.

Only the vocabulary and model output size are domain-specific. Data streaming,
answer-only masking, batching, optimization, validation, checkpointing, and
the Causal Transformer/MLP classes remain in ``neurips_permutations``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from neurips_permutations import training as shared_training
from neurips_permutations.models import build_model

from .generate import TASK_NAMES
from .passage import TOKEN_TO_ID


NESTED_TASKS: Mapping[int, tuple[str, ...]] = {
    size: TASK_NAMES[:size] for size in (1, 2, 4, 8)
}


def automatic_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _integer_model_factory(
    config: shared_training.TrainConfig,
) -> torch.nn.Module:
    values = dict(config.model_config)
    requested_vocab = values.pop("vocab_size", len(TOKEN_TO_ID))
    if requested_vocab != len(TOKEN_TO_ID):
        raise ValueError("model_config vocab_size disagrees with integer vocabulary")
    values.update(
        model_type=config.architecture,
        vocab_size=len(TOKEN_TO_ID),
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        layers=(
            config.num_layers
            if config.num_layers is not None
            else (4 if config.architecture == "transformer" else 1)
        ),
        dropout=config.dropout,
        mlp_ratio=config.mlp_ratio,
        tie_embeddings=config.tie_embeddings,
    )
    if config.architecture == "transformer":
        values["n_heads"] = config.num_heads
    return build_model(**values)


def build_integer_model(
    config: shared_training.TrainConfig | Mapping[str, Any],
) -> torch.nn.Module:
    """Build the integer model described by a serialized training config."""

    return _integer_model_factory(shared_training.TrainConfig.from_value(config))


def train_integer_run(
    run_config: shared_training.TrainConfig | Mapping[str, Any],
    *,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Run the shared trainer with integer vocabulary and task validation."""

    config = shared_training.TrainConfig.from_value(run_config)
    if config.tasks is None or not config.tasks:
        raise ValueError("integer training requires an explicit nested task subset")
    unknown = set(config.tasks) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"unknown integer training tasks: {sorted(unknown)}")
    if config.validation_tasks is None:
        config = replace(config, validation_tasks=config.tasks)

    # The upstream trainer's only domain-specific global is the collator's
    # token-to-ID mapping. Each formal run is one process, so temporarily
    # swapping it is isolated and lets every other upstream component remain
    # byte-for-byte unchanged.
    original_token_to_id = shared_training.TOKEN_TO_ID
    shared_training.TOKEN_TO_ID = TOKEN_TO_ID
    try:
        return shared_training.train_run(
            config,
            model_factory=_integer_model_factory,
            stop_after_steps=stop_after_steps,
        )
    finally:
        shared_training.TOKEN_TO_ID = original_token_to_id


def smoke_config(
    *,
    architecture: str,
    task_count: int,
    manifest: str = "data/integer-20k-pilot/manifest.json",
    output_root: str = "runs/integer-pilot-smoke",
    seed: int = 17,
    max_steps: int = 2,
    device: str | None = None,
) -> shared_training.TrainConfig:
    """Return one two-step, full-dimension pipeline-check configuration."""

    if architecture not in {"transformer", "mlp"}:
        raise ValueError("architecture must be transformer or mlp")
    try:
        tasks = NESTED_TASKS[task_count]
    except KeyError:
        raise ValueError("task_count must be one of 1, 2, 4, or 8") from None
    run_id = f"{architecture}-tasks{task_count:02d}-seed{seed}"
    return shared_training.TrainConfig(
        manifest=manifest,
        output_dir=str(Path(output_root) / run_id),
        architecture=architecture,
        d_model=256,
        num_layers=4 if architecture == "transformer" else 1,
        num_heads=8,
        dropout=0.1,
        mlp_ratio=4.0,
        tie_embeddings=True,
        tasks=tasks,
        validation_tasks=tasks,
        seed=seed,
        max_steps=max_steps,
        batch_size=4,
        validation_batch_size=4,
        gradient_accumulation_steps=1,
        max_seq_len=1_024,
        max_tokens_per_batch=4_096,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=1,
        min_lr_ratio=0.1,
        max_grad_norm=1.0,
        shuffle_buffer_size=1_000,
        num_workers=0,
        checkpoint_every=1,
        validate_every=1,
        validation_batches_per_task=1,
        device=device or automatic_device(),
        amp=False,
        bf16=False,
        shard_indices=tuple(range(98)),
        validation_shard_indices=(98,),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("transformer", "mlp"), required=True)
    parser.add_argument("--task-count", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--manifest", default="data/integer-20k-pilot/manifest.json")
    parser.add_argument("--output-root", default="runs/integer-pilot-smoke")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    config = smoke_config(
        architecture=args.architecture,
        task_count=args.task_count,
        manifest=args.manifest,
        output_root=args.output_root,
        seed=args.seed,
        max_steps=args.max_steps,
        device=args.device,
    )
    if args.print_config:
        from dataclasses import asdict

        print(json.dumps(asdict(config), indent=2, sort_keys=True))
        return 0
    summary = train_integer_run(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NESTED_TASKS",
    "automatic_device",
    "build_integer_model",
    "smoke_config",
    "train_integer_run",
]
