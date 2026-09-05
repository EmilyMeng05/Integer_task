"""T16 adapter for Xuanyu's shared streaming training implementation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

from neurips_permutations import training as shared_training
from neurips_permutations.models import build_model

from .generate_t16 import TASK_NAMES
from .passage_t16 import TOKEN_TO_ID


T16_TASKS: tuple[str, ...] = tuple(TASK_NAMES)
NESTED_TASKS: Mapping[int, tuple[str, ...]] = {16: T16_TASKS}


def automatic_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _t16_model_factory(config: shared_training.TrainConfig) -> torch.nn.Module:
    values = dict(config.model_config)
    requested_vocab = values.pop("vocab_size", len(TOKEN_TO_ID))
    if requested_vocab != len(TOKEN_TO_ID):
        raise ValueError("model_config vocab_size disagrees with T16 vocabulary")
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


def build_t16_model(
    config: shared_training.TrainConfig | Mapping[str, Any],
) -> torch.nn.Module:
    return _t16_model_factory(shared_training.TrainConfig.from_value(config))


def train_t16_run(
    run_config: shared_training.TrainConfig | Mapping[str, Any],
    *,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    config = shared_training.TrainConfig.from_value(run_config)
    if tuple(config.tasks or ()) != T16_TASKS:
        raise ValueError("T16 training requires the complete fixed 16-task order")
    if config.validation_tasks is None:
        config = replace(config, validation_tasks=T16_TASKS)
    elif tuple(config.validation_tasks) != T16_TASKS:
        raise ValueError("T16 validation must report all 16 tasks")

    original_token_to_id = shared_training.TOKEN_TO_ID
    shared_training.TOKEN_TO_ID = TOKEN_TO_ID
    try:
        return shared_training.train_run(
            config,
            model_factory=_t16_model_factory,
            stop_after_steps=stop_after_steps,
        )
    finally:
        shared_training.TOKEN_TO_ID = original_token_to_id


def smoke_config(
    *,
    architecture: str,
    manifest: str = "data/integer-8m-t16-v1/manifest.json",
    output_root: str = "runs/henry-integer-t16-v1-preflight",
    seed: int = 17,
    max_steps: int = 2,
    device: str | None = None,
) -> shared_training.TrainConfig:
    if architecture not in {"transformer", "mlp"}:
        raise ValueError("architecture must be transformer or mlp")
    run_id = f"{architecture}-tasks16-seed{seed}"
    return shared_training.TrainConfig(
        manifest=manifest,
        validation_manifest=manifest,
        output_dir=str(Path(output_root) / run_id),
        architecture=architecture,
        d_model=256,
        num_layers=4 if architecture == "transformer" else 1,
        num_heads=8,
        dropout=0.1,
        mlp_ratio=4.0,
        tie_embeddings=True,
        tasks=T16_TASKS,
        validation_tasks=T16_TASKS,
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
        checkpoint_every=max(1, max_steps),
        validate_every=max(1, max_steps),
        validation_batches_per_task=1,
        device=device or automatic_device(),
        amp=False,
        bf16=False,
        resume="auto",
        shard_indices=tuple(range(98)),
        validation_shard_indices=(98,),
    )


__all__ = [
    "NESTED_TASKS",
    "T16_TASKS",
    "automatic_device",
    "build_t16_model",
    "smoke_config",
    "train_t16_run",
]
