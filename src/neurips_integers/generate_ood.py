"""Generate the frozen 6--10-digit integer length-OOD evaluation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .generate import OOD_SCHEMA_VERSION, TASK_NAMES, generate


DEFAULT_OUTPUT_DIR = Path("data/integer-ood-40k-v1")
DEFAULT_COUNT = 40_000
DEFAULT_SHARD_SIZE = 400
DEFAULT_MIN_DIGITS = 6
DEFAULT_MAX_DIGITS = 10
DEFAULT_SEED = 20_260_901


def generate_ood(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    count: int = DEFAULT_COUNT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create 1,000 examples per task at every OOD decimal length."""

    group_count = len(TASK_NAMES) * (
        DEFAULT_MAX_DIGITS - DEFAULT_MIN_DIGITS + 1
    )
    if count <= 0 or count % group_count:
        raise ValueError("OOD count must be positive and divisible by 8 tasks x 5 lengths")
    manifest = generate(
        output_dir,
        count=count,
        shard_size=shard_size,
        min_digits=DEFAULT_MIN_DIGITS,
        max_digits=DEFAULT_MAX_DIGITS,
        seed=seed,
        schema_version=OOD_SCHEMA_VERSION,
    )
    manifest["purpose"] = "length_ood_evaluation_only"
    manifest["splits"] = {"ood": [0, 99]}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    manifest = generate_ood(
        args.output_dir,
        count=args.count,
        shard_size=args.shard_size,
        seed=args.seed,
    )
    per_group = manifest["count"] // (len(TASK_NAMES) * 5)
    print(f"Generated {manifest['count']:,} length-OOD records")
    print(f"Tasks: {len(TASK_NAMES)}; decimal lengths: 6-10")
    print(f"Records per task per length: {per_group:,}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


__all__ = ["generate_ood"]
