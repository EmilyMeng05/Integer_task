"""Create deterministic train/validation/test manifest views over data shards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SPLIT = (98, 1, 1)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def create_split_manifests(
    manifest_path: str | Path,
    *,
    train_shards: int = 98,
    validation_shards: int = 1,
    test_shards: int = 1,
) -> dict[str, Path]:
    """Write non-overlapping shard-level split manifests next to the source."""

    source = Path(manifest_path)
    source_bytes = source.read_bytes()
    manifest = json.loads(source_bytes)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("source manifest must contain a nonempty shards list")
    sizes = (train_shards, validation_shards, test_shards)
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes):
        raise ValueError("every split must receive at least one shard")
    if sum(sizes) != len(shards):
        raise ValueError("split shard counts must sum to source shard_count")

    source_hash = hashlib.sha256(source_bytes).hexdigest()
    result: dict[str, Path] = {}
    offset = 0
    for name, size in zip(("train", "validation", "test"), sizes, strict=True):
        selected = shards[offset : offset + size]
        offset += size
        counts: Counter[str] = Counter()
        for shard in selected:
            counts.update(shard["task_counts"])
        view = {
            key: value
            for key, value in manifest.items()
            if key not in {"count", "shard_count", "task_counts", "total_bytes", "shards"}
        }
        view.update(
            {
                "split": name,
                "parent_manifest": source.name,
                "parent_manifest_sha256": source_hash,
                "count": sum(int(shard["record_count"]) for shard in selected),
                "shard_count": len(selected),
                "task_counts": {
                    task: counts[task] for task in manifest["tasks"]
                },
                "total_bytes": sum(int(shard["byte_size"]) for shard in selected),
                "shards": selected,
            }
        )
        destination = source.with_name(f"{name}_manifest.json")
        _atomic_json(destination, view)
        result[name] = destination
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--train-shards", type=int, default=98)
    parser.add_argument("--validation-shards", type=int, default=1)
    parser.add_argument("--test-shards", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = create_split_manifests(
        args.manifest,
        train_shards=args.train_shards,
        validation_shards=args.validation_shards,
        test_shards=args.test_shards,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

