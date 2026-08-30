from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurips_permutations.splits import create_split_manifests


def _source(path: Path) -> Path:
    shards = []
    for index in range(4):
        shards.append(
            {
                "index": index,
                "filename": f"part-{index:05d}.jsonl.gz",
                "record_count": 20,
                "byte_size": 100 + index,
                "task_counts": {"a": 10, "b": 10},
            }
        )
    value = {
        "schema_version": "test/v1",
        "tasks": ["a", "b"],
        "count": 80,
        "shard_count": 4,
        "task_counts": {"a": 40, "b": 40},
        "total_bytes": 406,
        "shards": shards,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_split_views_are_nonoverlapping_and_counted(tmp_path: Path) -> None:
    paths = create_split_manifests(
        _source(tmp_path / "manifest.json"),
        train_shards=2,
        validation_shards=1,
        test_shards=1,
    )
    views = {name: json.loads(path.read_text()) for name, path in paths.items()}
    assert views["train"]["count"] == 40
    assert views["validation"]["count"] == 20
    assert views["test"]["count"] == 20
    shard_sets = [
        {entry["filename"] for entry in view["shards"]}
        for view in views.values()
    ]
    assert not (shard_sets[0] & shard_sets[1])
    assert not (shard_sets[0] & shard_sets[2])
    assert not (shard_sets[1] & shard_sets[2])
    assert all(view["task_counts"] in ({"a": 20, "b": 20}, {"a": 10, "b": 10}) for view in views.values())


def test_split_sizes_must_cover_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sum"):
        create_split_manifests(
            _source(tmp_path / "manifest.json"),
            train_shards=1,
            validation_shards=1,
            test_shards=1,
        )

