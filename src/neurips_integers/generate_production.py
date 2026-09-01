"""Resumable parallel generation of the four-million-record integer corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generate import (
    DEFAULT_MAX_DIGITS,
    DEFAULT_MIN_DIGITS,
    DEFAULT_SEED,
    PRODUCTION_SCHEMA_VERSION,
    TASK_NAMES,
    TASK_ORDER_SEED,
    build_record,
)


DEFAULT_COUNT = 4_000_000
DEFAULT_SHARD_SIZE = 40_000
DEFAULT_OUTPUT_DIR = Path("data/integer-4m-v1")
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
GZIP_COMPRESSLEVEL = 6


@dataclass(frozen=True, slots=True)
class ShardJob:
    index: int
    first_id: int
    record_count: int
    output_dir: str
    min_digits: int
    max_digits: int
    seed: int

    @property
    def filename(self) -> str:
        return f"part-{self.index:05d}.jsonl.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_shard(job: ShardJob) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    final_path = output_dir / job.filename
    partial_path = output_dir / f"{job.filename}.partial"
    counts: Counter[str] = Counter()
    with partial_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=GZIP_COMPRESSLEVEL,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            for record_id in range(job.first_id, job.first_id + job.record_count):
                record = build_record(
                    record_id,
                    min_digits=job.min_digits,
                    max_digits=job.max_digits,
                    seed=job.seed,
                    schema_version=PRODUCTION_SCHEMA_VERSION,
                )
                counts[record["task"]] += 1
                line = json.dumps(
                    record, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                )
                compressed.write(line.encode("utf-8") + b"\n")
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(partial_path, final_path)
    return {
        "index": job.index,
        "filename": job.filename,
        "first_id": job.first_id,
        "last_id": job.first_id + job.record_count - 1,
        "record_count": job.record_count,
        "byte_size": final_path.stat().st_size,
        "sha256": _sha256(final_path),
        "task_counts": {task: counts[task] for task in TASK_NAMES},
    }


def _compatible_entry(job: ShardJob, entry: object) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    path = Path(job.output_dir) / job.filename
    expected = {
        "index": job.index,
        "filename": job.filename,
        "first_id": job.first_id,
        "last_id": job.first_id + job.record_count - 1,
        "record_count": job.record_count,
    }
    if any(entry.get(key) != value for key, value in expected.items()):
        return None
    if not path.is_file() or path.stat().st_size != entry.get("byte_size"):
        return None
    if _sha256(path) != entry.get("sha256"):
        return None
    counts = entry.get("task_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(TASK_NAMES):
        return None
    if sum(counts.values()) != job.record_count:
        return None
    return dict(entry)


def _base_manifest(
    *,
    count: int,
    shard_size: int,
    min_digits: int,
    max_digits: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "count": count,
        "base": 100,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "seed": seed,
        "task_order_seed": TASK_ORDER_SEED,
        "tasks": list(TASK_NAMES),
        "records_per_task": count // len(TASK_NAMES),
        "records_per_task_per_digit_length": count
        // len(TASK_NAMES)
        // (max_digits - min_digits + 1),
        "shard_size": shard_size,
        "shard_count": count // shard_size,
        "splits": {"train": [0, 97], "validation": [98, 98], "test": [99, 99]},
    }


def _assert_existing_manifest_compatible(
    existing: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    keys = (
        "schema_version",
        "count",
        "base",
        "min_digits",
        "max_digits",
        "seed",
        "task_order_seed",
        "tasks",
        "shard_size",
        "shard_count",
        "splits",
    )
    mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(
            "existing production manifest uses a different configuration: "
            + ", ".join(mismatches)
        )


def generate_production(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    count: int = DEFAULT_COUNT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    min_digits: int = DEFAULT_MIN_DIGITS,
    max_digits: int = DEFAULT_MAX_DIGITS,
    seed: int = DEFAULT_SEED,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Generate or safely resume the frozen production corpus."""

    if count <= 0 or count % len(TASK_NAMES):
        raise ValueError("count must be positive and divisible by eight")
    if shard_size <= 0 or count % shard_size:
        raise ValueError("shard_size must be positive and divide count")
    if count // shard_size != 100:
        raise ValueError("production data must contain exactly 100 shards")
    if not 1 <= min_digits <= max_digits:
        raise ValueError("invalid decimal digit range")
    cells = len(TASK_NAMES) * (max_digits - min_digits + 1)
    if count % cells:
        raise ValueError("count must balance every task-by-digit-length cell")
    if workers < 1:
        raise ValueError("workers must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = _base_manifest(
        count=count,
        shard_size=shard_size,
        min_digits=min_digits,
        max_digits=max_digits,
        seed=seed,
    )
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("existing manifest must be a JSON object")
        _assert_existing_manifest_compatible(existing, manifest)
    existing_entries = {
        entry.get("index"): entry
        for entry in existing.get("shards", [])
        if isinstance(entry, Mapping)
    }

    jobs = tuple(
        ShardJob(
            index=index,
            first_id=index * shard_size,
            record_count=shard_size,
            output_dir=str(output_dir),
            min_digits=min_digits,
            max_digits=max_digits,
            seed=seed,
        )
        for index in range(100)
    )
    entries: dict[int, dict[str, Any]] = {}
    pending: list[ShardJob] = []
    for job in jobs:
        reusable = _compatible_entry(job, existing_entries.get(job.index))
        if reusable is None:
            pending.append(job)
        else:
            entries[job.index] = reusable
    if entries:
        print(f"Reusing {len(entries)}/100 verified shards from the existing manifest")

    def publish_progress() -> None:
        ordered = [entries[index] for index in sorted(entries)]
        counts: Counter[str] = Counter()
        for entry in ordered:
            counts.update(entry["task_counts"])
        payload = {
            **manifest,
            "completed_shards": len(ordered),
            "task_counts": {task: counts[task] for task in TASK_NAMES},
            "shards": ordered,
        }
        _atomic_json(payload, manifest_path)

    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            future_jobs = {pool.submit(_write_shard, job): job for job in pending}
            for completed, future in enumerate(as_completed(future_jobs), start=1):
                entry = future.result()
                entries[int(entry["index"])] = entry
                publish_progress()
                if completed == 1 or completed % 5 == 0 or completed == len(pending):
                    print(
                        f"Generated {completed}/{len(pending)} pending shards "
                        f"({len(entries)}/100 total complete)",
                        flush=True,
                    )
    else:
        publish_progress()

    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    if final.get("completed_shards") != 100:
        raise RuntimeError("production generation ended with incomplete shards")
    expected_task_count = count // len(TASK_NAMES)
    if final.get("task_counts") != {
        task: expected_task_count for task in TASK_NAMES
    }:
        raise RuntimeError("production task allocation is not exactly balanced")
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--min-digits", type=int, default=DEFAULT_MIN_DIGITS)
    parser.add_argument("--max-digits", type=int, default=DEFAULT_MAX_DIGITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    result = generate_production(
        args.output_dir,
        count=args.count,
        shard_size=args.shard_size,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        seed=args.seed,
        workers=args.workers,
    )
    print(f"Production generation complete: {result['count']:,} records")
    print(f"Records per task: {result['records_per_task']:,}")
    print(f"Saved to: {args.output_dir}")
    print("Run the full verifier before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_production"]
