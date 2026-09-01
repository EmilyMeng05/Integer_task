"""Launch the eight Phase 1 integer smoke-test runs sequentially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ARCHITECTURES = ("transformer", "mlp")
TASK_COUNTS = (1, 2, 4, 8)


def commands(
    *,
    manifest: str,
    output_root: str,
    seed: int,
    max_steps: int,
    device: str | None,
) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []
    for task_count in TASK_COUNTS:
        for architecture in ARCHITECTURES:
            command = [
                sys.executable,
                "-m",
                "neurips_integers.training",
                "--architecture",
                architecture,
                "--task-count",
                str(task_count),
                "--manifest",
                manifest,
                "--output-root",
                output_root,
                "--seed",
                str(seed),
                "--max-steps",
                str(max_steps),
            ]
            if device:
                command.extend(("--device", device))
            result.append(tuple(command))
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/integer-20k-pilot/manifest.json")
    parser.add_argument("--output-root", default="runs/integer-pilot-smoke")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"pilot manifest not found: {manifest}")
    planned = commands(
        manifest=str(manifest),
        output_root=args.output_root,
        seed=args.seed,
        max_steps=args.max_steps,
        device=args.device,
    )
    print("Integer pilot smoke matrix:")
    for index, command in enumerate(planned, start=1):
        print(f"  {index}/8: {' '.join(command)}")
    if args.dry_run:
        return 0

    completed: list[str] = []
    for index, command in enumerate(planned, start=1):
        print(f"\nRunning {index}/8: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
        completed.append(f"{command[4]}-tasks{int(command[6]):02d}")
    print(json.dumps({"status": "passed", "completed_runs": completed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
