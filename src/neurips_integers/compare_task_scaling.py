"""Compare shared-task accuracy across the nested T1/T2/T4/T8/T16 runs.

This module only reads completed evaluation JSON files. It never trains or
evaluates a model. It supports both the legacy T1--T8 evaluation schema and
the newer T16 schema, and derives the OOD regimes from decimal-length groups
so their definitions are consistent:

* 6 digits: magnitude OOD with familiar base-100 token width.
* 7--10 digits: genuine token-length OOD.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence


ARCHITECTURES = ("transformer", "mlp")
SEEDS = (17, 42, 314159)
STAGES = (1, 2, 4, 8, 16)

TASKS_BY_STAGE: dict[int, tuple[str, ...]] = {
    1: ("decimal_digit_sum",),
    2: ("decimal_digit_sum", "greatest_common_divisor"),
    4: (
        "decimal_digit_sum",
        "greatest_common_divisor",
        "multiplication",
        "greater_than",
    ),
    8: (
        "decimal_digit_sum",
        "greatest_common_divisor",
        "multiplication",
        "greater_than",
        "integer_list_sum",
        "modulo",
        "addition",
        "successor",
    ),
    16: (
        "decimal_digit_sum",
        "greatest_common_divisor",
        "multiplication",
        "greater_than",
        "integer_list_sum",
        "modulo",
        "addition",
        "successor",
        "subtraction",
        "integer_division",
        "number_of_decimal_digits",
        "reverse_decimal_digits",
        "decimal_digit_occurrence_count",
        "even_odd",
        "divisibility",
        "factorial",
    ),
}

# The scaling comparison follows the eight tasks that existed before T16.
SHARED_TASKS = TASKS_BY_STAGE[8]
INTRODUCTION_STAGE = {
    task: min(stage for stage, tasks in TASKS_BY_STAGE.items() if task in tasks)
    for task in SHARED_TASKS
}

FIELDNAMES = (
    "architecture",
    "seed",
    "stage",
    "task",
    "iid_exact_accuracy",
    "iid_well_formed_rate",
    "magnitude_ood_exact_accuracy",
    "magnitude_ood_well_formed_rate",
    "token_length_ood_exact_accuracy",
    "token_length_ood_well_formed_rate",
    "overall_ood_exact_accuracy",
    "overall_ood_well_formed_rate",
    "ood_most_common_target_baseline",
    "iid_evaluation_data",
    "ood_evaluation_data",
)


@dataclass(frozen=True)
class Observation:
    architecture: str
    seed: int
    stage: int
    task: str
    iid_exact_accuracy: float
    iid_well_formed_rate: float
    magnitude_ood_exact_accuracy: float
    magnitude_ood_well_formed_rate: float
    token_length_ood_exact_accuracy: float
    token_length_ood_well_formed_rate: float
    overall_ood_exact_accuracy: float
    overall_ood_well_formed_rate: float
    ood_most_common_target_baseline: float
    iid_evaluation_data: str
    ood_evaluation_data: str


def _result_path(
    *, architecture: str, stage: int, seed: int, split: str,
    legacy_root: Path, t16_root: Path,
) -> Path:
    root = t16_root if stage == 16 else legacy_root
    return root / f"{architecture}-tasks{stage:02d}-seed{seed}-{split}.json"


def _read_result(
    path: Path, *, architecture: str, stage: int, seed: int, split: str,
    minimum_examples_per_task_per_length: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required result: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    run = value.get("run", {})
    expected_run = f"{architecture}-tasks{stage:02d}-seed{seed}"
    if run.get("run_id") != expected_run:
        raise ValueError(f"{path}: expected run_id {expected_run!r}")
    if run.get("architecture") != architecture or int(run.get("seed")) != seed:
        raise ValueError(f"{path}: run metadata does not match its filename")
    if value.get("split") != split:
        raise ValueError(f"{path}: expected split {split!r}")
    n = int(value.get("examples_per_task_per_length", 0))
    if n < minimum_examples_per_task_per_length:
        raise ValueError(
            f"{path}: only {n} examples per task per length; "
            f"at least {minimum_examples_per_task_per_length} required"
        )
    actual_tasks = set(value.get("metrics", {}).get("by_task", {}))
    expected_tasks = set(TASKS_BY_STAGE[stage])
    if actual_tasks != expected_tasks:
        raise ValueError(
            f"{path}: task mismatch; missing={sorted(expected_tasks-actual_tasks)}, "
            f"extra={sorted(actual_tasks-expected_tasks)}"
        )
    return value


def _evaluation_identity(result: Mapping[str, Any]) -> str:
    for key in (
        "evaluation_data_sha256",
        "evaluation_manifest_sha256",
        "validation_manifest_sha256",
    ):
        value = result.get(key)
        if value:
            return f"{key}:{value}"
    for key in ("evaluation_data", "evaluation_manifest"):
        value = result.get(key)
        if value:
            return f"{key}:{value}"
    return "not-recorded"


def _length_groups(result: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    metrics = result["metrics"]
    if "by_task_and_evaluation_bucket" in metrics:
        return metrics["by_task_and_evaluation_bucket"][task]
    if "by_task_and_digit_length" in metrics:
        return metrics["by_task_and_digit_length"][task]
    raise ValueError("result has no supported per-task length groups")


def _aggregate_groups(
    groups: Mapping[str, Mapping[str, Any]], lengths: Iterable[int],
) -> tuple[float, float]:
    selected = []
    for length in lengths:
        key = str(length)
        if key not in groups:
            raise ValueError(f"result is missing decimal-length group {key}")
        selected.append(groups[key])
    examples = sum(int(row["examples"]) for row in selected)
    if examples == 0:
        raise ValueError("cannot aggregate an empty metric group")
    exact = sum(int(row["autoregressive_exact"]) for row in selected) / examples
    well_formed = sum(int(row["well_formed"]) for row in selected) / examples
    return exact, well_formed


def extract_observation(
    iid: Mapping[str, Any], ood: Mapping[str, Any], *, task: str,
) -> Observation:
    iid_task = iid["metrics"]["by_task"][task]
    ood_task = ood["metrics"]["by_task"][task]
    groups = _length_groups(ood, task)
    magnitude_exact, magnitude_well_formed = _aggregate_groups(groups, (6,))
    length_exact, length_well_formed = _aggregate_groups(groups, (7, 8, 9, 10))
    baseline = ood_task.get("most_common_target_baseline")
    if not isinstance(baseline, Mapping) or "accuracy" not in baseline:
        raise ValueError(f"OOD result lacks most-common-answer baseline for {task}")
    run = iid["run"]
    if ood["run"]["run_id"] != run["run_id"]:
        raise ValueError("IID and OOD files belong to different runs")
    return Observation(
        architecture=str(run["architecture"]),
        seed=int(run["seed"]),
        stage=int(run["task_count"]),
        task=task,
        iid_exact_accuracy=float(iid_task["autoregressive_exact_accuracy"]),
        iid_well_formed_rate=float(iid_task["well_formed_rate"]),
        magnitude_ood_exact_accuracy=magnitude_exact,
        magnitude_ood_well_formed_rate=magnitude_well_formed,
        token_length_ood_exact_accuracy=length_exact,
        token_length_ood_well_formed_rate=length_well_formed,
        overall_ood_exact_accuracy=float(ood_task["autoregressive_exact_accuracy"]),
        overall_ood_well_formed_rate=float(ood_task["well_formed_rate"]),
        ood_most_common_target_baseline=float(baseline["accuracy"]),
        iid_evaluation_data=_evaluation_identity(iid),
        ood_evaluation_data=_evaluation_identity(ood),
    )


def collect_observations(
    *, legacy_root: Path, t16_root: Path,
    minimum_examples_per_task_per_length: int = 1_000,
) -> list[Observation]:
    observations: list[Observation] = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            for stage in STAGES:
                iid = _read_result(
                    _result_path(
                        architecture=architecture, stage=stage, seed=seed,
                        split="iid", legacy_root=legacy_root, t16_root=t16_root,
                    ),
                    architecture=architecture, stage=stage, seed=seed, split="iid",
                    minimum_examples_per_task_per_length=minimum_examples_per_task_per_length,
                )
                ood = _read_result(
                    _result_path(
                        architecture=architecture, stage=stage, seed=seed,
                        split="ood", legacy_root=legacy_root, t16_root=t16_root,
                    ),
                    architecture=architecture, stage=stage, seed=seed, split="ood",
                    minimum_examples_per_task_per_length=minimum_examples_per_task_per_length,
                )
                for task in SHARED_TASKS:
                    if task in TASKS_BY_STAGE[stage]:
                        observations.append(extract_observation(iid, ood, task=task))
    return observations


def _mean_sd(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize no values")
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "seeds": len(values),
    }


def summarize(observations: Sequence[Observation]) -> dict[str, Any]:
    metrics = (
        "iid_exact_accuracy",
        "magnitude_ood_exact_accuracy",
        "token_length_ood_exact_accuracy",
        "overall_ood_exact_accuracy",
        "ood_most_common_target_baseline",
        "iid_well_formed_rate",
        "magnitude_ood_well_formed_rate",
        "token_length_ood_well_formed_rate",
        "overall_ood_well_formed_rate",
    )
    aggregated: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        aggregated[architecture] = {}
        for task in SHARED_TASKS:
            aggregated[architecture][task] = {}
            for stage in STAGES:
                rows = [
                    row for row in observations
                    if row.architecture == architecture
                    and row.task == task and row.stage == stage
                ]
                if not rows:
                    continue
                if {row.seed for row in rows} != set(SEEDS):
                    raise ValueError(
                        f"expected seeds {SEEDS} for {architecture}/{task}/T{stage}"
                    )
                aggregated[architecture][task][str(stage)] = {
                    metric: _mean_sd([float(getattr(row, metric)) for row in rows])
                    for metric in metrics
                }

    deltas: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        deltas[architecture] = {}
        for task in SHARED_TASKS:
            start = str(INTRODUCTION_STAGE[task])
            end = "16"
            rows = aggregated[architecture][task]
            deltas[architecture][task] = {
                metric: rows[end][metric]["mean"] - rows[start][metric]["mean"]
                for metric in (
                    "iid_exact_accuracy",
                    "magnitude_ood_exact_accuracy",
                    "token_length_ood_exact_accuracy",
                )
            }

    identities = {
        row.ood_evaluation_data for row in observations
    }
    paired_examples = len(identities) == 1
    return {
        "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS),
        "stages": list(STAGES),
        "shared_tasks": list(SHARED_TASKS),
        "evaluation_protocol_audit": {
            "ood_data_identities": sorted(identities),
            "exactly_paired_examples_across_all_stages": paired_examples,
            "interpretation": (
                "paired comparison on the same recorded evaluation data"
                if paired_examples else
                "distribution-level comparison only; evaluation data identities differ"
            ),
        },
        "aggregates": aggregated,
        "introduction_to_t16_deltas": deltas,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _cell(summary: Mapping[str, Any], metric: str) -> str:
    value = summary[metric]
    return f"{_pct(value['mean'])} ± {_pct(value['sample_std'])}"


def _trajectory_table(
    summary: Mapping[str, Any], architecture: str, metric: str,
) -> list[str]:
    lines = [
        f"### {architecture.title()}",
        "",
        "| Task | T1 | T2 | T4 | T8 | T16 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in SHARED_TASKS:
        stage_rows = summary["aggregates"][architecture][task]
        cells = [
            _cell(stage_rows[str(stage)], metric) if str(stage) in stage_rows else "—"
            for stage in STAGES
        ]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return lines


def render_markdown(summary: Mapping[str, Any]) -> str:
    audit = summary["evaluation_protocol_audit"]
    lines = [
        "# Integer Task-Scaling Results: T1 to T16",
        "",
        "## Purpose",
        "",
        "This report follows each shared task after it enters the nested T1, T2, T4, T8, and T16 curriculum. Values are autoregressive exact-sequence accuracy, reported as mean ± sample standard deviation across seeds 17, 42, and 314159.",
        "",
        "## Evaluation protocol audit",
        "",
        f"- Exact paired examples across all stages: **{'yes' if audit['exactly_paired_examples_across_all_stages'] else 'no'}**",
        f"- Interpretation: **{audit['interpretation']}**",
        "",
    ]
    if not audit["exactly_paired_examples_across_all_stages"]:
        lines.extend([
            "> **Important:** The legacy T1–T8 results and T16 results record different evaluation-data identities. The tables are useful as a distribution-level comparison because they use the same decimal-length regimes and sample counts, but they are not a strictly paired comparison on identical examples. Do not attribute differences solely to task count without stating this limitation.",
            "",
        ])
    sections = (
        ("IID exact accuracy", "iid_exact_accuracy"),
        ("Six-digit magnitude-OOD exact accuracy", "magnitude_ood_exact_accuracy"),
        ("Seven-to-ten-digit token-length-OOD exact accuracy", "token_length_ood_exact_accuracy"),
    )
    for title, metric in sections:
        lines.extend([f"## {title}", ""])
        for architecture in ARCHITECTURES:
            lines.extend(_trajectory_table(summary, architecture, metric))
            lines.append("")

    lines.extend([
        "## Change from task introduction to T16",
        "",
        "Positive values indicate higher accuracy at T16; negative values indicate lower accuracy. These are percentage-point changes in the three-seed means.",
        "",
        "| Architecture | Task | Starting stage | IID Δ | Six-digit Δ | 7–10-digit Δ |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for architecture in ARCHITECTURES:
        for task in SHARED_TASKS:
            delta = summary["introduction_to_t16_deltas"][architecture][task]
            lines.append(
                f"| {architecture} | {task} | T{INTRODUCTION_STAGE[task]} | "
                f"{100*delta['iid_exact_accuracy']:+.2f} | "
                f"{100*delta['magnitude_ood_exact_accuracy']:+.2f} | "
                f"{100*delta['token_length_ood_exact_accuracy']:+.2f} |"
            )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Compare a task only across stages in which that task was trained.",
        "- Do not compare changing macro averages across different task compositions.",
        "- Compare GCD results with the most-common-target baseline because `1` is frequent.",
        "- Treat six-digit magnitude OOD separately from genuine 7–10-digit token-length OOD.",
        "- Under the fixed 20,000-step budget, later stages provide fewer examples per task; task diversity and per-task exposure are therefore not separately identified.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(
    observations: Sequence[Observation], summary: Mapping[str, Any], output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "task_scaling_records.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in observations:
            writer.writerow(asdict(row))
    (output_dir / "task_scaling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README_TASK_SCALING_RESULTS.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=Path("results/integer-v1"))
    parser.add_argument("--t16-root", type=Path, default=Path("results/integer-t16-v1"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/integer-task-scaling-v1")
    )
    parser.add_argument(
        "--minimum-examples-per-task-per-length", type=int, default=1_000
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    observations = collect_observations(
        legacy_root=args.legacy_root,
        t16_root=args.t16_root,
        minimum_examples_per_task_per_length=args.minimum_examples_per_task_per_length,
    )
    summary = summarize(observations)
    write_outputs(observations, summary, args.output_dir)
    print(f"Validated {len(observations)} task/run observations")
    print(f"Saved: {args.output_dir / 'task_scaling_records.csv'}")
    print(f"Saved: {args.output_dir / 'task_scaling_summary.json'}")
    print(f"Saved: {args.output_dir / 'README_TASK_SCALING_RESULTS.md'}")
    audit = summary["evaluation_protocol_audit"]
    print(f"Evaluation comparison: {audit['interpretation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Observation",
    "collect_observations",
    "extract_observation",
    "render_markdown",
    "summarize",
]
