from __future__ import annotations

from neurips_integers import compare_task_scaling as comparison


def _group(exact: int, examples: int, well_formed: int) -> dict[str, object]:
    return {
        "autoregressive_exact": exact,
        "autoregressive_exact_accuracy": exact / examples,
        "examples": examples,
        "well_formed": well_formed,
        "well_formed_rate": well_formed / examples,
    }


def _result(split: str, schema: str) -> dict[str, object]:
    task = "decimal_digit_sum"
    by_task = {
        task: {
            **_group(10 if split == "iid" else 4, 10, 10 if split == "iid" else 8),
            "most_common_target_baseline": {"accuracy": 0.2, "value": 1},
        }
    }
    metrics: dict[str, object] = {"by_task": by_task}
    if split == "ood":
        groups = {
            "6": _group(6, 10, 9),
            "7": _group(1, 10, 8),
            "8": _group(2, 10, 7),
            "9": _group(3, 10, 6),
            "10": _group(4, 10, 5),
        }
        key = (
            "by_task_and_evaluation_bucket"
            if schema == "t16" else "by_task_and_digit_length"
        )
        metrics[key] = {task: groups}
    return {
        "run": {
            "architecture": "transformer",
            "seed": 17,
            "task_count": 1,
            "run_id": "transformer-tasks01-seed17",
        },
        "split": split,
        "metrics": metrics,
        "evaluation_data_sha256": "same-data",
    }


def test_extracts_identical_regimes_from_both_schemas() -> None:
    iid = _result("iid", "legacy")
    legacy = comparison.extract_observation(
        iid, _result("ood", "legacy"), task="decimal_digit_sum"
    )
    t16 = comparison.extract_observation(
        iid, _result("ood", "t16"), task="decimal_digit_sum"
    )
    assert legacy.magnitude_ood_exact_accuracy == 0.6
    assert legacy.token_length_ood_exact_accuracy == 0.25
    assert legacy.magnitude_ood_well_formed_rate == 0.9
    assert legacy.token_length_ood_well_formed_rate == 0.65
    assert legacy == t16


def test_stage_registry_is_nested() -> None:
    previous: set[str] = set()
    for stage in comparison.STAGES:
        current = set(comparison.TASKS_BY_STAGE[stage])
        assert previous <= current
        previous = current
    assert len(comparison.TASKS_BY_STAGE[16]) == 16
    assert len(comparison.SHARED_TASKS) == 8


def test_markdown_warns_when_evaluation_data_differ() -> None:
    rows = []
    for architecture in comparison.ARCHITECTURES:
        for task in comparison.SHARED_TASKS:
            start = comparison.INTRODUCTION_STAGE[task]
            for stage in comparison.STAGES:
                if stage < start:
                    continue
                for seed in comparison.SEEDS:
                    rows.append(comparison.Observation(
                        architecture=architecture,
                        seed=seed,
                        stage=stage,
                        task=task,
                        iid_exact_accuracy=0.5,
                        iid_well_formed_rate=1.0,
                        magnitude_ood_exact_accuracy=0.4,
                        magnitude_ood_well_formed_rate=1.0,
                        token_length_ood_exact_accuracy=0.1,
                        token_length_ood_well_formed_rate=0.8,
                        overall_ood_exact_accuracy=0.16,
                        overall_ood_well_formed_rate=0.84,
                        ood_most_common_target_baseline=0.05,
                        iid_evaluation_data="iid-a",
                        ood_evaluation_data="ood-a" if stage < 16 else "ood-b",
                    ))
    summary = comparison.summarize(rows)
    assert not summary["evaluation_protocol_audit"][
        "exactly_paired_examples_across_all_stages"
    ]
    assert "distribution-level comparison only" in comparison.render_markdown(summary)
