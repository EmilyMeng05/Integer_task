from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import unittest

from neurips_integers.generate import TASK_NAMES
from neurips_integers.production import (
    build_matrix,
    build_train_config,
    validate_protocol,
)


CONFIG = Path("configs/henry_integer_production.toml")


class IntegerProductionTests(unittest.TestCase):
    def test_formal_matrix_has_24_unique_nested_runs(self) -> None:
        runs = build_matrix(CONFIG)
        self.assertEqual(len(runs), 24)
        self.assertEqual(len({run.run_id for run in runs}), 24)
        self.assertEqual({run.task_count for run in runs}, {1, 2, 4, 8})
        self.assertEqual(
            {run.architecture for run in runs}, {"transformer", "mlp"}
        )
        self.assertEqual({run.seed for run in runs}, {17, 42, 314159})
        for run in runs:
            self.assertEqual(run.tasks, TASK_NAMES[: run.task_count])

    def test_first_run_is_transformer_t1_seed17(self) -> None:
        run = build_matrix(CONFIG)[0]
        self.assertEqual(run.run_id, "transformer-tasks01-seed17")
        self.assertEqual(run.tasks, ("decimal_digit_sum",))

    def test_formal_train_config_uses_frozen_data_and_schedule(self) -> None:
        run = build_matrix(CONFIG)[0]
        config = build_train_config(run, config_path=CONFIG, device="cpu")
        self.assertEqual(config.manifest, "data/integer-4m-v1/manifest.json")
        self.assertEqual(config.validation_manifest, config.manifest)
        self.assertEqual(config.shard_indices, tuple(range(98)))
        self.assertEqual(config.validation_shard_indices, (98,))
        self.assertEqual(config.max_steps, 20_000)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.tasks, ("decimal_digit_sum",))
        self.assertEqual(config.validation_tasks, config.tasks)
        self.assertEqual(config.resume, "auto")
        self.assertEqual(
            config.output_dir,
            "runs/henry-integer-v1/transformer-tasks01-seed17",
        )

    def test_preflight_is_short_and_separate_from_formal_output(self) -> None:
        run = build_matrix(CONFIG)[0]
        formal = build_train_config(run, config_path=CONFIG, device="cpu")
        preflight = build_train_config(
            run, config_path=CONFIG, preflight_steps=10, device="cpu"
        )
        self.assertEqual(preflight.max_steps, 10)
        self.assertEqual(preflight.validate_every, 10)
        self.assertEqual(preflight.checkpoint_every, 10)
        self.assertEqual(preflight.validation_batches_per_task, 1)
        self.assertTrue(
            preflight.output_dir.endswith(
                "henry-integer-v1-preflight/steps00010/"
                "transformer-tasks01-seed17"
            )
        )
        self.assertNotEqual(preflight.output_dir, formal.output_dir)
        self.assertEqual(preflight.tasks, formal.tasks)
        self.assertEqual(preflight.manifest, formal.manifest)

    def test_protocol_rejects_task_order_changes(self) -> None:
        config = {
            "task_order": list(reversed(TASK_NAMES)),
            "task_subset_sizes": [1, 2, 4, 8],
            "architectures": ["transformer", "mlp"],
            "model_seeds": [17, 42, 314159],
            "total_records": 4_000_000,
            "records_per_task": 500_000,
            "dataset_manifest": "data/integer-4m-v1/manifest.json",
            "output_dir": "runs/henry-integer-v1",
            "preflight_output_dir": "runs/henry-integer-v1-preflight",
            "data": {
                "train_shards": "000-097",
                "validation_shards": "098",
                "test_shards": "099",
            },
            "model": {"d_model": 256},
            "training": {
                "max_steps": 20_000,
                "micro_batch_size": 16,
                "gradient_accumulation_steps": 4,
            },
        }
        with self.assertRaisesRegex(ValueError, "task_order"):
            validate_protocol(config)

    def test_train_config_is_json_compatible_after_asdict(self) -> None:
        run = build_matrix(CONFIG)[-1]
        payload = asdict(build_train_config(run, config_path=CONFIG, device="cpu"))
        self.assertEqual(payload["architecture"], "mlp")
        self.assertEqual(payload["tasks"], TASK_NAMES)
        self.assertEqual(payload["seed"], 314159)


if __name__ == "__main__":
    unittest.main()
