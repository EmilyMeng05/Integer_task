"""Small resumability check for the parallel T16 production generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurips_integers.generate_t16 import TASK_NAMES
from neurips_integers.generate_t16_production import generate_production
from neurips_integers.verify_t16 import verify


class IntegerT16ProductionTests(unittest.TestCase):
    def test_small_parallel_run_is_balanced_resumable_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = generate_production(
                output, count=8_000, shard_size=80, workers=1
            )
            second = generate_production(
                output, count=8_000, shard_size=80, workers=1
            )
            self.assertEqual(first["task_counts"], second["task_counts"])
            self.assertEqual(
                second["task_counts"], {task: 500 for task in TASK_NAMES}
            )
            self.assertEqual(second["completed_shards"], 100)
            self.assertEqual(verify(output)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
