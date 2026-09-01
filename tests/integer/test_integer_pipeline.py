"""Dependency-light checks for integer rendering, generation, and verification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurips_integers.generate import TASK_NAMES, build_record, generate
from neurips_integers.passage import encode_number, passage_tokens
from neurips_integers.verify import verify


class IntegerPipelineTests(unittest.TestCase):
    def test_base100_is_shared(self) -> None:
        self.assertEqual(encode_number(7), ("07",))
        self.assertEqual(encode_number(137), ("<NUM_START>", "01", "37", "<NUM_END>"))

    def test_addition_example(self) -> None:
        tokens = passage_tokens("addition", 332, size=3, primary=247, operand=85)
        self.assertEqual(
            " ".join(tokens),
            "<BOS> <SIZE> 03 <NUM_START> 02 47 <NUM_END> + "
            "<ARG_START> 85 <ARG_END> <ADDITION> = "
            "<NUM_START> 03 32 <NUM_END> <EOS>",
        )

    def test_task_schedule_is_balanced(self) -> None:
        records = [build_record(index) for index in range(80)]
        self.assertEqual({task: 10 for task in TASK_NAMES}, {
            task: sum(record["task"] == task for record in records)
            for task in TASK_NAMES
        })

    def test_small_full_generation_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # The generator intentionally requires 100 shards, even in tests.
            generate(Path(directory), count=800, shard_size=8, max_digits=4)
            result = verify(Path(directory))
            self.assertEqual(result["records"], 800)
            self.assertEqual(set(result["task_counts"]), set(TASK_NAMES))


if __name__ == "__main__":
    unittest.main()
