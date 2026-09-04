"""Checks for the isolated T16 rendering and pilot-data pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurips_integers.generate import build_record as build_legacy_record
from neurips_integers.generate_t16 import TASK_NAMES, build_record, generate
from neurips_integers.passage import TOKEN_TO_ID as LEGACY_TOKEN_TO_ID
from neurips_integers.passage_t16 import TOKEN_TO_ID, passage_tokens
from neurips_integers.verify_t16 import verify


class IntegerT16PipelineTests(unittest.TestCase):
    def test_legacy_vocabulary_ids_are_unchanged(self) -> None:
        for token, token_id in LEGACY_TOKEN_TO_ID.items():
            self.assertEqual(TOKEN_TO_ID[token], token_id)

    def test_original_eight_inputs_and_answers_are_exactly_reused(self) -> None:
        for occurrence in range(25):
            for task_index in range(8):
                legacy = build_legacy_record(occurrence * 8 + task_index)
                t16 = build_record(occurrence * 16 + task_index)
                self.assertEqual(t16["task"], legacy["task"])
                self.assertEqual(t16["inputs"], legacy["inputs"])
                self.assertEqual(t16["answer"], legacy["answer"])
                self.assertEqual(t16["tokens"], legacy["tokens"])

    def test_subtraction_example(self) -> None:
        tokens = passage_tokens(
            "subtraction", 162, size=3, primary=247, operand=85
        )
        self.assertEqual(
            " ".join(tokens),
            "<BOS> <SIZE> 03 <NUM_START> 02 47 <NUM_END> - "
            "<ARG_START> 85 <ARG_END> <SUBTRACTION> = "
            "<NUM_START> 01 62 <NUM_END> <EOS>",
        )

    def test_factorial_example(self) -> None:
        tokens = passage_tokens("factorial", 120, size=1, primary=5)
        self.assertEqual(
            " ".join(tokens),
            "<BOS> <SIZE> 01 05 ! <FACTORIAL> = "
            "<NUM_START> 01 20 <NUM_END> <EOS>",
        )

    def test_all_new_tasks_build_valid_records(self) -> None:
        found = {}
        for record_id in range(16 * 10):
            record = build_record(record_id)
            if record["task"] not in found:
                found[record["task"]] = record
        self.assertEqual(set(found), set(TASK_NAMES))
        for record in found.values():
            self.assertEqual(record["tokens"][0], "<BOS>")
            self.assertEqual(record["tokens"][-1], "<EOS>")
            self.assertEqual(record["canonical_text"], " ".join(record["tokens"]))

    def test_task_schedule_is_balanced(self) -> None:
        records = [build_record(index) for index in range(16 * 25)]
        self.assertEqual(
            {task: 25 for task in TASK_NAMES},
            {
                task: sum(record["task"] == task for record in records)
                for task in TASK_NAMES
            },
        )

    def test_small_generation_and_full_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # 8,000 = 16 tasks x 5 buckets x 100 records per cell.
            generate(Path(directory), count=8_000, shard_size=80)
            result = verify(Path(directory))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["records"], 8_000)
            self.assertEqual(
                result["task_counts"], {task: 500 for task in TASK_NAMES}
            )


if __name__ == "__main__":
    unittest.main()
