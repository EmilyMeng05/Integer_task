from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from neurips_integers.evaluate import (
    MetricAccumulator,
    greedy_generate_batch,
    load_balanced_records,
    parse_generated_answer,
    split_prompt_and_answer,
)
from neurips_integers.generate import OOD_SCHEMA_VERSION, TASK_NAMES
from neurips_integers.generate_ood import generate_ood
from neurips_integers.passage import TOKEN_TO_ID
from neurips_integers.verify import verify


class ScheduledNextTokenModel(torch.nn.Module):
    def __init__(self, prompt_length: int, generated: tuple[str, ...]) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.generated = tuple(TOKEN_TO_ID[token] for token in generated)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, width = input_ids.shape
        logits = torch.full(
            (batch, width, len(TOKEN_TO_ID)),
            -100.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        lengths = attention_mask.sum(dim=1)
        for index in range(batch):
            generated_count = int(lengths[index]) - self.prompt_length
            next_id = self.generated[generated_count]
            logits[index, int(lengths[index]) - 1, next_id] = 100.0
        return logits


class IntegerEvaluationTests(unittest.TestCase):
    def test_prompt_split_hides_all_answer_tokens(self) -> None:
        tokens = ("<BOS>", "<SIZE>", "03", "<SUCCESSOR>", "=", "04", "<EOS>")
        prompt, answer = split_prompt_and_answer(tokens)
        self.assertEqual(prompt[-1], "=")
        self.assertEqual(answer, ("04", "<EOS>"))
        self.assertNotIn("04", prompt)

    def test_generated_number_requires_canonical_eos_terminated_form(self) -> None:
        self.assertEqual(parse_generated_answer(("04", "<EOS>")), 4)
        self.assertEqual(
            parse_generated_answer(
                ("<NUM_START>", "01", "37", "<NUM_END>", "<EOS>")
            ),
            137,
        )
        self.assertIsNone(parse_generated_answer(("04",)))
        self.assertIsNone(parse_generated_answer(("04", "05", "<EOS>")))

    def test_greedy_decoder_receives_prompt_only(self) -> None:
        prompt = ("<BOS>", "<SIZE>", "01", "<SUCCESSOR>", "=")
        model = ScheduledNextTokenModel(len(prompt), ("04", "<EOS>"))
        generated = greedy_generate_batch(
            model,
            (prompt,),
            device=torch.device("cpu"),
            max_new_tokens=4,
            max_seq_len=32,
        )
        self.assertEqual(generated, [("04", "<EOS>")])

    def test_metric_accumulator_tracks_generation_and_teacher_forcing(self) -> None:
        metrics = MetricAccumulator()
        metrics.update(
            expected_tokens=("04", "<EOS>"),
            generated_tokens=("04", "<EOS>"),
            expected_value=4,
            teacher_loss_sum=0.2,
            teacher_tokens=2,
            teacher_token_correct=2,
            teacher_sequence_correct=True,
        )
        summary = metrics.summary()
        self.assertEqual(summary["autoregressive_exact_accuracy"], 1.0)
        self.assertEqual(summary["well_formed_rate"], 1.0)
        self.assertAlmostEqual(float(summary["teacher_forced_loss"]), 0.1)

    def test_small_ood_corpus_is_balanced_and_fully_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "ood"
            manifest = generate_ood(output, count=800, shard_size=8, seed=123)
            self.assertEqual(manifest["schema_version"], OOD_SCHEMA_VERSION)
            self.assertEqual(manifest["splits"], {"ood": [0, 99]})
            self.assertEqual(manifest["task_counts"], {task: 100 for task in TASK_NAMES})
            result = verify(output)
            self.assertEqual(result["records"], 800)
            records = load_balanced_records(
                output / "manifest.json",
                tasks=("decimal_digit_sum",),
                lengths=(6, 7, 8, 9, 10),
                examples_per_task_per_length=20,
            )
            self.assertEqual(len(records), 100)
            self.assertEqual(
                {length: sum(record["n_digits"] == length for record in records) for length in range(6, 11)},
                {length: 20 for length in range(6, 11)},
            )


if __name__ == "__main__":
    unittest.main()
