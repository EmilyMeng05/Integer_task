"""Architecture, causality, masking, and serialization tests for models."""

from __future__ import annotations

import io
import unittest

import torch
from torch.nn import functional as F

from neurips_permutations.models import (
    CausalMLP,
    CausalTransformer,
    ModelConfig,
    build_model,
    count_parameters,
)


VOCAB_SIZE = 53
MAX_SEQUENCE = 16
D_MODEL = 32


def _model_configs(*, dropout: float = 0.0) -> tuple[ModelConfig, ModelConfig]:
    common = {
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQUENCE,
        "d_model": D_MODEL,
        "layers": 2,
        "dropout": dropout,
        "mlp_ratio": 2.0,
    }
    return (
        ModelConfig(model_type="transformer", n_heads=4, **common),
        ModelConfig(model_type="mlp", **common),
    )


class ConfigurationTests(unittest.TestCase):
    def test_builder_returns_requested_architecture(self) -> None:
        transformer = build_model(_model_configs()[0])
        mlp = build_model("causal_mlp", **{
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": MAX_SEQUENCE,
            "d_model": D_MODEL,
            "layers": 1,
        })
        self.assertIsInstance(transformer, CausalTransformer)
        self.assertIsInstance(mlp, CausalMLP)
        self.assertEqual(transformer.config.resolved_n_heads, 4)

    def test_builder_accepts_keywords_and_aliases(self) -> None:
        model = build_model(
            vocab_size=VOCAB_SIZE,
            max_seq_len=MAX_SEQUENCE,
            d_model=D_MODEL,
            layers=1,
            model_type="decoder_transformer",
        )
        self.assertIsInstance(model, CausalTransformer)

    def test_config_rejects_invalid_values(self) -> None:
        invalid_overrides = (
            {"vocab_size": 0},
            {"max_seq_len": 0},
            {"d_model": -1},
            {"layers": 0},
            {"dropout": 1.0},
            {"model_type": "rnn"},
            {"n_heads": 3},
            {"mlp_ratio": 0},
        )
        base = {
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": MAX_SEQUENCE,
            "d_model": D_MODEL,
            "layers": 2,
        }
        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaises((TypeError, ValueError)):
                    ModelConfig(**(base | override))

    def test_embedding_tying_is_default_and_optional(self) -> None:
        tied = CausalTransformer(
            VOCAB_SIZE, MAX_SEQUENCE, D_MODEL, 1, n_heads=4
        )
        untied = CausalMLP(
            VOCAB_SIZE,
            MAX_SEQUENCE,
            D_MODEL,
            1,
            tie_embeddings=False,
        )
        self.assertIs(tied.token_embedding.weight, tied.lm_head.weight)
        self.assertIsNot(untied.token_embedding.weight, untied.lm_head.weight)


class ForwardAndGradientTests(unittest.TestCase):
    def test_shapes_and_finite_backward_on_available_accelerator(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]], device=device
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], device=device
        )
        targets = torch.tensor(
            [[2, 3, 4, 5, 6, 7, 8], [6, 5, 4, 3, 2, 1, 0]], device=device
        )

        for config in _model_configs(dropout=0.1):
            with self.subTest(model_type=config.model_type, device=device.type):
                torch.manual_seed(17)
                model = build_model(config).to(device)
                model.train()
                logits = model(input_ids, attention_mask)
                self.assertEqual(tuple(logits.shape), (2, 7, VOCAB_SIZE))
                self.assertTrue(torch.isfinite(logits).all().item())

                loss = F.cross_entropy(
                    logits[attention_mask.bool()], targets[attention_mask.bool()]
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(
                            torch.isfinite(parameter.grad).all().item(), name
                        )
                model.zero_grad(set_to_none=True)

    def test_input_validation(self) -> None:
        model = build_model(_model_configs()[0])
        with self.assertRaises(ValueError):
            model(torch.ones(3, dtype=torch.long))
        with self.assertRaises(TypeError):
            model(torch.ones(1, 3, dtype=torch.float32))
        with self.assertRaises(ValueError):
            model(torch.ones(1, MAX_SEQUENCE + 1, dtype=torch.long))
        with self.assertRaises(ValueError):
            model(
                torch.ones(1, 3, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.bool),
            )


class CausalityAndPaddingTests(unittest.TestCase):
    def test_changing_suffix_cannot_change_prefix_logits(self) -> None:
        original = torch.tensor([[4, 8, 15, 16, 23, 42, 3, 9]])
        changed_suffix = original.clone()
        changed_suffix[:, 4:] = torch.tensor([[2, 5, 7, 11]])
        prefix_length = 4

        for config in _model_configs(dropout=0.2):
            with self.subTest(model_type=config.model_type):
                torch.manual_seed(23)
                model = build_model(config).eval()
                with torch.no_grad():
                    original_logits = model(original)
                    changed_logits = model(changed_suffix)
                torch.testing.assert_close(
                    original_logits[:, :prefix_length],
                    changed_logits[:, :prefix_length],
                    rtol=0.0,
                    atol=1e-6,
                )

    def test_masked_tokens_never_affect_valid_tokens(self) -> None:
        first = torch.tensor([[3, 5, 7, 11, 13, 17]])
        second = first.clone()
        second[:, 2] = 41
        second[:, 4] = 43
        mask = torch.tensor([[1, 1, 0, 1, 0, 1]], dtype=torch.bool)

        for config in _model_configs():
            with self.subTest(model_type=config.model_type):
                torch.manual_seed(29)
                model = build_model(config).eval()
                with torch.no_grad():
                    first_logits = model(first, mask)
                    second_logits = model(second, mask)
                torch.testing.assert_close(
                    first_logits[mask], second_logits[mask], rtol=0.0, atol=1e-6
                )
                self.assertTrue(torch.equal(first_logits[~mask], torch.zeros_like(first_logits[~mask])))
                self.assertTrue(torch.equal(second_logits[~mask], torch.zeros_like(second_logits[~mask])))

    def test_fully_padded_sequences_are_finite_zeros(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for config in _model_configs():
            with self.subTest(model_type=config.model_type):
                model = build_model(config).eval()
                with torch.no_grad():
                    logits = model(input_ids, mask)
                self.assertTrue(torch.isfinite(logits).all().item())
                self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))


class PersistenceAndScaleTests(unittest.TestCase):
    def test_state_dict_round_trip_is_reproducible(self) -> None:
        input_ids = torch.tensor([[1, 4, 9, 16, 25]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]])

        for config in _model_configs(dropout=0.25):
            with self.subTest(model_type=config.model_type):
                torch.manual_seed(31)
                original = build_model(config).eval()
                with torch.no_grad():
                    expected = original(input_ids, attention_mask)

                buffer = io.BytesIO()
                torch.save(original.state_dict(), buffer)
                buffer.seek(0)
                restored = build_model(config)
                state = torch.load(buffer, map_location="cpu", weights_only=True)
                restored.load_state_dict(state)
                restored.eval()
                with torch.no_grad():
                    actual = restored(input_ids, attention_mask)
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
                if config.tie_embeddings:
                    self.assertIs(
                        restored.token_embedding.weight, restored.lm_head.weight
                    )

    def test_parameter_count_is_reasonable_at_1024_tokens(self) -> None:
        for model_type in ("transformer", "mlp"):
            with self.subTest(model_type=model_type):
                config = ModelConfig(
                    vocab_size=136,
                    max_seq_len=1024,
                    d_model=64,
                    layers=4,
                    dropout=0.0,
                    model_type=model_type,
                    n_heads=8,
                    mlp_ratio=2.0,
                )
                model = build_model(config)
                parameter_count = count_parameters(model)
                self.assertGreater(parameter_count, 0)
                self.assertLess(parameter_count, 20_000_000)
                unique_manual_count = sum(
                    parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad
                )
                self.assertEqual(parameter_count, unique_manual_count)


if __name__ == "__main__":
    unittest.main()
