import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is required for typed slot tests")

from symbol_grounding.typed_slots import NounModule, AdjectiveModule, VerbModule


class TestTypedSlots(unittest.TestCase):
    def test_noun_module_vq(self) -> None:
        module = NounModule(num_codes=8, code_dim=4, input_dim=4)
        feats = torch.randn(2, 4)
        quantized, indices = module(feats)
        self.assertEqual(quantized.shape, feats.shape)
        self.assertEqual(indices.shape, feats.shape[:-1])
        self.assertIsNotNone(module.last_vq_loss)

    def test_adjective_film(self) -> None:
        module = AdjectiveModule(feature_dim=4, attribute_dim=3, mode="film")
        feats = torch.randn(2, 4)
        attrs = torch.randn(2, 3)
        out = module(feats, attrs)
        self.assertEqual(out.shape, feats.shape)

    def test_verb_fixed(self) -> None:
        module = VerbModule(mode="fixed", clamp=True)
        coords = torch.tensor([[[0.5, 0.5]]])
        out = module(coords, "move_right")
        self.assertGreaterEqual(out[0, 0, 0].item(), coords[0, 0, 0].item())


if __name__ == "__main__":
    unittest.main()
