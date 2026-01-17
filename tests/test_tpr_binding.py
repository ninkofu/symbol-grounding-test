import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is required for TPR tests")

from symbol_grounding.tpr import bind, unbind


class TestTPRBinding(unittest.TestCase):
    def test_tpr_binding_no_mix(self) -> None:
        # Two orthonormal roles: shoes and shirt
        roles = torch.eye(2)
        role_shoes = roles[0]
        role_shirt = roles[1]

        # Color fillers (one-hot)
        red = torch.tensor([1.0, 0.0, 0.0])
        blue = torch.tensor([0.0, 1.0, 0.0])
        green = torch.tensor([0.0, 0.0, 1.0])

        memory = bind(red, role_shoes) + bind(blue, role_shirt)

        red_hat = unbind(memory, role_shoes)
        blue_hat = unbind(memory, role_shirt)

        self.assertTrue(torch.allclose(red_hat, red, atol=1e-6))
        self.assertTrue(torch.allclose(blue_hat, blue, atol=1e-6))

        # Change only shoes from red -> green
        memory2 = memory - bind(red, role_shoes) + bind(green, role_shoes)
        green_hat = unbind(memory2, role_shoes)
        blue_hat2 = unbind(memory2, role_shirt)

        self.assertTrue(torch.allclose(green_hat, green, atol=1e-6))
        self.assertTrue(torch.allclose(blue_hat2, blue, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
