import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("numpy is required for locality eval tests")

from symbol_grounding.eval.locality import compute_locality_metrics


class TestLocalityEval(unittest.TestCase):
    def test_outside_zero_when_only_inside_changes(self) -> None:
        before = np.zeros((4, 4, 3), dtype=np.float32)
        after = before.copy()
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        after[1:3, 1:3] = 1.0
        metrics = compute_locality_metrics(before, after, mask, threshold=0.1)
        self.assertEqual(metrics["outside_mean_abs_diff"], 0.0)

    def test_outside_positive_when_outside_changes(self) -> None:
        before = np.zeros((4, 4, 3), dtype=np.float32)
        after = before.copy()
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        after[0, 0] = 1.0
        metrics = compute_locality_metrics(before, after, mask, threshold=0.1)
        self.assertGreater(metrics["outside_mean_abs_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
