import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("numpy is required for semantic bbox tests")

from symbol_grounding.eval.semantic import compute_bbox_from_mask, expand_bbox


class TestSemanticBBox(unittest.TestCase):
    def test_compute_bbox(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 2:5] = True
        bbox = compute_bbox_from_mask(mask)
        self.assertEqual(bbox, (2, 1, 5, 4))

    def test_expand_bbox(self) -> None:
        bbox = (2, 2, 4, 4)
        expanded = expand_bbox(bbox, (6, 6), pad_px=2)
        self.assertEqual(expanded, (0, 0, 6, 6))


if __name__ == "__main__":
    unittest.main()
