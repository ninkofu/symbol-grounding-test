import unittest

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("Pillow is required for layout mask tests")

from symbol_grounding.layout import layout_to_mask
from symbol_grounding.utils import BoundingBox, Layout


class TestLayoutMask(unittest.TestCase):
    def test_mask_padding_increases_area(self) -> None:
        layout = Layout(boxes={"obj1": BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)})
        mask_small = layout_to_mask(layout, target_id="obj1", image_size=64, pad_px=0)
        mask_padded = layout_to_mask(layout, target_id="obj1", image_size=64, pad_px=8)
        self.assertGreater(sum(mask_padded.tobytes()), sum(mask_small.tobytes()))

    def test_mask_blur_keeps_size(self) -> None:
        layout = Layout(boxes={"obj1": BoundingBox(x=0.1, y=0.1, width=0.3, height=0.3)})
        mask_blur = layout_to_mask(layout, target_id="obj1", image_size=64, blur=2.0)
        self.assertEqual(mask_blur.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
