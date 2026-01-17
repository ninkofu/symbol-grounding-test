import unittest

from symbol_grounding.layout import generate_layout
from symbol_grounding.utils import SceneGraph, SceneObject


class TestLayout(unittest.TestCase):
    def test_layout_non_empty(self) -> None:
        objects = [
            SceneObject(id="obj1", noun="cat"),
            SceneObject(id="obj2", noun="dog"),
            SceneObject(id="obj3", noun="table"),
            SceneObject(id="obj4", noun="chair"),
        ]
        sg = SceneGraph(objects=objects, relations=[])
        layout = generate_layout(sg)
        self.assertEqual(len(layout.boxes), 4)
        for box in layout.boxes.values():
            self.assertGreaterEqual(box.x, 0.0)
            self.assertGreaterEqual(box.y, 0.0)
            self.assertLessEqual(box.x + box.width, 1.0)
            self.assertLessEqual(box.y + box.height, 1.0)

    def test_layout_empty(self) -> None:
        sg = SceneGraph(objects=[], relations=[])
        layout = generate_layout(sg)
        self.assertEqual(layout.boxes, {})


if __name__ == "__main__":
    unittest.main()
