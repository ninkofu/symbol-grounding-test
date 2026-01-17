import unittest

from symbol_grounding.scene_graph import parse_text


class TestSceneGraph(unittest.TestCase):
    def test_parse_text_basic(self) -> None:
        sg = parse_text("a red cat on a table")
        nouns = [obj.noun for obj in sg.objects]
        self.assertIn("cat", nouns)
        self.assertIn("table", nouns)

        cat_obj = next(obj for obj in sg.objects if obj.noun == "cat")
        table_obj = next(obj for obj in sg.objects if obj.noun == "table")
        self.assertEqual(cat_obj.attributes.get("color"), "red")

        self.assertIn((cat_obj.id, "on", table_obj.id), sg.relations)


if __name__ == "__main__":
    unittest.main()
