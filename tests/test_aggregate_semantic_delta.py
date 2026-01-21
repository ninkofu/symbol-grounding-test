import json
import tempfile
import unittest
from pathlib import Path

from symbol_grounding.scripts.aggregate_locality import main as aggregate_main


class TestAggregateSemanticDelta(unittest.TestCase):
    def test_clip_delta_and_pareto(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "case_000_seed000"
            base_dir = case_dir / "base"
            edit_dir = case_dir / "edits" / "edit_000"
            null_dir = edit_dir / "null"
            base_dir.mkdir(parents=True, exist_ok=True)
            edit_dir.mkdir(parents=True, exist_ok=True)
            null_dir.mkdir(parents=True, exist_ok=True)

            base_image = base_dir / "image.png"
            edited = edit_dir / "edited.png"
            mask = edit_dir / "mask.png"
            null_edited = null_dir / "edited.png"

            for p in [base_image, edited, mask, null_edited]:
                p.write_bytes(b"")

            results = {
                "case": 0,
                "seed": 0,
                "edit_index": 0,
                "target": "obj1",
                "strength": 0.5,
                "base_prompt": "a red cat",
                "paths": {
                    "base_image": str(base_image),
                    "edited": str(edited),
                    "mask": str(mask),
                    "metrics": None,
                },
                "metrics": {
                    "outside_mean_abs_diff": 0.3,
                    "outside_mean_sq_diff": 0.0,
                    "outside_frac_changed": 0.5,
                    "inside_mean_abs_diff": 0.2,
                    "inside_frac_changed": 0.6,
                    "threshold": 0.1,
                },
                "null": {"edited": str(null_edited), "metrics": None},
                "null_metrics": {
                    "outside_mean_abs_diff": 0.1,
                    "outside_frac_changed": 0.2,
                },
                "semantic_metrics": {"clip_similarity": 0.6, "clip_margin": 0.3},
                "null_semantic_metrics": {"clip_similarity": 0.4, "clip_margin": 0.1},
            }

            results_jsonl = root / "results.jsonl"
            results_jsonl.write_text(json.dumps(results) + "\n", encoding="utf-8")

            out_json = root / "summary.json"
            out_csv = root / "summary.csv"
            aggregate_main(["--results", str(results_jsonl), "--out-json", str(out_json), "--out-csv", str(out_csv)])

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            row = payload["rows"][0]

            self.assertAlmostEqual(row["clip_similarity_delta_raw"], 0.2)
            self.assertAlmostEqual(row["clip_similarity_delta_clipped"], 0.2)
            self.assertAlmostEqual(row["clip_similarity_adj"], 0.2)
            self.assertAlmostEqual(row["clip_margin_delta_raw"], 0.2)
            self.assertAlmostEqual(row["clip_margin_delta_clipped"], 0.2)

            summary = payload["summary"]
            self.assertIn("pareto_front", summary)
            self.assertEqual(len(summary["pareto_front"]), 1)
            self.assertIn("summary_by_strength", summary)
            self.assertIn("0.50", summary["summary_by_strength"])
            self.assertAlmostEqual(
                summary["summary_by_strength"]["0.50"]["clip_similarity_delta_raw"]["mean"], 0.2
            )


if __name__ == "__main__":
    unittest.main()
