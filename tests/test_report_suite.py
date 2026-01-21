import json
import tempfile
import unittest
from pathlib import Path

from symbol_grounding.scripts.report_suite import main as report_suite_main


class TestReportSuite(unittest.TestCase):
    def test_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_dir = root / "report"

            # Dummy files
            base_image = root / "base.png"
            edit_dir = root / "edits" / "edit_000_strength0p50"
            edit_dir.mkdir(parents=True, exist_ok=True)
            edited = edit_dir / "edited.png"
            null_edited = root / "null.png"
            mask = root / "mask.png"
            for p in [base_image, edited, null_edited, mask]:
                p.write_bytes(b"")

            meta = {
                "base_prompt": "a red cat",
                "generate_prompt": "a red cat on a table",
                "edit_base_prompt": "a cat on a table",
                "edit_prompt": "a blue cat",
                "semantic_text": "a blue cat",
                "semantic_neg_text": "a red cat",
                "mask_pad_px": 8,
                "mask_blur": 8.0,
                "strength": 0.5,
                "negative_prompt": "low quality",
                "seed": 42,
            }
            (edit_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

            summary = {
                "summary": {
                    "pareto_front": [
                        {
                            "case": 0,
                            "seed": 0,
                            "edit_index": 0,
                            "target": "obj1",
                            "target_id_spec": "obj1",
                            "target_noun_spec": "cat",
                            "target_id_resolved": "obj1",
                            "target_noun_resolved": "cat",
                            "strength": 0.5,
                            "outside_mean_abs_diff_adj": 0.1,
                            "outside_mean_abs_diff_raw": 0.2,
                            "outside_mean_abs_diff_null": 0.05,
                            "outside_mean_abs_diff_delta_raw": 0.15,
                            "outside_mean_abs_diff_delta_clipped": 0.15,
                            "clip_similarity_delta_raw": 0.2,
                            "paths": {
                                "base_image": str(base_image),
                                "edited": str(edited),
                                "null_edited": str(null_edited),
                            },
                        }
                    ]
                },
                "rows": [
                    {
                        "case": 0,
                        "seed": 0,
                        "edit_index": 0,
                        "target": "obj1",
                        "target_id_spec": "obj1",
                        "target_noun_spec": "cat",
                        "target_id_resolved": "obj1",
                        "target_noun_resolved": "cat",
                        "strength": 0.5,
                        "outside_mean_abs_diff_adj": 0.1,
                        "outside_mean_abs_diff_raw": 0.2,
                        "outside_mean_abs_diff_null": 0.05,
                        "outside_mean_abs_diff_delta_raw": 0.15,
                        "outside_mean_abs_diff_delta_clipped": 0.15,
                        "clip_similarity_delta_raw": 0.2,
                        "base_image": str(base_image),
                        "edited": str(edited),
                        "null_edited": str(null_edited),
                        "mask": str(mask),
                    }
                ],
            }
            summary_path = root / "locality_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            suite_index = {
                "entries": [
                    {
                        "experiment_name": "dummy",
                        "status": "ok",
                        "summary_json": str(summary_path),
                    }
                ]
            }
            suite_index_path = root / "suite_index.json"
            suite_index_path.write_text(json.dumps(suite_index), encoding="utf-8")

            report_suite_main(["--suite-index", str(suite_index_path), "--out", str(report_dir), "--topk", "2"])

            report_md = report_dir / "report.md"
            self.assertTrue(report_md.exists())
            report_text = report_md.read_text(encoding="utf-8")
            self.assertIn("base_prompt: a red cat", report_text)
            self.assertIn("generate_prompt: a red cat on a table", report_text)
            self.assertIn("edit_base_prompt: a cat on a table", report_text)
            self.assertIn("edit_prompt: a blue cat", report_text)
            self.assertIn("semantic_neg_text: a red cat", report_text)
            self.assertIn("negative_prompt: low quality", report_text)
            self.assertIn("target_id_resolved: obj1", report_text)
            self.assertIn("outside_mean_abs_diff_delta_raw: 0.15", report_text)


if __name__ == "__main__":
    unittest.main()
