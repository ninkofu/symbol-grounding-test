import json
import tempfile
import unittest
from pathlib import Path

from symbol_grounding.scripts.plot_suite import main as plot_suite_main


class TestPlotSuiteParse(unittest.TestCase):
    def test_plot_suite_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "plots"
            summary_path = root / "locality_summary.json"

            summary_payload = {
                "summary": {
                    "pareto_front": [
                        {
                            "case": 0,
                            "seed": 0,
                            "edit_index": 0,
                            "target": "obj1",
                            "strength": 0.5,
                            "outside_mean_abs_diff_adj": 0.1,
                            "outside_frac_changed_adj": 0.01,
                            "clip_similarity_delta_raw": 0.2,
                            "paths": {
                                "base_image": "base.png",
                                "edited": "edited.png",
                                "null_edited": "null.png",
                            },
                        }
                    ],
                    "summary_by_strength": {
                        "0.50": {
                            "outside_mean_abs_diff_adj": {"n": 1, "mean": 0.1, "median": 0.1},
                            "clip_similarity_delta_raw": {"n": 1, "mean": 0.2, "median": 0.2},
                        }
                    },
                }
            }
            summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

            suite_index = {
                "entries": [
                    {
                        "config_path": "dummy.json",
                        "experiment_name": "dummy",
                        "status": "ok",
                        "summary_json": str(summary_path),
                    }
                ]
            }
            suite_index_path = root / "suite_index.json"
            suite_index_path.write_text(json.dumps(suite_index), encoding="utf-8")

            plot_suite_main(["--suite-index", str(suite_index_path), "--out", str(out_dir)])

            self.assertTrue((out_dir / "pareto_dummy.csv").exists())
            self.assertTrue((out_dir / "strength_dummy.csv").exists())
            self.assertTrue((out_dir / "suite_pareto.csv").exists())


if __name__ == "__main__":
    unittest.main()
