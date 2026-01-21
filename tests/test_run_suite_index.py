import json
import tempfile
import unittest
from pathlib import Path

from symbol_grounding.scripts.run_suite import main as run_suite_main


class TestRunSuiteIndex(unittest.TestCase):
    def test_suite_index_generation_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configs_dir = root / "configs"
            out_dir = root / "suite_out"
            configs_dir.mkdir(parents=True, exist_ok=True)

            for name in ["a.json", "b.json"]:
                (configs_dir / name).write_text(json.dumps({"name": name}), encoding="utf-8")

            run_suite_main(
                [
                    "--configs-dir",
                    str(configs_dir),
                    "--out",
                    str(out_dir),
                    "--pattern",
                    "*.json",
                    "--dry-run",
                ]
            )

            suite_index = out_dir / "suite_index.json"
            self.assertTrue(suite_index.exists())
            payload = json.loads(suite_index.read_text(encoding="utf-8"))
            entries = payload.get("entries", [])
            self.assertEqual(len(entries), 2)
            for entry in entries:
                self.assertEqual(entry.get("status"), "dry-run")


if __name__ == "__main__":
    unittest.main()
