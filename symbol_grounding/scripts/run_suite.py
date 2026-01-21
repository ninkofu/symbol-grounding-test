"""Run a suite of experiment configs and aggregate results."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..experiments import run_experiment
from .aggregate_locality import main as aggregate_main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple experiment configs as a suite.")
    parser.add_argument("--configs-dir", required=True, help="Directory containing experiment JSON configs")
    parser.add_argument("--out", required=True, help="Suite output root directory")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern to select configs (default: *.json)")
    parser.add_argument("--aggregate", action="store_true", help="Run aggregate_locality after each experiment")
    parser.add_argument("--dry-run", action="store_true", help="Do not run experiments; only emit suite_index.json")
    parser.add_argument("--skip-generate", action="store_true", help="Skip base image generation")
    parser.add_argument("--skip-edit", action="store_true", help="Skip editing")
    parser.add_argument("--skip-eval", action="store_true", help="Skip locality evaluation")
    return parser


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    configs_dir = Path(args.configs_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    config_paths = sorted(configs_dir.glob(args.pattern))
    if not config_paths:
        print(f"[WARN] No configs matched pattern '{args.pattern}' in {configs_dir}")

    entries: List[Dict[str, Any]] = []

    for config_path in config_paths:
        entry: Dict[str, Any] = {
            "config_path": str(config_path),
            "experiment_name": config_path.stem,
            "experiment_dir": None,
            "summary_json": None,
            "summary_csv": None,
            "status": "pending",
            "error": None,
        }

        try:
            if args.dry_run:
                entry["status"] = "dry-run"
                print(f"[DRY-RUN] {config_path}")
            else:
                print(f"[RUN] {config_path}")
                exp_dir = run_experiment(
                    config_path=str(config_path),
                    output_root=str(out_root),
                    skip_generate=args.skip_generate,
                    skip_edit=args.skip_edit,
                    skip_eval=args.skip_eval,
                )
                entry["experiment_dir"] = exp_dir

                if args.aggregate:
                    try:
                        aggregate_main(["--results", exp_dir])
                        entry["summary_json"] = str(Path(exp_dir) / "locality_summary.json")
                        entry["summary_csv"] = str(Path(exp_dir) / "locality_summary.csv")
                    except SystemExit as exc:
                        code = exc.code
                        if code in (0, None):
                            entry["summary_json"] = str(Path(exp_dir) / "locality_summary.json")
                            entry["summary_csv"] = str(Path(exp_dir) / "locality_summary.csv")
                        else:
                            entry["status"] = "error"
                            entry["error"] = f"aggregate failed: SystemExit({code})"
                            print(
                                f"[ERROR] aggregate failed for {config_path}: SystemExit({code})",
                                file=sys.stderr,
                            )
                            entries.append(entry)
                            continue
                    except Exception as exc:
                        entry["status"] = "error"
                        entry["error"] = f"aggregate failed: {exc}"
                        print(f"[ERROR] aggregate failed for {config_path}: {exc}", file=sys.stderr)
                        entries.append(entry)
                        continue

                entry["status"] = "ok"
                print(f"[OK] {config_path} -> {exp_dir}")

        except ImportError as exc:
            entry["status"] = "error"
            entry["error"] = f"missing dependency: {exc}"
            print(f"[SKIP] {config_path} missing dependency: {exc}", file=sys.stderr)
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"[ERROR] {config_path}: {exc}", file=sys.stderr)

        entries.append(entry)

    suite_index = {
        "suite_root": str(out_root),
        "configs_dir": str(configs_dir),
        "pattern": args.pattern,
        "aggregate": bool(args.aggregate),
        "dry_run": bool(args.dry_run),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entries": entries,
    }

    suite_path = out_root / "suite_index.json"
    _write_json(suite_path, suite_index)
    print(f"[DONE] suite_index.json written to {suite_path}")


if __name__ == "__main__":
    main()
