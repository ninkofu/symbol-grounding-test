"""CLI to run experiments from a JSON config."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..experiments import run_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run symbol grounding experiments from config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--out", required=True, help="Output root directory")
    parser.add_argument("--skip-generate", action="store_true", help="Skip base image generation")
    parser.add_argument("--skip-edit", action="store_true", help="Skip editing")
    parser.add_argument("--skip-eval", action="store_true", help="Skip locality evaluation")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        exp_dir = run_experiment(
            config_path=args.config,
            output_root=args.out,
            skip_generate=args.skip_generate,
            skip_edit=args.skip_edit,
            skip_eval=args.skip_eval,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Experiment outputs saved to {exp_dir}")


if __name__ == "__main__":
    main()
