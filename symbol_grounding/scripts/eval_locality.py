"""CLI for locality evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from ..eval.locality import evaluate_locality


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate locality (mask leakage) between two images.")
    parser.add_argument("--before", required=True, help="Path to the pre-edit image")
    parser.add_argument("--after", required=True, help="Path to the edited image")
    parser.add_argument("--mask", required=True, help="Path to the mask image")
    parser.add_argument("--out", required=True, help="Output JSON path for metrics")
    parser.add_argument("--threshold", type=float, default=10 / 255, help="Change threshold (0..1)")
    parser.add_argument("--save-diff", default=None, help="Optional path to save diff image")
    parser.add_argument("--save-heatmap", default=None, help="Optional path to save heatmap image")
    parser.add_argument("--save-overlay", default=None, help="Optional path to save overlay image")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        metrics = evaluate_locality(
            before_path=args.before,
            after_path=args.after,
            mask_path=args.mask,
            threshold=args.threshold,
            out_path=args.out,
            save_diff=args.save_diff,
            save_heatmap=args.save_heatmap,
            save_overlay=args.save_overlay,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
