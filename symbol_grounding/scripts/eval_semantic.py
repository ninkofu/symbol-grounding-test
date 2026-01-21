"""CLI for semantic success evaluation using CLIP."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from ..eval.semantic import evaluate_semantic


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate semantic success using CLIP on a masked crop.")
    parser.add_argument("--after", required=True, help="Path to edited image")
    parser.add_argument("--mask", required=True, help="Path to mask image")
    parser.add_argument("--text", required=True, help="Text to compare with crop")
    parser.add_argument("--neg-text", default=None, help="Optional negative text to compute margin")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--pad-px", type=int, default=8, help="Padding in pixels for crop bbox")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32", help="CLIP model id")
    parser.add_argument("--device", default="auto", help="Device: auto/cuda/cpu")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = evaluate_semantic(
            after_path=args.after,
            mask_path=args.mask,
            text=args.text,
            out_path=None,
            model_id=args.model_id,
            device=args.device,
            pad_px=args.pad_px,
        )
        if args.neg_text:
            neg_result = evaluate_semantic(
                after_path=args.after,
                mask_path=args.mask,
                text=args.neg_text,
                out_path=None,
                model_id=args.model_id,
                device=args.device,
                pad_px=args.pad_px,
            )
            result["neg_text"] = args.neg_text
            result["clip_similarity_neg"] = float(neg_result["clip_similarity"])
            result["clip_margin"] = float(result["clip_similarity"]) - float(neg_result["clip_similarity"])
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
