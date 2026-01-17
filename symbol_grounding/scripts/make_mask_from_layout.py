"""CLI to generate a mask from a prompt-derived layout."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from ..layout import generate_layout, save_layout_mask
from ..scene_graph import parse_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a mask from layout and target object id.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate scene graph/layout")
    parser.add_argument("--target", required=True, help="Target object id (e.g., obj1)")
    parser.add_argument("--out", default="outputs/mask.png", help="Output mask path")
    parser.add_argument("--image-size", type=int, default=512, help="Mask image size (square)")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        scene_graph = parse_text(args.prompt)
        layout = generate_layout(scene_graph)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        path = save_layout_mask(layout, target_id=args.target, output_path=args.out, image_size=args.image_size)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Mask saved to {path}")


if __name__ == "__main__":
    main()
