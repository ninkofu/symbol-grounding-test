"""Command-line entry point for the System 2 planning pipeline."""
from __future__ import annotations

import argparse
from typing import Optional

from ..system2_pipeline import System2Pipeline


def _parse_layout_edits(edits: Optional[list[str]]) -> list[tuple[str, float, float]]:
    if not edits:
        return []
    parsed: list[tuple[str, float, float]] = []
    for item in edits:
        try:
            obj_id, dx, dy = item.split(":")
            parsed.append((obj_id, float(dx), float(dy)))
        except ValueError as exc:
            raise ValueError(f"Invalid --move format: {item}. Expected obj_id:dx:dy") from exc
    return parsed


def _parse_attribute_edits(edits: Optional[list[str]]) -> list[tuple[str, str, str]]:
    if not edits:
        return []
    parsed: list[tuple[str, str, str]] = []
    for item in edits:
        try:
            obj_id, rest = item.split(":", 1)
            key, value = rest.split("=", 1)
            parsed.append((obj_id, key, value))
        except ValueError as exc:
            raise ValueError(f"Invalid --set-attr format: {item}. Expected obj_id:key=value") from exc
    return parsed


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the System 2 symbolic pipeline.")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the scene to generate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to store outputs")
    parser.add_argument("--image-size", type=int, default=512, help="Square image size in pixels")
    parser.add_argument(
        "--move",
        action="append",
        help="Layout edit: obj_id:dx:dy (normalized offsets, may repeat)",
    )
    parser.add_argument(
        "--set-attr",
        action="append",
        help="Attribute edit: obj_id:key=value (may repeat)",
    )
    parser.add_argument(
        "--lock-token",
        action="append",
        help="Token to lock in the attention plan (may repeat)",
    )

    args = parser.parse_args(argv)
    pipeline = System2Pipeline()
    layout_edits = _parse_layout_edits(args.move)
    attribute_edits = _parse_attribute_edits(args.set_attr)
    result = pipeline.run(
        prompt=args.prompt,
        output_dir=args.output_dir,
        image_size=args.image_size,
        layout_edits=layout_edits,
        attribute_edits=attribute_edits,
        lock_tokens=args.lock_token,
    )

    print(f"Plan saved to {result.plan_path}")
    print(f"Layout saved to {result.layout_path}")
    print(f"Image saved to {result.image_path}")


if __name__ == "__main__":
    main()
