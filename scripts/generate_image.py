"""Command‑line entry point to generate images from text prompts.

This script exposes a simple interface to the symbol grounding pipeline.
Run it with a prompt string to produce a placeholder image that
illustrates the layout and object parsing.  Example usage::

    python -m symbol_grounding.scripts.generate_image \
        --prompt "a red cat on a table" \
        --output-dir outputs

The generated image will be saved in the specified directory with a
unique name derived from the prompt and timestamp.
"""
from __future__ import annotations

import argparse
import sys

from ..pipeline import generate_image


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using the symbol grounding pipeline.")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the scene to generate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to store the generated image")
    args = parser.parse_args(argv)
    image_path = generate_image(args.prompt, output_dir=args.output_dir)
    print(f"Image saved to {image_path}")


if __name__ == "__main__":
    main()