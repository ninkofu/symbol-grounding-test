"""Command-line entry point to generate images from text prompts."""
from __future__ import annotations

import argparse
from typing import Optional

from ..pipeline import generate_image


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a placeholder image from a text prompt."
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the scene to generate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to store the generated image")
    parser.add_argument("--image-size", type=int, default=512, help="Square image size in pixels")
    args = parser.parse_args(argv)
    image_path = generate_image(args.prompt, output_dir=args.output_dir, image_size=args.image_size)
    print(f"Image saved to {image_path}")


if __name__ == "__main__":
    main()
