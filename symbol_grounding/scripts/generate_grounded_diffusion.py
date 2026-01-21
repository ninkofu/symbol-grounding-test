"""CLI for grounded diffusion with layout control and attention locks."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..diffusion import DiffusionConfig
from ..pipeline_grounded_diffusion import generate_grounded_image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a grounded diffusion image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to render")
    parser.add_argument("--out", "--output-dir", dest="output_dir", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=30, help="Sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8)")
    parser.add_argument("--model", dest="model_id", default="runwayml/stable-diffusion-v1-5", help="Base model id")
    parser.add_argument("--controlnet-model", dest="controlnet_model_id", default=None, help="ControlNet model id")
    parser.add_argument("--controlnet-scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--control-mode", default="scribble", help="Control image mode: scribble|seg")
    parser.add_argument("--device", default="auto", help="Device: auto/cuda/cpu")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt (optional)")
    parser.add_argument("--attention-slicing", action="store_true", help="Enable attention slicing for low VRAM")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use local model files only")
    parser.add_argument("--base-prompt", default=None, help="Base prompt for attention locking")
    parser.add_argument("--lock-token", action="append", help="Token to lock in the attention plan (may repeat)")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = DiffusionConfig(
        model_id=args.model_id,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device,
        negative_prompt=args.negative_prompt,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        attention_slicing=args.attention_slicing,
        cpu_offload=args.cpu_offload,
        use_controlnet=bool(args.controlnet_model_id),
        controlnet_model_id=args.controlnet_model_id,
        controlnet_conditioning_scale=args.controlnet_scale,
    )

    try:
        result = generate_grounded_image(
            prompt=args.prompt,
            output_dir=args.output_dir,
            config=config,
            control_mode=args.control_mode,
            base_prompt=args.base_prompt,
            lock_tokens=args.lock_token,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Image saved to {result.image_path}")
    print(f"Control image saved to {result.control_path}")


if __name__ == "__main__":
    main()
