"""CLI to generate images with diffusers via the unified pipeline."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..diffusion import DiffusionConfig
from ..pipeline_diffusion import generate_diffusion_image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image using diffusers (Stable Diffusion)."
    )
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
    parser.add_argument(
        "--control-mode",
        default="scribble",
        help="Control image mode: scribble|seg",
    )
    parser.add_argument("--device", default="auto", help="Device: auto/cuda/cpu")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt (optional)")
    parser.add_argument("--attention-slicing", action="store_true", help="Enable attention slicing for low VRAM")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use local model files only")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    use_controlnet = bool(args.controlnet_model_id)
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
        use_controlnet=use_controlnet,
        controlnet_model_id=args.controlnet_model_id,
        controlnet_conditioning_scale=args.controlnet_scale,
    )

    try:
        result = generate_diffusion_image(
            prompt=args.prompt,
            output_dir=args.output_dir,
            config=config,
            control_mode=args.control_mode,
        )
    except ImportError as exc:
        message = str(exc)
        print(
            "Missing dependencies for diffusers/control images. Install with:\n"
            "  pip install diffusers transformers accelerate safetensors pillow",
            file=sys.stderr,
        )
        print(message, file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        if "Failed to load diffusers pipeline" in str(exc):
            print(
                "Hint: check your model IDs and network access, or try --local-files-only.",
                file=sys.stderr,
            )
        sys.exit(1)

    print(f"Image saved to {result.image_path}")
    print(f"Control image saved to {result.control_path}")


if __name__ == "__main__":
    main()
