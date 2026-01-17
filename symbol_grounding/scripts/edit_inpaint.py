"""CLI for inpainting-based editing."""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from typing import Optional

from ..editing import EditRequest, InpaintEditor
from ..utils import safe_prompt_slug


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edit an image using inpainting.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--mask", default=None, help="Path to the mask image (white = edit)")
    parser.add_argument("--prompt", required=True, help="Prompt describing the edited region")
    parser.add_argument("--base-prompt", default=None, help="Base prompt describing the original scene (optional)")
    parser.add_argument(
        "--keep-prompt",
        default=None,
        help="(Deprecated) Alias for --base-prompt. Will not be used as negative prompt.",
    )
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt (optional)")
    parser.add_argument("--auto-mask-from-prompt", action="store_true", help="Generate mask from prompt + target object")
    parser.add_argument("--target", default=None, help="Target object id (e.g., obj1) for auto mask")
    parser.add_argument("--mask-pad-px", type=int, default=0, help="Pad auto mask bbox in pixels")
    parser.add_argument("--mask-blur", type=float, default=0.0, help="Gaussian blur radius for auto mask")
    parser.add_argument("--list-objects", action="store_true", help="List detected objects and exit")
    parser.add_argument("--out", "--output-dir", dest="output_dir", default="outputs", help="Output directory")
    parser.add_argument("--model", dest="model_id", default="runwayml/stable-diffusion-inpainting", help="Inpainting model id")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=30, help="Sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--height", type=int, default=None, help="Output height (optional)")
    parser.add_argument("--width", type=int, default=None, help="Output width (optional)")
    parser.add_argument("--device", default="auto", help="Device: auto/cuda/cpu")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use local model files only")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_objects:
        try:
            from ..scene_graph import parse_text
            from ..layout import generate_layout
        except Exception as exc:
            print(f"[ERROR] Failed to import layout utilities: {exc}", file=sys.stderr)
            sys.exit(1)

        layout_prompt = args.base_prompt or args.prompt
        try:
            scene_graph = parse_text(layout_prompt)
            layout = generate_layout(scene_graph)
            for obj in scene_graph.objects:
                bbox = layout.boxes.get(obj.id)
                attrs = (
                    "{"
                    + ",".join(f"{k}={v}" for k, v in obj.attributes.items())
                    + "}"
                    if obj.attributes
                    else "{}"
                )
                if bbox is None:
                    bbox_str = "bbox=(missing)"
                else:
                    bbox_str = (
                        f"bbox=(x={bbox.x:.2f},y={bbox.y:.2f},w={bbox.width:.2f},h={bbox.height:.2f})"
                    )
                print(f"{obj.id}: {obj.noun} attrs={attrs} {bbox_str}")
            return
        except Exception as exc:
            print(f"[ERROR] Failed to list objects: {exc}", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(args.image):
        print(f"[ERROR] Input image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.mask is None and not args.auto_mask_from_prompt:
        print("[ERROR] Provide --mask or enable --auto-mask-from-prompt.", file=sys.stderr)
        sys.exit(1)

    if args.mask is not None and not os.path.exists(args.mask):
        print(f"[ERROR] Mask image not found: {args.mask}", file=sys.stderr)
        sys.exit(1)

    base_prompt = args.base_prompt or args.keep_prompt

    mask_image = None
    if args.auto_mask_from_prompt:
        if not args.target:
            print("[ERROR] --target is required when using --auto-mask-from-prompt.", file=sys.stderr)
            sys.exit(1)
        try:
            from ..scene_graph import parse_text
            from ..layout import generate_layout, layout_to_mask
        except Exception as exc:
            print(f"[ERROR] Failed to import layout utilities: {exc}", file=sys.stderr)
            sys.exit(1)

        layout_prompt = base_prompt or args.prompt
        try:
            scene_graph = parse_text(layout_prompt)
            layout = generate_layout(scene_graph)
            mask_image = layout_to_mask(
                layout,
                target_id=args.target,
                pad_px=args.mask_pad_px,
                blur=args.mask_blur,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to auto-generate mask: {exc}", file=sys.stderr)
            sys.exit(1)

    request = EditRequest(
        image_path=args.image,
        mask_path=args.mask,
        mask_image=mask_image,
        edit_prompt=args.prompt,
        base_prompt=base_prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    editor = InpaintEditor()
    try:
        result = editor.edit(request)
    except ImportError as exc:
        message = str(exc)
        if "Pillow" in message or "pillow" in message:
            print("Missing dependency: pillow. Install with `pip install pillow`.", file=sys.stderr)
        else:
            print(
                "Missing dependencies for diffusers. Install with:\n"
                "  pip install diffusers transformers accelerate safetensors",
                file=sys.stderr,
            )
        print(message, file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = safe_prompt_slug(args.prompt)
    filename = f"edit_{safe_prompt}_{timestamp}_seed{args.seed}.png"
    output_path = os.path.join(args.output_dir, filename)
    result.image.save(output_path)
    print(f"Edited image saved to {output_path}")


if __name__ == "__main__":
    main()
