"""Grounded diffusion pipeline with ControlNet and attention locking."""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from .attention_control import prompt_to_prompt_edit
from .diffusion import DiffusionConfig, layout_to_control_image, load_pipeline
from .layout import generate_layout
from .scene_graph import parse_text
from .utils import SceneGraph, Layout, safe_prompt_slug


@dataclass
class GroundedDiffusionResult:
    image_path: str
    control_path: str
    seed: int
    scene_graph: SceneGraph
    layout: Layout


def generate_grounded_image(
    prompt: str,
    output_dir: str,
    config: DiffusionConfig,
    control_mode: str = "scribble",
    scene_graph: Optional[SceneGraph] = None,
    layout: Optional[Layout] = None,
    base_prompt: Optional[str] = None,
    lock_tokens: Optional[Iterable[str]] = None,
) -> GroundedDiffusionResult:
    """Generate an image using layout control and optional attention locking."""
    if scene_graph is None:
        scene_graph = parse_text(prompt)
    if layout is None:
        layout = generate_layout(scene_graph)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = safe_prompt_slug(prompt)
    base_name = f"{safe_prompt}_{timestamp}_seed{config.seed}"

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{base_name}.png")
    control_path = os.path.join(output_dir, f"{base_name}_control.png")

    control_image = layout_to_control_image(
        layout,
        scene_graph=scene_graph,
        image_size=(config.width, config.height),
        mode=control_mode,
    )
    control_image.save(control_path)

    pipe, device, _ = load_pipeline(config)
    generator = None
    if device is not None:
        import torch

        generator = torch.Generator(device=device).manual_seed(config.seed)

    extra_kwargs = {
        "negative_prompt": config.negative_prompt,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "generator": generator,
        "height": config.height,
        "width": config.width,
    }
    if config.use_controlnet or config.controlnet_model_id:
        extra_kwargs["image"] = control_image
        extra_kwargs["controlnet_conditioning_scale"] = config.controlnet_conditioning_scale

    if base_prompt and lock_tokens:
        result = prompt_to_prompt_edit(
            pipe,
            base_prompt=base_prompt,
            edit_prompt=prompt,
            lock_tokens=lock_tokens,
            **extra_kwargs,
        )
        image = result.images[0]
    else:
        result = pipe(prompt=prompt, **extra_kwargs)
        image = result.images[0]

    image.save(image_path)

    return GroundedDiffusionResult(
        image_path=image_path,
        control_path=control_path,
        seed=config.seed,
        scene_graph=scene_graph,
        layout=layout,
    )


__all__ = ["GroundedDiffusionResult", "generate_grounded_image"]
