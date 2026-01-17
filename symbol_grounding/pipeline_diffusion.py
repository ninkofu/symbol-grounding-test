"""Diffusion pipeline orchestrator with optional ControlNet conditioning."""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Optional

from .scene_graph import parse_text
from .layout import generate_layout
from .diffusion import DiffusionConfig, layout_to_control_image, generate_with_diffusers
from .utils import SceneGraph, Layout, safe_prompt_slug


@dataclass
class DiffusionResult:
    image_path: str
    control_path: str
    seed: int
    scene_graph: SceneGraph
    layout: Layout


def generate_diffusion_image(
    prompt: str,
    output_dir: str,
    config: DiffusionConfig,
    control_mode: str = "scribble",
    scene_graph: Optional[SceneGraph] = None,
    layout: Optional[Layout] = None,
) -> DiffusionResult:
    """Generate an image with diffusers and save control image to disk."""
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

    image = generate_with_diffusers(
        prompt=prompt,
        config=config,
        control_image=control_image if (config.use_controlnet or config.controlnet_model_id) else None,
    )
    image.save(image_path)

    return DiffusionResult(
        image_path=image_path,
        control_path=control_path,
        seed=config.seed,
        scene_graph=scene_graph,
        layout=layout,
    )


__all__ = ["DiffusionResult", "generate_diffusion_image"]
