"""High-level orchestration of the symbol grounding pipeline."""
from __future__ import annotations

import os
import datetime as _dt
from typing import Optional

from .scene_graph import parse_text
from .layout import generate_layout
from .diffusion_interface import render
from .utils import SceneGraph, Layout, safe_prompt_slug


def generate_image(
    prompt: str,
    output_dir: str = "outputs",
    scene_graph: Optional[SceneGraph] = None,
    layout: Optional[Layout] = None,
    image_size: int = 512,
) -> str:
    """Run the full (dummy) pipeline on a prompt and save the resulting image."""
    if scene_graph is None:
        scene_graph = parse_text(prompt)

    if layout is None:
        layout = generate_layout(scene_graph)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = safe_prompt_slug(prompt)
    filename = f"{safe_prompt}_{timestamp}.png"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    image_path = render(
        prompt,
        layout,
        scene_graph=scene_graph,
        output_path=output_path,
        image_size=image_size,
    )
    return image_path


__all__ = ["generate_image"]
