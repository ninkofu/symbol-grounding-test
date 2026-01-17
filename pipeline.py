"""High‑level orchestration of the symbol grounding pipeline.

This module glues together the individual components of the system.  The
main entry point is the :func:`generate_image` function, which takes a
text prompt, constructs a scene graph, generates a layout, and
invokes a renderer to produce the final image.  While the underlying
implementations are stubs, the control flow mirrors that of a complete
system.
"""
from __future__ import annotations

import os
import datetime as _dt
from typing import Optional

from .scene_graph import parse_text
from .layout import generate_layout
from .diffusion_interface import render
from .utils import SceneGraph, Layout


def generate_image(prompt: str, output_dir: str = "outputs", 
                   scene_graph: Optional[SceneGraph] = None,
                   layout: Optional[Layout] = None) -> str:
    """Run the full pipeline on a prompt and save the resulting image.

    Args:
        prompt: A natural language description of the desired image.
        output_dir: Directory where the output image will be saved.  The
            function will create the directory if it does not exist.
        scene_graph: Optional precomputed scene graph.  If ``None``, the
            scene graph will be derived from the prompt using
            :func:`symbol_grounding.scene_graph.parse_text`.
        layout: Optional precomputed layout.  If ``None``, the layout
            will be generated from the scene graph.

    Returns:
        Path to the saved image file.
    """
    # Derive scene graph if not provided
    if scene_graph is None:
        scene_graph = parse_text(prompt)

    # Derive layout if not provided
    if layout is None:
        layout = generate_layout(scene_graph)

    # Build a unique filename based on the prompt and timestamp
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "_".join(prompt.strip().split())[:20]
    filename = f"{safe_prompt}_{timestamp}.png"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Invoke the renderer.  Pass scene_graph to annotate boxes with nouns.
    image_path = render(prompt, layout, scene_graph=scene_graph, output_path=output_path)
    return image_path


__all__ = ["generate_image"]