"""Wrapper around image generation backends.

This module defines a simple interface for rendering an image given a
prompt and a layout.  The default implementation here produces a
placeholder visualisation using Pillow: coloured rectangles labelled with
object names.  In practice you would integrate a diffusion model
(e.g. Stable Diffusion) via the `diffusers` library or another backend.
"""
from __future__ import annotations

import os
from typing import Optional

from .utils import Layout, SceneGraph

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False


def render(prompt: str, layout: Layout, scene_graph: Optional[SceneGraph] = None,
           output_path: str = "output.png", image_size: int = 512) -> str:
    """Render an image given a text prompt and optional layout.

    This function currently generates a dummy visualisation by drawing
    coloured boxes for each object defined in the layout and annotating
    them with the object's noun.  It is primarily intended for
    debugging and demonstration purposes.

    Args:
        prompt: The original textual description (unused in this stub).
        layout: A :class:`Layout` mapping object ids to bounding boxes.
        scene_graph: The scene graph corresponding to the prompt.  If
            provided, object names will be drawn on the boxes.  If ``None``,
            object ids are used instead.
        output_path: File system path where the image will be saved.
        image_size: The width and height of the generated image in pixels.

    Returns:
        The path to the saved image file.

    Raises:
        ImportError: If Pillow is not installed and no real renderer is
            available.
    """
    if not _PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required for the dummy renderer. Install pillow or "
            "replace `render` with a call to a real diffusion backend."
        )

    # Create a blank canvas
    img = Image.new("RGB", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(img)

    # Attempt to load a default font.  If unavailable, fall back to the
    # basic built‑in font.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore

    # Generate distinct colours for up to 10 objects using an HSV palette.
    import colorsys
    num_objects = len(layout.boxes)
    for idx, (obj_id, bbox) in enumerate(layout.boxes.items()):
        hue = idx / max(1, num_objects)
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.5, 0.9)]
        x0 = int(bbox.x * image_size)
        y0 = int(bbox.y * image_size)
        x1 = int((bbox.x + bbox.width) * image_size)
        y1 = int((bbox.y + bbox.height) * image_size)
        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline="black")
        # Determine label
        label = obj_id
        if scene_graph is not None:
            obj = scene_graph.get_object(obj_id)
            if obj is not None:
                label = obj.noun
        # Draw the label at the top‑left corner of the box
        if font is not None:
            text_width, text_height = draw.textsize(label, font=font)
            text_x = x0 + 2
            text_y = y0 + 2
            draw.rectangle([text_x, text_y, text_x + text_width + 2, text_y + text_height + 2], fill="white")
            draw.text((text_x + 1, text_y + 1), label, fill="black", font=font)
        else:
            draw.text((x0 + 2, y0 + 2), label, fill="black")

    # Ensure the output directory exists
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    img.save(output_path)
    return output_path


__all__ = ["render"]