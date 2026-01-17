"""Conditioning image generation from layouts."""
from __future__ import annotations

from typing import Optional, Union, Tuple

from ..utils import Layout, SceneGraph

try:
    from PIL import Image, ImageDraw  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False


def layout_to_control_image(
    layout: Layout,
    scene_graph: Optional[SceneGraph] = None,
    image_size: Union[int, Tuple[int, int]] = 512,
    mode: str = "boxes",
    line_width: int = 4,
) -> "Image.Image":
    """Create a control image from a layout.

    Modes:
    - "boxes" / "scribble": white box outlines on black background.
    - "mask" / "seg": filled boxes with distinct colors.
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to build conditioning images.")

    mode = mode.lower()
    if isinstance(image_size, tuple):
        width, height = image_size
    else:
        width = height = image_size

    if mode in {"boxes", "scribble"}:
        img = Image.new("RGB", (width, height), color="black")
        draw = ImageDraw.Draw(img)
        for bbox in layout.boxes.values():
            x0 = int(bbox.x * width)
            y0 = int(bbox.y * height)
            x1 = int((bbox.x + bbox.width) * width)
            y1 = int((bbox.y + bbox.height) * height)
            draw.rectangle([x0, y0, x1, y1], outline="white", width=line_width)
        return img

    if mode in {"mask", "seg"}:
        img = Image.new("RGB", (width, height), color="black")
        draw = ImageDraw.Draw(img)
        import colorsys

        num_objects = max(1, len(layout.boxes))
        for idx, bbox in enumerate(layout.boxes.values()):
            hue = idx / num_objects
            r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.6, 0.9)]
            x0 = int(bbox.x * width)
            y0 = int(bbox.y * height)
            x1 = int((bbox.x + bbox.width) * width)
            y1 = int((bbox.y + bbox.height) * height)
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b))
        return img

    raise ValueError(f"Unsupported control image mode: {mode}")


__all__ = ["layout_to_control_image"]
