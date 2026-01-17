"""Spatial layout generation from scene graphs."""
from __future__ import annotations

import os
from typing import Dict, Tuple, Union

from .utils import SceneGraph, Layout, BoundingBox

try:
    from PIL import Image, ImageDraw  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False


def generate_layout(scene_graph: SceneGraph) -> Layout:
    """Produce a simple grid layout for the objects in the scene."""
    num_objects = len(scene_graph.objects)
    if num_objects == 0:
        return Layout(boxes={})

    max_cols = 3
    cols = min(num_objects, max_cols)
    rows = (num_objects + max_cols - 1) // max_cols

    box_width = 1.0 / cols
    box_height = 1.0 / rows

    boxes: Dict[str, BoundingBox] = {}
    for idx, obj in enumerate(scene_graph.objects):
        row = idx // max_cols
        col = idx % max_cols
        x = col * box_width
        y = row * box_height
        boxes[obj.id] = BoundingBox(x=x, y=y, width=box_width, height=box_height)

    layout = Layout(boxes=boxes)
    layout.clamp_boxes()
    return layout


def render_layout_wireframe(
    layout: Layout,
    image_size: Union[int, Tuple[int, int]] = 512,
    line_width: int = 3,
) -> "Image.Image":
    """Render a white background with black box outlines for the layout."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to render layout wireframes.")

    if isinstance(image_size, tuple):
        width, height = image_size
    else:
        width = height = image_size

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    for bbox in layout.boxes.values():
        x0 = int(bbox.x * width)
        y0 = int(bbox.y * height)
        x1 = int((bbox.x + bbox.width) * width)
        y1 = int((bbox.y + bbox.height) * height)
        draw.rectangle([x0, y0, x1, y1], outline="black", width=line_width)
    return img


def layout_to_mask(
    layout: Layout,
    target_id: Union[str, list[str]],
    image_size: Union[int, Tuple[int, int]] = 512,
) -> "Image.Image":
    """Create a binary mask for target object ids (white = edit region)."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to render layout masks.")

    if isinstance(image_size, tuple):
        width, height = image_size
    else:
        width = height = image_size

    if isinstance(target_id, str):
        target_ids = {target_id}
    else:
        target_ids = set(target_id)

    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    found = False
    for obj_id, bbox in layout.boxes.items():
        if obj_id not in target_ids:
            continue
        found = True
        x0 = int(bbox.x * width)
        y0 = int(bbox.y * height)
        x1 = int((bbox.x + bbox.width) * width)
        y1 = int((bbox.y + bbox.height) * height)
        draw.rectangle([x0, y0, x1, y1], fill=255)

    if not found:
        raise ValueError(f"Target object id(s) not found in layout: {sorted(target_ids)}")

    return img


def save_layout_mask(
    layout: Layout,
    target_id: Union[str, list[str]],
    output_path: str = "outputs/mask.png",
    image_size: Union[int, Tuple[int, int]] = 512,
) -> str:
    """Save a binary mask for target object ids to disk."""
    img = layout_to_mask(layout, target_id=target_id, image_size=image_size)
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    img.save(output_path)
    return output_path


def save_layout_wireframe(
    layout: Layout,
    output_path: str = "outputs/control_layout.png",
    image_size: Union[int, Tuple[int, int]] = 512,
    line_width: int = 3,
) -> str:
    """Save a wireframe visualization of the layout to disk."""
    img = render_layout_wireframe(layout, image_size=image_size, line_width=line_width)
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    img.save(output_path)
    return output_path


__all__ = [
    "generate_layout",
    "render_layout_wireframe",
    "save_layout_wireframe",
    "layout_to_mask",
    "save_layout_mask",
]
