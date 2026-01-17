"""Spatial layout generation from scene graphs.

The layout module translates a :class:`symbol_grounding.utils.SceneGraph` into
a mapping from object identifiers to bounding boxes.  In a real
implementation this might involve learning a neural network that predicts
object positions from textual descriptions or employing a layout model
like GLIGEN.  Here we use a deterministic heuristic: objects are
arranged in a grid with equal spacing.
"""
from __future__ import annotations

from typing import Dict

from .utils import SceneGraph, Layout, BoundingBox


def generate_layout(scene_graph: SceneGraph) -> Layout:
    """Produce a simple grid layout for the objects in the scene.

    The algorithm places objects left to right in rows of at most three
    elements.  Bounding boxes are assigned equal width and height and
    do not overlap.  For more sophisticated arrangements, replace this
    function with a learned model.

    Args:
        scene_graph: Parsed scene description.

    Returns:
        A :class:`Layout` with a bounding box for each object identifier.
    """
    num_objects = len(scene_graph.objects)
    if num_objects == 0:
        return Layout(boxes={})

    # Determine grid dimensions: at most 3 columns per row
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


__all__ = ["generate_layout"]