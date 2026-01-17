"""Compatibility wrapper for symbol_grounding.utils."""

from symbol_grounding.utils import (
    BoundingBox,
    SceneObject,
    SceneGraph,
    Layout,
    safe_prompt_slug,
)

__all__ = [
    "BoundingBox",
    "SceneObject",
    "SceneGraph",
    "Layout",
    "safe_prompt_slug",
]
