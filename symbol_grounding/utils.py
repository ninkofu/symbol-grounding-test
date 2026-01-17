"""Utility types and helper functions for the symbol grounding project.

This module defines lightweight data classes used to share structured
information between the different stages of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class BoundingBox:
    """Axis-aligned bounding box normalized to the unit square."""

    x: float
    y: float
    width: float
    height: float

    def clamp(self) -> None:
        """Clamp the bounding box values to [0, 1] in place."""
        self.x = min(max(self.x, 0.0), 1.0)
        self.y = min(max(self.y, 0.0), 1.0)
        self.width = min(max(self.width, 0.0), 1.0)
        self.height = min(max(self.height, 0.0), 1.0)


@dataclass
class SceneObject:
    """Representation of an object extracted from text."""

    id: str
    noun: str
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class SceneGraph:
    """Structured representation of a textual prompt."""

    objects: List[SceneObject]
    relations: List[Tuple[str, str, str]] = field(default_factory=list)

    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Retrieve an object by its identifier."""
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None


@dataclass
class Layout:
    """Mapping from object identifiers to spatial locations."""

    boxes: Dict[str, BoundingBox]

    def clamp_boxes(self) -> None:
        """Clamp all bounding boxes to the unit square in place."""
        for box in self.boxes.values():
            box.clamp()


def safe_prompt_slug(prompt: str, max_length: int = 24) -> str:
    """Create a filesystem-friendly slug from a prompt."""
    slug = "_".join(prompt.strip().split())
    if not slug:
        slug = "prompt"
    return slug[:max_length]


__all__ = [
    "BoundingBox",
    "SceneObject",
    "SceneGraph",
    "Layout",
    "safe_prompt_slug",
]
