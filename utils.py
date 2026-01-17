"""Utility types and helper functions for the symbol grounding project.

This module defines a handful of lightweight data classes used to share
structured information between the different stages of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class BoundingBox:
    """Axis‑aligned bounding box normalised to the unit square.

    The coordinates ``x`` and ``y`` denote the top‑left corner, while
    ``width`` and ``height`` specify the box size.  All values are
    floats in the range [0, 1], where (0,0) corresponds to the top left of the
    canvas and (1,1) to the bottom right.
    """

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
    """Representation of an object extracted from text.

    Attributes:
        id: Unique identifier for the object within the scene graph.
        noun: Core identity of the object (e.g. ``"cat"``).
        attributes: Dictionary mapping attribute types (e.g. ``"color"``) to
            attribute values (e.g. ``"red"``).
    """

    id: str
    noun: str
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class SceneGraph:
    """Structured representation of a textual prompt.

    The scene graph contains a list of objects and a list of binary relations.
    Each relation is a tuple of (subject_id, predicate, object_id).
    """

    objects: List[SceneObject]
    relations: List[Tuple[str, str, str]] = field(default_factory=list)

    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Retrieve an object by its identifier.

        Args:
            object_id: Identifier of the object to retrieve.

        Returns:
            The corresponding :class:`SceneObject` if found, otherwise ``None``.
        """
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None


@dataclass
class Layout:
    """Mapping from object identifiers to spatial locations.

    A layout is produced by the layout generator and describes where
    each object should appear in the image.  The mapping keys must
    correspond to identifiers defined in a :class:`SceneGraph`.
    """

    boxes: Dict[str, BoundingBox]

    def clamp_boxes(self) -> None:
        """Clamp all bounding boxes to the unit square in place."""
        for box in self.boxes.values():
            box.clamp()