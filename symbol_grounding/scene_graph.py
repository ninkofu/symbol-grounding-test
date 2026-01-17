"""Scene graph extraction from natural language."""
from __future__ import annotations

import re
from typing import List, Dict, Tuple

from .utils import SceneGraph, SceneObject

KNOWN_NOUNS = {
    "cat", "dog", "ball", "table", "chair", "flower", "tree", "bird",
    "car", "bicycle", "cup", "person", "human", "girl", "boy", "book",
}

COLOR_ADJECTIVES = {
    "red", "blue", "green", "yellow", "purple", "white", "black", "brown",
    "orange", "pink", "grey", "gray", "cyan",
}

SIZE_ADJECTIVES = {"big", "small", "large", "tiny", "huge"}

RELATION_PREDICATES = {"on", "under", "beside", "next to", "behind", "in front of"}


def parse_text(prompt: str) -> SceneGraph:
    """Convert a natural language prompt into a scene graph."""
    text = prompt.lower()
    tokens = re.findall(r"\b\w+\b", text)

    objects: List[SceneObject] = []
    relations: List[Tuple[str, str, str]] = []

    current_attrs: Dict[str, str] = {}
    obj_counter = 1

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in COLOR_ADJECTIVES:
            current_attrs["color"] = token
            i += 1
            continue
        if token in SIZE_ADJECTIVES:
            current_attrs["size"] = token
            i += 1
            continue

        if token in KNOWN_NOUNS:
            obj_id = f"obj{obj_counter}"
            obj_counter += 1
            objects.append(SceneObject(id=obj_id, noun=token, attributes=current_attrs.copy()))
            current_attrs.clear()
            i += 1
            continue

        if token in RELATION_PREDICATES:
            if objects and i + 1 < len(tokens):
                subject_id = objects[-1].id
                predicate = token
                for j in range(i + 1, len(tokens)):
                    if tokens[j] in KNOWN_NOUNS:
                        provisional_id = f"unknown_{tokens[j]}_{j}"
                        relations.append((subject_id, predicate, provisional_id))
                        break
            i += 1
            continue

        i += 1

    final_relations: List[Tuple[str, str, str]] = []
    for sub, pred, obj in relations:
        if obj.startswith("unknown"):
            _, noun, _ = obj.split("_", 2)
            matched_id = None
            for scene_obj in objects:
                if scene_obj.noun == noun:
                    matched_id = scene_obj.id
                    break
            if matched_id is not None:
                final_relations.append((sub, pred, matched_id))
        else:
            final_relations.append((sub, pred, obj))

    return SceneGraph(objects=objects, relations=final_relations)


__all__ = ["parse_text"]
