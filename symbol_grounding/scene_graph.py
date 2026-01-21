"""Scene graph extraction from natural language."""
from __future__ import annotations

import re
from typing import List, Dict, Tuple

from .utils import SceneGraph, SceneObject

KNOWN_NOUNS = {
    "cat", "dog", "ball", "table", "chair", "flower", "tree", "bird",
    "car", "bicycle", "cup", "mug", "apple", "laptop",
    "person", "human", "girl", "boy", "book",
    "plate", "road", "park", "wall", "room", "desk", "vase",
    "lamp", "window", "letter", "envelope", "bed", "sofa", "rug",
}

COLOR_ADJECTIVES = {
    "red", "blue", "green", "yellow", "purple", "white", "black", "brown",
    "orange", "pink", "grey", "gray", "cyan",
}

SIZE_ADJECTIVES = {"big", "small", "large", "tiny", "huge"}

ATTRIBUTE_ADJECTIVES = {
    "withered": ("state", "withered"),
    "dim": ("lighting", "dim"),
    "dark": ("lighting", "dark"),
    "scattered": ("arrangement", "scattered"),
    "lonely": ("mood", "lonely"),
    "empty": ("state", "empty"),
}

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
        bigram = " ".join(tokens[i:i + 2]) if i + 1 < len(tokens) else None
        trigram = " ".join(tokens[i:i + 3]) if i + 2 < len(tokens) else None

        if token in COLOR_ADJECTIVES:
            current_attrs["color"] = token
            i += 1
            continue
        if token in SIZE_ADJECTIVES:
            current_attrs["size"] = token
            i += 1
            continue
        if token in ATTRIBUTE_ADJECTIVES:
            key, value = ATTRIBUTE_ADJECTIVES[token]
            current_attrs[key] = value
            i += 1
            continue

        if token in KNOWN_NOUNS:
            obj_id = f"obj{obj_counter}"
            obj_counter += 1
            objects.append(SceneObject(id=obj_id, noun=token, attributes=current_attrs.copy()))
            current_attrs.clear()
            i += 1
            continue

        predicate = None
        if trigram and trigram in RELATION_PREDICATES:
            predicate = trigram
            i += 3
        elif bigram and bigram in RELATION_PREDICATES:
            predicate = bigram
            i += 2
        elif token in RELATION_PREDICATES:
            predicate = token
            i += 1
        if predicate is not None:
            if objects and i + 1 < len(tokens):
                subject_id = objects[-1].id
                for j in range(i + 1, len(tokens)):
                    if tokens[j] in KNOWN_NOUNS:
                        provisional_id = f"unknown_{tokens[j]}_{j}"
                        relations.append((subject_id, predicate, provisional_id))
                        break
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
