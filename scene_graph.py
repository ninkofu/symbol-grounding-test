"""Scene graph extraction from natural language.

This module contains a basic parser that converts a free‑form text prompt
into a :class:`symbol_grounding.utils.SceneGraph`.  The current
implementation is intentionally simple: it uses hard‑coded lists of
common nouns and adjectives to identify objects and attributes.  In a
production system you might replace this with a call to a large
language model (LLM) or a grammar parser to obtain more accurate
results.
"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple

from .utils import SceneGraph, SceneObject

# Lists of known nouns and adjectives for the toy parser.  Extend these
# lists as needed for your application.  Note that this parser is case
# insensitive.
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
    """Convert a natural language prompt into a scene graph.

    The parser splits the prompt into tokens, identifies objects and
    their attributes, and extracts simple binary relations such as
    ``"cat on table"``.  Objects are assigned unique identifiers
    (``obj1``, ``obj2``, …) in the order in which they appear.

    Args:
        prompt: A free‑form text description of a scene.

    Returns:
        A :class:`SceneGraph` containing objects and relations.
    """
    text = prompt.lower()

    # Very naive tokenisation: split on whitespace and punctuation.
    tokens = re.findall(r"\b\w+\b", text)

    objects: List[SceneObject] = []
    relations: List[Tuple[str, str, str]] = []

    current_attrs: Dict[str, str] = {}
    obj_counter = 1

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Check for adjectives preceding a noun
        if token in COLOR_ADJECTIVES:
            current_attrs["color"] = token
            i += 1
            continue
        if token in SIZE_ADJECTIVES:
            current_attrs["size"] = token
            i += 1
            continue

        # Identify nouns
        if token in KNOWN_NOUNS:
            obj_id = f"obj{obj_counter}"
            obj_counter += 1
            objects.append(SceneObject(id=obj_id, noun=token, attributes=current_attrs.copy()))
            current_attrs.clear()
            i += 1
            continue

        # Identify relations: lookahead for patterns like "noun predicate noun"
        if token in RELATION_PREDICATES:
            # We need at least one object before and one after
            if objects and i + 1 < len(tokens):
                subject_id = objects[-1].id
                predicate = token
                # Find the next noun after the predicate
                for j in range(i + 1, len(tokens)):
                    if tokens[j] in KNOWN_NOUNS:
                        # Assign id lazily if object hasn't been created yet
                        # but still record the relation; actual SceneObject will
                        # be created when we encounter the noun token later.
                        # Use a provisional id like 'unknown_n' based on noun
                        # string and index to refer to it.
                        provisional_id = f"unknown_{tokens[j]}_{j}"
                        relations.append((subject_id, predicate, provisional_id))
                        break
            i += 1
            continue

        i += 1

    # Post‑process relations: replace provisional ids with actual object ids when possible
    final_relations: List[Tuple[str, str, str]] = []
    for sub, pred, obj in relations:
        # If the object id starts with 'unknown', try to find matching object by noun
        if obj.startswith("unknown"):
            # Extract noun and index from provisional id
            _, noun, idx = obj.split("_", 2)
            # Find the first object with the matching noun that appears after the index
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