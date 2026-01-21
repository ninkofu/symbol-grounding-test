"""System 2-style symbolic planning pipeline for grounded image generation."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .diffusion_interface import render
from .layout import generate_layout, save_layout_wireframe
from .scene_graph import parse_text
from .utils import BoundingBox, Layout, SceneGraph, SceneObject, safe_prompt_slug

ABSTRACT_SCENE_EXPANSIONS: Dict[str, List[Tuple[str, Dict[str, str]]]] = {
    "lonely": [
        ("window", {"state": "closed"}),
        ("flower", {"state": "withered"}),
        ("lamp", {"lighting": "dim"}),
        ("letter", {"arrangement": "scattered"}),
    ],
    "寂しい": [
        ("window", {"state": "closed"}),
        ("flower", {"state": "withered"}),
        ("lamp", {"lighting": "dim"}),
        ("letter", {"arrangement": "scattered"}),
    ],
    "gloomy": [
        ("lamp", {"lighting": "dark"}),
        ("room", {"state": "empty"}),
    ],
}


@dataclass
class ScenePlan:
    """Structured plan for a scene before layout."""

    prompt: str
    scene_graph: SceneGraph
    abstract_terms: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt,
            "abstract_terms": self.abstract_terms,
            "notes": self.notes,
            "scene_graph": _scene_graph_to_dict(self.scene_graph),
        }


@dataclass
class LayoutPlan:
    """Layout plan produced after spatial reasoning."""

    layout: Layout
    relation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "relation_notes": self.relation_notes,
            "layout": _layout_to_dict(self.layout),
        }


@dataclass
class AttentionPlan:
    """Token-to-object alignment and lock state for editing."""

    token_map: Dict[str, List[str]]
    locked_tokens: List[str] = field(default_factory=list)

    def lock(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            if token not in self.locked_tokens:
                self.locked_tokens.append(token)

    def unlock(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            if token in self.locked_tokens:
                self.locked_tokens.remove(token)

    def to_dict(self) -> Dict[str, object]:
        return {
            "token_map": self.token_map,
            "locked_tokens": self.locked_tokens,
        }


@dataclass
class DisentangledLatent:
    """Simple disentangled latent container (shape/color/position/etc.)."""

    shape: np.ndarray
    color: np.ndarray
    position: np.ndarray
    scale: np.ndarray
    style: np.ndarray

    def to_dict(self) -> Dict[str, list]:
        return {
            "shape": self.shape.tolist(),
            "color": self.color.tolist(),
            "position": self.position.tolist(),
            "scale": self.scale.tolist(),
            "style": self.style.tolist(),
        }


@dataclass
class System2Result:
    """Outputs from running the System 2 pipeline."""

    image_path: str
    plan_path: str
    layout_path: str


class System2Pipeline:
    """Pipeline that exposes symbolic planning, layout control and edits."""

    def plan_scene(self, prompt: str) -> ScenePlan:
        base_graph = parse_text(prompt)
        abstract_terms = _collect_abstract_terms(prompt)
        expanded_objects = _expand_abstract_terms(abstract_terms)

        scene_graph = _merge_scene_graph(base_graph, expanded_objects)
        notes = []
        if abstract_terms:
            notes.append("Expanded abstract descriptors into concrete objects.")
        return ScenePlan(prompt=prompt, scene_graph=scene_graph, abstract_terms=abstract_terms, notes=notes)

    def plan_layout(self, scene_graph: SceneGraph) -> LayoutPlan:
        layout = generate_layout(scene_graph)
        relation_notes = _apply_relations_to_layout(layout, scene_graph.relations)
        return LayoutPlan(layout=layout, relation_notes=relation_notes)

    def build_attention_plan(self, scene_graph: SceneGraph) -> AttentionPlan:
        token_map: Dict[str, List[str]] = {}
        for obj in scene_graph.objects:
            token_map.setdefault(obj.noun, []).append(obj.id)
            for attr_value in obj.attributes.values():
                token_map.setdefault(attr_value, []).append(obj.id)
        return AttentionPlan(token_map=token_map)

    def build_disentangled_latent(self, seed: int = 0) -> DisentangledLatent:
        rng = np.random.default_rng(seed)
        return DisentangledLatent(
            shape=rng.standard_normal(8),
            color=rng.standard_normal(8),
            position=rng.standard_normal(4),
            scale=rng.standard_normal(2),
            style=rng.standard_normal(8),
        )

    def apply_layout_edits(self, layout: Layout, edits: Sequence[Tuple[str, float, float]]) -> None:
        for obj_id, dx, dy in edits:
            if obj_id not in layout.boxes:
                raise ValueError(f"Unknown object id for layout edit: {obj_id}")
            box = layout.boxes[obj_id]
            box.x += dx
            box.y += dy
            box.clamp()

    def apply_attribute_edits(self, scene_graph: SceneGraph, edits: Sequence[Tuple[str, str, str]]) -> None:
        for obj_id, key, value in edits:
            obj = scene_graph.get_object(obj_id)
            if obj is None:
                raise ValueError(f"Unknown object id for attribute edit: {obj_id}")
            obj.attributes[key] = value

    def run(
        self,
        prompt: str,
        output_dir: str = "outputs",
        image_size: int = 512,
        layout_edits: Optional[Sequence[Tuple[str, float, float]]] = None,
        attribute_edits: Optional[Sequence[Tuple[str, str, str]]] = None,
        lock_tokens: Optional[Sequence[str]] = None,
    ) -> System2Result:
        plan = self.plan_scene(prompt)
        layout_plan = self.plan_layout(plan.scene_graph)

        if attribute_edits:
            self.apply_attribute_edits(plan.scene_graph, attribute_edits)

        if layout_edits:
            self.apply_layout_edits(layout_plan.layout, layout_edits)

        attention_plan = self.build_attention_plan(plan.scene_graph)
        if lock_tokens:
            attention_plan.lock(lock_tokens)

        latent = self.build_disentangled_latent()
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = safe_prompt_slug(prompt)
        os.makedirs(output_dir, exist_ok=True)

        plan_path = os.path.join(output_dir, f"plan_{safe_prompt}_{timestamp}.json")
        layout_path = os.path.join(output_dir, f"layout_{safe_prompt}_{timestamp}.png")
        image_path = os.path.join(output_dir, f"render_{safe_prompt}_{timestamp}.png")

        save_layout_wireframe(layout_plan.layout, output_path=layout_path, image_size=image_size)
        render(prompt, layout_plan.layout, scene_graph=plan.scene_graph, output_path=image_path, image_size=image_size)

        payload = {
            "scene_plan": plan.to_dict(),
            "layout_plan": layout_plan.to_dict(),
            "attention_plan": attention_plan.to_dict(),
            "latent": latent.to_dict(),
        }
        _write_json(plan_path, payload)

        return System2Result(image_path=image_path, plan_path=plan_path, layout_path=layout_path)


def _collect_abstract_terms(prompt: str) -> List[str]:
    lower = prompt.lower()
    terms = []
    for key in ABSTRACT_SCENE_EXPANSIONS:
        if key in lower:
            terms.append(key)
    return terms


def _expand_abstract_terms(abstract_terms: Iterable[str]) -> List[SceneObject]:
    expanded: List[SceneObject] = []
    obj_counter = 1
    for term in abstract_terms:
        for noun, attrs in ABSTRACT_SCENE_EXPANSIONS.get(term, []):
            obj_id = f"abs{obj_counter}"
            obj_counter += 1
            expanded.append(SceneObject(id=obj_id, noun=noun, attributes=attrs.copy()))
    return expanded


def _merge_scene_graph(base_graph: SceneGraph, extra_objects: List[SceneObject]) -> SceneGraph:
    objects = list(base_graph.objects)
    relations = list(base_graph.relations)
    if extra_objects:
        existing_ids = {obj.id for obj in objects}
        next_idx = len(objects) + 1
        for extra in extra_objects:
            if extra.id in existing_ids:
                extra = dataclasses.replace(extra, id=f"obj{next_idx}")
                next_idx += 1
            objects.append(extra)
    return SceneGraph(objects=objects, relations=relations)


def _apply_relations_to_layout(layout: Layout, relations: Sequence[Tuple[str, str, str]]) -> List[str]:
    notes = []
    for subject_id, predicate, object_id in relations:
        if subject_id not in layout.boxes or object_id not in layout.boxes:
            continue
        subject_box = layout.boxes[subject_id]
        object_box = layout.boxes[object_id]
        if predicate == "on":
            subject_box.y = max(0.0, object_box.y - subject_box.height * 0.8)
            notes.append(f"{subject_id} placed on {object_id}")
        elif predicate == "under":
            subject_box.y = min(1.0 - subject_box.height, object_box.y + object_box.height + 0.02)
            notes.append(f"{subject_id} placed under {object_id}")
        elif predicate in {"beside", "next to"}:
            subject_box.x = min(1.0 - subject_box.width, object_box.x + object_box.width + 0.02)
            notes.append(f"{subject_id} placed beside {object_id}")
        elif predicate == "behind":
            notes.append(f"{subject_id} placed behind {object_id} (depth only)")
        elif predicate == "in front of":
            notes.append(f"{subject_id} placed in front of {object_id} (depth only)")

    layout.clamp_boxes()
    return notes


def _scene_graph_to_dict(scene_graph: SceneGraph) -> Dict[str, object]:
    return {
        "objects": [
            {"id": obj.id, "noun": obj.noun, "attributes": obj.attributes}
            for obj in scene_graph.objects
        ],
        "relations": [
            {"subject": sub, "predicate": pred, "object": obj}
            for sub, pred, obj in scene_graph.relations
        ],
    }


def _layout_to_dict(layout: Layout) -> Dict[str, Dict[str, float]]:
    return {
        obj_id: {
            "x": bbox.x,
            "y": bbox.y,
            "width": bbox.width,
            "height": bbox.height,
        }
        for obj_id, bbox in layout.boxes.items()
    }


def _write_json(path: str, payload: Dict[str, object]) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = ["System2Pipeline", "System2Result", "ScenePlan", "LayoutPlan", "AttentionPlan", "DisentangledLatent"]
