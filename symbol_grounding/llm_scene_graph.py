"""LLM-backed scene graph planner with rule-based fallback."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib import request

from .scene_graph import parse_text
from .utils import SceneGraph, SceneObject


@dataclass
class LLMPlannerConfig:
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: int = 30


class LLMSceneGraphPlanner:
    """Plan a scene graph from text using an LLM API."""

    def __init__(self, config: Optional[LLMPlannerConfig] = None):
        self.config = config or LLMPlannerConfig()

    def plan(self, prompt: str) -> SceneGraph:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            return parse_text(prompt)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Convert the user prompt into a JSON scene graph with keys "
                        "`objects` (list of {id,noun,attributes}) and `relations` "
                        "(list of {subject,predicate,object}). Return ONLY JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.api_base,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        response = json.loads(raw)
        content = response["choices"][0]["message"]["content"]
        return _scene_graph_from_json(content, fallback_prompt=prompt)


def _scene_graph_from_json(content: str, fallback_prompt: str) -> SceneGraph:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return parse_text(fallback_prompt)
    objects: List[SceneObject] = []
    relations: List[tuple[str, str, str]] = []
    for obj in parsed.get("objects", []):
        obj_id = obj.get("id") or f"obj{len(objects) + 1}"
        noun = obj.get("noun", "object")
        attributes = obj.get("attributes") or {}
        objects.append(SceneObject(id=obj_id, noun=noun, attributes=attributes))
    for rel in parsed.get("relations", []):
        relations.append(
            (rel.get("subject", ""), rel.get("predicate", ""), rel.get("object", ""))
        )
    if not objects:
        return parse_text(fallback_prompt)
    return SceneGraph(objects=objects, relations=relations)


__all__ = ["LLMSceneGraphPlanner", "LLMPlannerConfig"]
