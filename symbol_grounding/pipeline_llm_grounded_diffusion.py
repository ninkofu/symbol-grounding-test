"""LLM-driven grounded diffusion pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .diffusion import DiffusionConfig
from .layout import generate_layout
from .llm_scene_graph import LLMPlannerConfig, LLMSceneGraphPlanner
from .pipeline_grounded_diffusion import generate_grounded_image
from .utils import SceneGraph, Layout


@dataclass
class LLMGroundedResult:
    image_path: str
    control_path: str
    seed: int
    scene_graph: SceneGraph
    layout: Layout


def generate_llm_grounded_image(
    prompt: str,
    output_dir: str,
    config: DiffusionConfig,
    llm_config: Optional[LLMPlannerConfig] = None,
    control_mode: str = "scribble",
    base_prompt: Optional[str] = None,
    lock_tokens: Optional[list[str]] = None,
) -> LLMGroundedResult:
    planner = LLMSceneGraphPlanner(llm_config)
    scene_graph = planner.plan(prompt)
    layout = generate_layout(scene_graph)
    result = generate_grounded_image(
        prompt=prompt,
        output_dir=output_dir,
        config=config,
        control_mode=control_mode,
        scene_graph=scene_graph,
        layout=layout,
        base_prompt=base_prompt,
        lock_tokens=lock_tokens,
    )
    return LLMGroundedResult(
        image_path=result.image_path,
        control_path=result.control_path,
        seed=result.seed,
        scene_graph=result.scene_graph,
        layout=result.layout,
    )


__all__ = ["generate_llm_grounded_image", "LLMGroundedResult"]
