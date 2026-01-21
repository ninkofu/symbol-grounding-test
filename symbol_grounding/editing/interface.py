"""Editing interface definitions for future swap-in implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class EditRequest:
    """Input payload for image editing."""

    image_path: str
    mask_path: Optional[str]
    mask_image: Optional["object"] = None
    edit_prompt: str = ""
    base_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    strength: float = 0.75
    seed: int = 0
    steps: int = 30
    guidance_scale: float = 7.5
    height: Optional[int] = None
    width: Optional[int] = None
    model_id: str = "runwayml/stable-diffusion-inpainting"
    device: str = "auto"
    cache_dir: Optional[str] = None
    local_files_only: bool = False


@dataclass
class EditResult:
    """Output payload for image editing."""

    image: "object"
    seed: int


class Editor(Protocol):
    """Protocol for editing backends (e.g., inpainting, P2P)."""

    def edit(self, request: EditRequest) -> EditResult:
        """Run the edit and return the resulting image."""
        raise NotImplementedError


__all__ = ["EditRequest", "EditResult", "Editor"]
