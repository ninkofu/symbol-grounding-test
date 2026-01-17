"""Compatibility wrapper for symbol_grounding.layout."""

from symbol_grounding.layout import (
    generate_layout,
    render_layout_wireframe,
    save_layout_wireframe,
    layout_to_mask,
    save_layout_mask,
)

__all__ = [
    "generate_layout",
    "render_layout_wireframe",
    "save_layout_wireframe",
    "layout_to_mask",
    "save_layout_mask",
]
