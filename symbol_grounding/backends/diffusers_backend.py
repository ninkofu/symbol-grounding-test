"""Compatibility wrapper for symbol_grounding.diffusion.diffusers_backend."""

from symbol_grounding.diffusion.diffusers_backend import DiffusionConfig as DiffusersConfig
from symbol_grounding.diffusion.diffusers_backend import generate_image

__all__ = ["DiffusersConfig", "generate_image"]
