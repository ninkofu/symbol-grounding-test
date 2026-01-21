"""Top level package for the Symbol Grounding project."""

from .pipeline import generate_image
from .system2_pipeline import System2Pipeline, System2Result

__all__ = ["generate_image", "System2Pipeline", "System2Result"]
