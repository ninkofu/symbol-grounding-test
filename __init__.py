"""Top level package for the Symbol Grounding test project.

Importing this package will expose a high‑level convenience function
`generate_image` that ties together the different submodules.  See
`pipeline.generate_image` for details.
"""

from .pipeline import generate_image

__all__ = ["generate_image"]