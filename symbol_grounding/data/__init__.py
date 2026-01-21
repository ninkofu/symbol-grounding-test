"""Datasets for symbol grounding experiments."""

from .synthetic_shapes import SyntheticShapesDataset, SHAPE_NAMES, COLOR_NAMES
from .captioned_shapes import CaptionedShapesDataset, CaptionedShapesConfig
from .image_folder import ImageFolderDataset, ImageFolderConfig

__all__ = [
    "SyntheticShapesDataset",
    "SHAPE_NAMES",
    "COLOR_NAMES",
    "CaptionedShapesDataset",
    "CaptionedShapesConfig",
    "ImageFolderDataset",
    "ImageFolderConfig",
]
