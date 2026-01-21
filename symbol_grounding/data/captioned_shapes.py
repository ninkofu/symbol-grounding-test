"""Synthetic shapes dataset with auto-generated captions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .synthetic_shapes import SyntheticShapesConfig, SyntheticShapesDataset, SHAPE_NAMES, COLOR_NAMES


@dataclass
class CaptionedShapesConfig(SyntheticShapesConfig):
    max_objects_in_caption: int = 2


class CaptionedShapesDataset:
    """Wrap SyntheticShapesDataset and emit a simple caption."""

    def __init__(self, config: CaptionedShapesConfig):
        self.config = config
        self.base = SyntheticShapesDataset(config)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.base[idx]
        labels: List[Dict[str, object]] = sample["labels"]  # type: ignore[assignment]
        caption = _labels_to_caption(labels, max_objects=self.config.max_objects_in_caption)
        return {"image": sample["image"], "labels": labels, "caption": caption}


def _labels_to_caption(labels: List[Dict[str, object]], max_objects: int) -> str:
    chunks = []
    for label in labels[:max_objects]:
        shape_idx = int(label["shape_idx"])
        color_idx = int(label["color_idx"])
        chunks.append(f"a {COLOR_NAMES[color_idx]} {SHAPE_NAMES[shape_idx]}")
    if not chunks:
        return "a shape"
    if len(chunks) == 1:
        return chunks[0]
    return " and ".join(chunks)


__all__ = ["CaptionedShapesDataset", "CaptionedShapesConfig"]
