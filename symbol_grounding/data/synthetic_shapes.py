"""Synthetic shapes dataset for slot attention experiments."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image, ImageDraw  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange"]
COLORS = [
    (220, 50, 47),
    (133, 153, 0),
    (38, 139, 210),
    (181, 137, 0),
    (211, 54, 130),
    (42, 161, 152),
    (203, 75, 22),
]

SHAPE_NAMES = ["circle", "square", "triangle"]


@dataclass
class SyntheticShapesConfig:
    length: int = 10000
    image_size: int = 64
    min_objects: int = 2
    max_objects: int = 4
    seed: int = 0


class SyntheticShapesDataset(Dataset):
    """Dataset that generates simple colored shapes on the fly."""

    def __init__(self, config: SyntheticShapesConfig):
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow is required to generate synthetic shapes.")
        self.config = config

    def __len__(self) -> int:
        return self.config.length

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rng = random.Random(self.config.seed + idx)
        image, labels = _render_sample(
            rng=rng,
            image_size=self.config.image_size,
            min_objects=self.config.min_objects,
            max_objects=self.config.max_objects,
        )
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return {"image": image_tensor, "labels": labels}


def _render_sample(
    rng: random.Random,
    image_size: int,
    min_objects: int,
    max_objects: int,
) -> Tuple["Image.Image", List[Dict[str, object]]]:
    img = Image.new("RGB", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(img)

    num_objects = rng.randint(min_objects, max_objects)
    labels: List[Dict[str, object]] = []

    for _ in range(num_objects):
        shape_idx = rng.randrange(len(SHAPE_NAMES))
        color_idx = rng.randrange(len(COLORS))
        color = COLORS[color_idx]

        size = rng.randint(image_size // 8, image_size // 3)
        x0 = rng.randint(0, image_size - size - 1)
        y0 = rng.randint(0, image_size - size - 1)
        x1 = x0 + size
        y1 = y0 + size

        if SHAPE_NAMES[shape_idx] == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=color)
        elif SHAPE_NAMES[shape_idx] == "square":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            # triangle
            points = [(x0 + size // 2, y0), (x0, y1), (x1, y1)]
            draw.polygon(points, fill=color)

        labels.append(
            {
                "shape_idx": shape_idx,
                "color_idx": color_idx,
                "bbox": (x0 / image_size, y0 / image_size, x1 / image_size, y1 / image_size),
            }
        )

    return img, labels


__all__ = ["SyntheticShapesDataset", "SyntheticShapesConfig", "SHAPE_NAMES", "COLOR_NAMES"]
