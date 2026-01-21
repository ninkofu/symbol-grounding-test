"""Simple image folder dataset for real-world disentanglement experiments."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class ImageFolderConfig:
    root: str
    image_size: int = 64
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")


class ImageFolderDataset(Dataset):
    """Load images from a folder and return normalized tensors."""

    def __init__(self, config: ImageFolderConfig):
        self.config = config
        self.paths = [
            os.path.join(config.root, fname)
            for fname in sorted(os.listdir(config.root))
            if fname.lower().endswith(config.extensions)
        ]
        if not self.paths:
            raise ValueError(f"No images found in {config.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = image.resize((self.config.image_size, self.config.image_size))
        array = np.array(image, dtype="uint8")
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        return {"image": tensor, "labels": None, "path": path}


__all__ = ["ImageFolderConfig", "ImageFolderDataset"]
