"""Semantic success evaluation using CLIP."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False


def compute_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute tight bbox from a boolean mask. Returns (x0, y0, x1, y1)."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("mask has no positive pixels")
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    pad_px: int,
) -> Tuple[int, int, int, int]:
    """Expand bbox by pad_px, clamp to image size."""
    x0, y0, x1, y1 = bbox
    width, height = image_size
    x0 = max(0, x0 - pad_px)
    y0 = max(0, y0 - pad_px)
    x1 = min(width, x1 + pad_px)
    y1 = min(height, y1 + pad_px)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid bbox after padding")
    return x0, y0, x1, y1


def _load_mask(path: str) -> np.ndarray:
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to load masks.")
    mask_img = Image.open(path).convert("L")
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return mask > 0.5


def _select_device(device: str) -> str:
    device = device.lower()
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


_CLIP_CACHE: Dict[Tuple[str, str], Tuple[object, object]] = {}


def _get_clip(model_id: str, device: str) -> Tuple[object, object]:
    key = (model_id, device)
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("transformers is required for CLIP semantic evaluation.") from exc

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required for CLIP semantic evaluation.") from exc

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    model.eval()

    _CLIP_CACHE[key] = (processor, model)
    return processor, model


def evaluate_semantic(
    after_path: str,
    mask_path: str,
    text: str,
    out_path: Optional[str] = None,
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    pad_px: int = 8,
) -> Dict[str, object]:
    """Compute CLIP similarity on crop defined by mask + padding."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to load images.")

    mask = _load_mask(mask_path)
    image = Image.open(after_path).convert("RGB")
    width, height = image.size

    bbox = compute_bbox_from_mask(mask)
    crop_box = expand_bbox(bbox, (width, height), pad_px)
    crop = image.crop(crop_box)

    device = _select_device(device)
    processor, model = _get_clip(model_id, device)

    import torch

    inputs = processor(text=[text], images=[crop], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        image_emb = outputs.image_embeds
        text_emb = outputs.text_embeds
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        similarity = (image_emb * text_emb).sum(dim=-1).item()

    result = {
        "clip_similarity": float(similarity),
        "crop_box": [int(crop_box[0]), int(crop_box[1]), int(crop_box[2]), int(crop_box[3])],
        "model_id": model_id,
        "device": device,
        "text": text,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


__all__ = ["compute_bbox_from_mask", "expand_bbox", "evaluate_semantic"]
