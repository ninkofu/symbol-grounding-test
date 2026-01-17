"""Locality evaluation utilities."""
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


def _load_image_rgb(path: str) -> np.ndarray:
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to load images.")
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _load_mask(path: str) -> np.ndarray:
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to load masks.")
    mask_img = Image.open(path).convert("L")
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return mask > 0.5


def compute_locality_metrics(
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray,
    threshold: float = 10 / 255,
) -> Dict[str, float | Tuple[int, int]]:
    """Compute locality metrics given arrays in [0,1]."""
    if before.shape != after.shape:
        raise ValueError("before/after shapes must match.")
    if before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("before/after must be RGB images.")

    if mask.shape[:2] != before.shape[:2]:
        raise ValueError("mask size must match image size.")

    mask_bool = mask
    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool > 0.5

    diff = np.mean(np.abs(after - before), axis=2)
    diff_sq = np.mean((after - before) ** 2, axis=2)

    outside = ~mask_bool
    inside = mask_bool

    outside_mean_abs = float(diff[outside].mean()) if outside.any() else 0.0
    outside_mean_sq = float(diff_sq[outside].mean()) if outside.any() else 0.0
    outside_frac = float((diff[outside] > threshold).mean()) if outside.any() else 0.0

    inside_mean_abs = float(diff[inside].mean()) if inside.any() else 0.0
    inside_frac = float((diff[inside] > threshold).mean()) if inside.any() else 0.0

    return {
        "outside_mean_abs_diff": outside_mean_abs,
        "outside_mean_sq_diff": outside_mean_sq,
        "outside_frac_changed": outside_frac,
        "inside_mean_abs_diff": inside_mean_abs,
        "inside_frac_changed": inside_frac,
        "image_size": (before.shape[1], before.shape[0]),
        "threshold": float(threshold),
    }


def _save_grayscale(path: str, img: np.ndarray) -> None:
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to save images.")
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def _compute_boundary(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    up = np.roll(mask_bool, 1, axis=0)
    down = np.roll(mask_bool, -1, axis=0)
    left = np.roll(mask_bool, 1, axis=1)
    right = np.roll(mask_bool, -1, axis=1)
    boundary = mask_bool & (~up | ~down | ~left | ~right)
    return boundary


def _save_overlay(path: str, before: np.ndarray, mask: np.ndarray) -> None:
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required to save images.")
    img = (np.clip(before, 0, 1) * 255.0).astype(np.uint8)
    boundary = _compute_boundary(mask)
    img[boundary] = np.array([255, 0, 0], dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def evaluate_locality(
    before_path: str,
    after_path: str,
    mask_path: str,
    threshold: float = 10 / 255,
    out_path: Optional[str] = None,
    save_diff: Optional[str] = None,
    save_heatmap: Optional[str] = None,
    save_overlay: Optional[str] = None,
) -> Dict[str, float | Tuple[int, int]]:
    """Compute locality metrics from image paths and optionally save outputs."""
    before = _load_image_rgb(before_path)
    after = _load_image_rgb(after_path)
    mask = _load_mask(mask_path)

    metrics = compute_locality_metrics(before, after, mask, threshold=threshold)

    if save_diff:
        diff = np.mean(np.abs(after - before), axis=2)
        _save_grayscale(save_diff, diff)

    if save_heatmap:
        diff = np.mean(np.abs(after - before), axis=2)
        _save_grayscale(save_heatmap, diff)

    if save_overlay:
        _save_overlay(save_overlay, before, mask)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics


__all__ = ["compute_locality_metrics", "evaluate_locality"]
