"""Evaluation utilities for symbol grounding experiments."""

from .locality import compute_locality_metrics, evaluate_locality
from .semantic import compute_bbox_from_mask, expand_bbox, evaluate_semantic

__all__ = [
    "compute_locality_metrics",
    "evaluate_locality",
    "compute_bbox_from_mask",
    "expand_bbox",
    "evaluate_semantic",
]
