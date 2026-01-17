"""Typed slot modules for nouns, adjectives and verbs.

This module contains three classes that encapsulate the logic for
handling different parts of speech in the latent space.  Each module
accepts input representations and produces transformed outputs:

* :class:`NounModule` maps continuous image features to discrete
  prototypes (codebook vectors) using vector quantisation.  It
  represents object identity.
* :class:`AdjectiveModule` injects style information (colour, texture)
  into a base representation via feature‑wise modulation (e.g. AdaIN or
  FiLM).
* :class:`VerbModule` encodes transformations or relations as
  functions that act on pose or other continuous variables.

These classes are currently stubs; replace the bodies with real
implementations as you develop the system.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn


class NounModule(nn.Module):
    """Map continuous features to discrete codebook entries.

    In a full implementation this module would implement vector
    quantisation (e.g. VQ‑VAE) to map each object to a prototype vector
    representing its identity.  Here we provide a simple lookup based on
    k‑means centroids or an identity mapping.
    """

    def __init__(self, num_codes: int = 512, code_dim: int = 64):
        super().__init__()
        # Create a learnable codebook (prototype vectors)
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise features to the nearest codebook entry.

        Args:
            features: Tensor of shape ``(batch, slot_dim)`` representing
                per‑object features extracted from the encoder.

        Returns:
            A tuple ``(quantised, indices)`` where ``quantised`` has the
            same shape as ``features`` and contains the nearest codebook
            vector, and ``indices`` are the integer code indices.
        """
        # TODO: implement nearest neighbour search.  For now return input.
        indices = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        return features, indices


class AdjectiveModule(nn.Module):
    """Inject style attributes via feature‑wise modulation.

    Real implementations might use AdaIN (Adaptive Instance
    Normalisation) or FiLM (Feature‑wise Linear Modulation) to apply
    colour, texture or other stylistic variations to a base noun
    representation.
    """

    def __init__(self, feature_dim: int = 64, attribute_dim: int = 16):
        super().__init__()
        # Two linear layers to produce scaling (gamma) and bias (beta)
        self.to_gamma = nn.Linear(attribute_dim, feature_dim)
        self.to_beta = nn.Linear(attribute_dim, feature_dim)

    def forward(self, noun_feats: torch.Tensor, attr_vec: torch.Tensor) -> torch.Tensor:
        """Apply style modulation to noun features.

        Args:
            noun_feats: Tensor of shape ``(batch, feature_dim)``.
            attr_vec: Tensor of shape ``(batch, attribute_dim)`` representing
                encoded adjectives (e.g. colour embeddings).

        Returns:
            Tensor of shape ``(batch, feature_dim)`` with modulated features.
        """
        gamma = self.to_gamma(attr_vec)
        beta = self.to_beta(attr_vec)
        return gamma * noun_feats + beta


class VerbModule(nn.Module):
    """Represent actions or relations as learned transformations.

    This module treats verbs as functions operating on pose or relation
    variables (e.g. position coordinates).  For example, "falling"
    decreases the y‑coordinate over time, while "rotating" applies a
    rotation matrix.  In this stub we implement a simple affine
    transformation learned from inputs.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, coords: torch.Tensor, verb_vec: torch.Tensor) -> torch.Tensor:
        """Apply a transformation to coordinates based on a verb embedding.

        Args:
            coords: Tensor of shape ``(batch, num_objects, 2)`` containing
                x and y positions.
            verb_vec: Tensor of shape ``(batch, verb_dim)`` representing
                encoded verbs (ignored in this stub).

        Returns:
            Tensor of shape ``(batch, num_objects, 2)`` with transformed
            coordinates.
        """
        # Flatten coords to (batch*num_objects, 2)
        b, n, d = coords.shape
        flat = coords.view(b * n, d)
        # In a real implementation, verb_vec would modulate the
        # transformation.  Here we ignore it and apply a learned affine map.
        transformed = self.net(flat)
        return transformed.view(b, n, d)


__all__ = ["NounModule", "AdjectiveModule", "VerbModule"]