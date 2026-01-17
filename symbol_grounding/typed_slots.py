"""Typed slot modules for nouns, adjectives and verbs."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NounModule(nn.Module):
    """VQ-VAE style codebook for discrete noun identities."""

    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 64,
        input_dim: Optional[int] = None,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.input_dim = input_dim or code_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        if self.input_dim != code_dim:
            self.to_code = nn.Linear(self.input_dim, code_dim)
            self.from_code = nn.Linear(code_dim, self.input_dim)
        else:
            self.to_code = None
            self.from_code = None

        self.last_vq_loss: Optional[torch.Tensor] = None

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = features.shape
        flat = features.reshape(-1, original_shape[-1])

        if self.to_code is not None:
            flat_code = self.to_code(flat)
        else:
            flat_code = flat

        # Compute nearest codebook entry
        codebook = self.codebook.weight
        dist = (
            torch.sum(flat_code**2, dim=1, keepdim=True)
            - 2 * flat_code @ codebook.t()
            + torch.sum(codebook**2, dim=1)
        )
        indices = torch.argmin(dist, dim=1)
        quantized = self.codebook(indices)

        # VQ losses (stored for optional use)
        commitment_loss = F.mse_loss(flat_code.detach(), quantized)
        codebook_loss = F.mse_loss(flat_code, quantized.detach())
        self.last_vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        if self.from_code is not None:
            quantized_out = self.from_code(quantized)
        else:
            quantized_out = quantized

        # Straight-through estimator
        quantized_out = flat + (quantized_out - flat).detach()
        quantized_out = quantized_out.reshape(original_shape)
        indices = indices.reshape(original_shape[:-1])
        return quantized_out, indices


class AdjectiveModule(nn.Module):
    """Minimal FiLM/AdaIN style modulation of noun features."""

    def __init__(self, feature_dim: int = 64, attribute_dim: int = 16, mode: str = "film"):
        super().__init__()
        self.feature_dim = feature_dim
        self.attribute_dim = attribute_dim
        self.mode = mode

        self.to_gamma = nn.Linear(attribute_dim, feature_dim)
        self.to_beta = nn.Linear(attribute_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim) if mode == "adain" else None

    def forward(self, noun_feats: torch.Tensor, attr_vec: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            feats = self.norm(noun_feats)
        else:
            feats = noun_feats

        gamma = self.to_gamma(attr_vec)
        beta = self.to_beta(attr_vec)

        # Broadcast to match noun_feats shape
        while gamma.dim() < feats.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return feats * (1.0 + gamma) + beta


class VerbModule(nn.Module):
    """Apply 2D state transitions with fixed or learned verbs."""

    def __init__(
        self,
        verbs: Optional[list[str]] = None,
        mode: str = "fixed",
        verb_dim: int = 16,
        hidden_dim: int = 64,
        clamp: bool = True,
    ):
        super().__init__()
        self.verbs = verbs or ["stay", "fall", "move_right", "move_left", "move_up", "move_down"]
        self.mode = mode
        self.clamp = clamp
        self.verb_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.verbs)}

        deltas = torch.tensor(
            [
                [0.0, 0.0],   # stay
                [0.0, 0.2],   # fall (increase y)
                [0.2, 0.0],   # move_right
                [-0.2, 0.0],  # move_left
                [0.0, -0.2],  # move_up
                [0.0, 0.2],   # move_down
            ],
            dtype=torch.float32,
        )
        if deltas.shape[0] < len(self.verbs):
            pad = len(self.verbs) - deltas.shape[0]
            deltas = torch.cat([deltas, torch.zeros(pad, 2)], dim=0)
        self.register_buffer("fixed_deltas", deltas[: len(self.verbs)])

        if mode == "learned":
            self.verb_embedding = nn.Embedding(len(self.verbs), verb_dim)
            self.net = nn.Sequential(
                nn.Linear(verb_dim + 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.verb_embedding = None
            self.net = None

    def forward(self, coords: torch.Tensor, verb: Union[str, torch.Tensor]) -> torch.Tensor:
        if coords.dim() != 3 or coords.shape[-1] != 2:
            raise ValueError("coords must have shape (batch, num_objects, 2)")

        b, n, _ = coords.shape
        device = coords.device

        if isinstance(verb, str):
            if verb not in self.verb_to_idx:
                raise ValueError(f"Unknown verb: {verb}")
            verb_ids = torch.full((b,), self.verb_to_idx[verb], device=device, dtype=torch.long)
            verb_tensor: Optional[torch.Tensor] = None
        elif isinstance(verb, torch.Tensor) and verb.dtype == torch.long:
            verb_ids = verb.to(device).view(-1)
            verb_tensor = None
        else:
            verb_ids = None
            verb_tensor = verb.to(device)  # type: ignore[arg-type]

        if self.mode == "fixed":
            if verb_ids is not None:
                delta = self.fixed_deltas[verb_ids]
            else:
                if verb_tensor.shape[-1] != 2:
                    raise ValueError("Fixed mode expects verb tensor with shape (..., 2)")
                delta = verb_tensor
                if delta.dim() == 1:
                    delta = delta.unsqueeze(0)
            delta = delta.view(-1, 1, 2)
            out = coords + delta
        else:
            if verb_ids is not None:
                verb_embed = self.verb_embedding(verb_ids)
            else:
                verb_embed = verb_tensor
                if verb_embed.dim() == 1:
                    verb_embed = verb_embed.unsqueeze(0)
            verb_embed = verb_embed.view(-1, 1, verb_embed.shape[-1]).expand(b, n, -1)
            inp = torch.cat([coords, verb_embed], dim=-1)
            delta = self.net(inp)
            out = coords + delta

        if self.clamp:
            out = out.clamp(0.0, 1.0)
        return out

    def available_verbs(self) -> list[str]:
        return list(self.verbs)


__all__ = ["NounModule", "AdjectiveModule", "VerbModule"]
