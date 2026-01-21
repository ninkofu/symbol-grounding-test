"""Slot Attention + Typed Slots + TPR integration pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .slot_attention import SlotAttentionAutoEncoder, SlotAttentionConfig
from .typed_slots import NounModule, AdjectiveModule, VerbModule
from .tpr import bind


@dataclass
class SlotTPRConfig:
    num_roles: int = 8
    role_dim: int = 32
    noun_codes: int = 512
    noun_dim: int = 64
    attribute_dim: int = 16
    verb_mode: str = "fixed"


class SlotTPRPipeline(nn.Module):
    """End-to-end pipeline from images to bound TPR representations."""

    def __init__(
        self,
        slot_config: Optional[SlotAttentionConfig] = None,
        tpr_config: Optional[SlotTPRConfig] = None,
    ) -> None:
        super().__init__()
        self.slot_config = slot_config or SlotAttentionConfig()
        self.tpr_config = tpr_config or SlotTPRConfig()

        self.slot_ae = SlotAttentionAutoEncoder(self.slot_config)
        self.noun_module = NounModule(num_codes=self.tpr_config.noun_codes, code_dim=self.tpr_config.noun_dim)
        self.adj_module = AdjectiveModule(
            feature_dim=self.tpr_config.noun_dim,
            attribute_dim=self.tpr_config.attribute_dim,
        )
        self.verb_module = VerbModule(mode=self.tpr_config.verb_mode)
        self.role_embedding = nn.Embedding(self.tpr_config.num_roles, self.tpr_config.role_dim)
        self.role_project = nn.Linear(self.slot_config.slot_dim, self.tpr_config.role_dim)

    def forward(
        self,
        images: torch.Tensor,
        attr_vecs: Optional[torch.Tensor] = None,
        verb: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        recon, info = self.slot_ae(images)
        slots = info["slots"]

        noun_feats, noun_indices = self.noun_module(slots)
        if attr_vecs is None:
            attr_vecs = torch.zeros(
                noun_feats.shape[0],
                noun_feats.shape[1],
                self.tpr_config.attribute_dim,
                device=noun_feats.device,
            )
        adj_feats = self.adj_module(noun_feats, attr_vecs)

        roles = self.role_project(slots)
        role_ids = torch.arange(roles.shape[1], device=roles.device).clamp(max=self.tpr_config.num_roles - 1)
        role_vecs = self.role_embedding(role_ids).unsqueeze(0).expand(roles.shape[0], -1, -1)

        bound = bind(adj_feats, role_vecs)
        if verb is not None:
            coords = torch.rand(adj_feats.shape[0], adj_feats.shape[1], 2, device=adj_feats.device)
            coords = self.verb_module(coords, verb)
        else:
            coords = torch.rand(adj_feats.shape[0], adj_feats.shape[1], 2, device=adj_feats.device)

        return {
            "recon": recon,
            "slots": slots,
            "noun_indices": noun_indices,
            "adj_feats": adj_feats,
            "role_vecs": role_vecs,
            "bound": bound,
            "coords": coords,
        }


__all__ = ["SlotTPRPipeline", "SlotTPRConfig"]
