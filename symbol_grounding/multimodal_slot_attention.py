"""Multimodal Slot Attention autoencoder with text conditioning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn

from .slot_attention import SlotAttentionAutoEncoder, SlotAttentionConfig, _build_grid


@dataclass
class MultiModalSlotAttentionConfig:
    slot_config: SlotAttentionConfig = SlotAttentionConfig()
    text_dim: int = 64
    conditioning_scale: float = 1.0


class MultiModalSlotAttentionAutoEncoder(nn.Module):
    """Slot Attention autoencoder conditioned on text embeddings."""

    def __init__(self, config: MultiModalSlotAttentionConfig):
        super().__init__()
        self.config = config
        self.slot_ae = SlotAttentionAutoEncoder(config.slot_config)
        self.text_to_slot = nn.Linear(config.text_dim, config.slot_config.slot_dim)

    def encode(self, images: torch.Tensor, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.slot_ae.encoder_cnn(images)
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)

        grid = _build_grid(h, w, feats.device)
        grid = grid.reshape(1, h * w, 2)
        feats = torch.cat([feats, grid.expand(b, -1, -1)], dim=-1)
        feats = self.slot_ae.encoder_pos(feats)
        feats = self.slot_ae.encoder_ln(feats)

        cond = self.text_to_slot(text_emb).unsqueeze(1)
        feats = feats + self.config.conditioning_scale * cond
        slots, attn = self.slot_ae.slot_attention(feats)
        return slots, attn

    def forward(self, images: torch.Tensor, text_emb: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        slots, attn = self.encode(images, text_emb)
        recon, masks = self.slot_ae.decode(slots)
        return recon, {"slots": slots, "attn": attn, "masks": masks}


__all__ = ["MultiModalSlotAttentionConfig", "MultiModalSlotAttentionAutoEncoder"]
