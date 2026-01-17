"""Slot Attention autoencoder (minimal working implementation)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SlotAttentionConfig:
    """Configuration parameters for the slot attention model."""

    image_size: int = 64
    num_channels: int = 3
    num_slots: int = 7
    slot_dim: int = 64
    hidden_dim: int = 128
    num_iterations: int = 3


def _build_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, steps=height, device=device)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid


class SlotAttention(nn.Module):
    """Iterative Slot Attention module."""

    def __init__(self, num_slots: int, in_dim: int, slot_dim: int, num_iterations: int, hidden_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.scale = slot_dim**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        self.norm_inputs = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(in_dim, slot_dim)
        self.to_v = nn.Linear(in_dim, slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = F.softplus(self.slots_sigma).expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        attn = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)

            attn_logits = torch.einsum("bnd,bsd->bns", k, q) * self.scale
            attn = F.softmax(attn_logits, dim=-1)
            attn = attn + 1e-8
            attn = attn / attn.sum(dim=1, keepdim=True)

            updates = torch.einsum("bnd,bns->bsd", v, attn)
            slots = self.gru(
                updates.reshape(-1, updates.shape[-1]),
                slots_prev.reshape(-1, slots_prev.shape[-1]),
            )
            slots = slots.reshape(b, self.num_slots, -1)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class SlotAttentionAutoEncoder(nn.Module):
    """Minimal Slot Attention autoencoder with spatial broadcast decoder."""

    def __init__(self, config: SlotAttentionConfig):
        super().__init__()
        self.config = config

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(config.num_channels, config.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, config.slot_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.encoder_pos = nn.Linear(config.slot_dim + 2, config.slot_dim)
        self.encoder_ln = nn.LayerNorm(config.slot_dim)

        self.slot_attention = SlotAttention(
            num_slots=config.num_slots,
            in_dim=config.slot_dim,
            slot_dim=config.slot_dim,
            num_iterations=config.num_iterations,
            hidden_dim=config.hidden_dim,
        )

        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(config.slot_dim + 2, config.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim, 4, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder_cnn(x)
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)

        grid = _build_grid(h, w, feats.device).reshape(1, h * w, 2)
        feats = torch.cat([feats, grid.expand(b, -1, -1)], dim=-1)
        feats = self.encoder_pos(feats)
        feats = self.encoder_ln(feats)

        slots, attn = self.slot_attention(feats)
        return slots, attn

    def decode(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, d = slots.shape
        h = w = self.config.image_size

        grid = _build_grid(h, w, slots.device)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(b, k, h, w, 2)

        slots = slots.unsqueeze(2).unsqueeze(2).expand(b, k, h, w, d)
        decoder_in = torch.cat([slots, grid], dim=-1)
        decoder_in = decoder_in.reshape(b * k, h, w, d + 2).permute(0, 3, 1, 2)

        out = self.decoder_cnn(decoder_in)
        out = out.reshape(b, k, 4, h, w)
        recons = out[:, :, :3, :, :]
        masks = out[:, :, 3:, :, :]
        masks = F.softmax(masks, dim=1)

        recon = torch.sum(recons * masks, dim=1)
        return recon, masks

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        slots, attn = self.encode(x)
        recon, masks = self.decode(slots)
        return recon, {"slots": slots, "attn": attn, "masks": masks}


__all__ = ["SlotAttentionConfig", "SlotAttentionAutoEncoder"]
