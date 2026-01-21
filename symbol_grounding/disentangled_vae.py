"""Disentangled VAE model with structured latent partitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DisentangledVAEConfig:
    image_size: int = 64
    num_channels: int = 3
    latent_dim: int = 32
    shape_dim: int = 8
    color_dim: int = 8
    position_dim: int = 4
    hidden_dim: int = 256


class DisentangledVAE(nn.Module):
    """Beta-VAE with explicit latent partitions for shape, color and position."""

    def __init__(self, config: DisentangledVAEConfig):
        super().__init__()
        self.config = config

        if config.shape_dim + config.color_dim + config.position_dim > config.latent_dim:
            raise ValueError("Sum of structured latent dims must be <= latent_dim.")

        self.encoder = nn.Sequential(
            nn.Conv2d(config.num_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        conv_out = config.image_size // 16
        enc_dim = 256 * conv_out * conv_out
        self.to_mu = nn.Linear(enc_dim, config.latent_dim)
        self.to_logvar = nn.Linear(enc_dim, config.latent_dim)

        self.decoder_input = nn.Linear(config.latent_dim, enc_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, config.num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.shape_head = nn.Sequential(
            nn.Linear(config.shape_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 3),
        )
        self.color_head = nn.Sequential(
            nn.Linear(config.color_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 7),
        )
        self.position_head = nn.Sequential(
            nn.Linear(config.position_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 4),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        flat = feats.flatten(start_dim=1)
        mu = self.to_mu(flat)
        logvar = self.to_logvar(flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        conv_out = self.config.image_size // 16
        hidden = self.decoder_input(z)
        hidden = hidden.view(z.size(0), 256, conv_out, conv_out)
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def split_latent(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.config
        shape_end = cfg.shape_dim
        color_end = shape_end + cfg.color_dim
        pos_end = color_end + cfg.position_dim
        return {
            "shape": z[:, :shape_end],
            "color": z[:, shape_end:color_end],
            "position": z[:, color_end:pos_end],
            "other": z[:, pos_end:],
        }

    def predict_attributes(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        parts = self.split_latent(z)
        return {
            "shape_logits": self.shape_head(parts["shape"]),
            "color_logits": self.color_head(parts["color"]),
            "position": self.position_head(parts["position"]),
        }


def beta_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl
    return {"loss": loss, "recon": recon_loss, "kl": kl}


__all__ = ["DisentangledVAEConfig", "DisentangledVAE", "beta_vae_loss"]
