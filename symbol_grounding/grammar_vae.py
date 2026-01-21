"""Grammar-aware VAE with noun/adjective/verb latent partitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .typed_slots import NounModule, AdjectiveModule


@dataclass
class GrammarVAEConfig:
    image_size: int = 64
    num_channels: int = 3
    latent_dim: int = 48
    noun_dim: int = 16
    adj_dim: int = 16
    verb_dim: int = 8
    hidden_dim: int = 256
    noun_classes: int = 3
    adj_classes: int = 7


class GrammarVAE(nn.Module):
    """Beta-VAE with explicit noun/adj/verb partitions and typed-slot decoding."""

    def __init__(self, config: GrammarVAEConfig):
        super().__init__()
        self.config = config
        if config.noun_dim + config.adj_dim + config.verb_dim > config.latent_dim:
            raise ValueError("Sum of noun/adj/verb dims must be <= latent_dim.")

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

        self.noun_module = NounModule(num_codes=256, code_dim=config.noun_dim, input_dim=config.noun_dim)
        self.adj_module = AdjectiveModule(feature_dim=config.noun_dim, attribute_dim=config.adj_dim)
        self.verb_proj = nn.Linear(config.verb_dim, config.noun_dim)
        self.other_proj = nn.Linear(config.latent_dim - config.noun_dim - config.adj_dim - config.verb_dim, config.noun_dim)

        decoder_in_dim = config.noun_dim + config.noun_dim
        self.decoder_input = nn.Linear(decoder_in_dim, enc_dim)
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

        self.noun_head = nn.Linear(config.noun_dim, config.noun_classes)
        self.adj_head = nn.Linear(config.adj_dim, config.adj_classes)

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

    def split_latent(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.config
        noun_end = cfg.noun_dim
        adj_end = noun_end + cfg.adj_dim
        verb_end = adj_end + cfg.verb_dim
        return {
            "noun": z[:, :noun_end],
            "adj": z[:, noun_end:adj_end],
            "verb": z[:, adj_end:verb_end],
            "other": z[:, verb_end:],
        }

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parts = self.split_latent(z)
        noun_feats, noun_indices = self.noun_module(parts["noun"])
        adj_feats = self.adj_module(noun_feats, parts["adj"])
        verb_feats = self.verb_proj(parts["verb"])
        other_feats = self.other_proj(parts["other"])
        fused = torch.cat([adj_feats + verb_feats, other_feats], dim=-1)

        conv_out = self.config.image_size // 16
        hidden = self.decoder_input(fused)
        hidden = hidden.view(z.size(0), 256, conv_out, conv_out)
        recon = self.decoder(hidden)
        return recon, {"noun_indices": noun_indices}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, extra = self.decode(z)
        parts = self.split_latent(z)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "noun_logits": self.noun_head(parts["noun"]),
            "adj_logits": self.adj_head(parts["adj"]),
            "noun_indices": extra["noun_indices"],
        }


def grammar_vae_loss(
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


__all__ = ["GrammarVAEConfig", "GrammarVAE", "grammar_vae_loss"]
