"""Object‑centric autoencoder using slot attention.

This module defines a skeleton for a slot attention based autoencoder.
Slot attention is a mechanism that allocates a fixed number of latent
"slots" to explain different parts of an image.  By combining slot
attention with a variational or deterministic autoencoder, one can learn
object‑centred representations that disentangle factors such as
identity (noun), appearance (adjective) and pose (verb).  See the
accompanying README for high‑level details.

The implementation below is deliberately incomplete: it illustrates the
expected API and highlights where key components should be inserted.
Replace the ``NotImplementedError`` exceptions with real layers and
training loops when building a working model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn


@dataclass
class SlotAttentionConfig:
    """Configuration parameters for the slot attention model."""

    image_size: int = 64  # width and height of input images
    num_channels: int = 3  # number of image channels
    num_slots: int = 7  # number of slots to allocate
    slot_dim: int = 64  # dimensionality of each slot
    hidden_dim: int = 128  # hidden dimension in encoder/decoder
    num_iterations: int = 3  # number of slot attention iterations


class SlotAttentionAutoEncoder(nn.Module):
    """Skeleton implementation of a slot attention autoencoder.

    This class defines the high‑level structure: an encoder to compute
    per‑pixel features, a slot attention module to aggregate these
    features into a fixed number of slots, and a decoder to reconstruct
    the image from the slots.  The actual layers are left undefined.
    """

    def __init__(self, config: SlotAttentionConfig):
        super().__init__()
        self.config = config
        # TODO: define encoder network (e.g. CNN producing feature map)
        self.encoder = nn.Identity()
        # TODO: define slot attention module
        self.slot_attention = nn.Identity()
        # TODO: define decoder network to reconstruct image from slots
        self.decoder = nn.Identity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image into slot features.

        Args:
            x: Tensor of shape ``(batch, channels, height, width)``.

        Returns:
            A tensor of shape ``(batch, num_slots, slot_dim)`` containing
            the latent representation for each slot.
        """
        # TODO: implement encoder and slot attention here
        raise NotImplementedError("SlotAttentionAutoEncoder.encode is not implemented")

    def decode(self, slots: torch.Tensor) -> torch.Tensor:
        """Decode slot features back into an image.

        Args:
            slots: Tensor of shape ``(batch, num_slots, slot_dim)``.

        Returns:
            Reconstructed images of shape ``(batch, channels, height, width)``.
        """
        # TODO: implement decoder to reconstruct from slots
        raise NotImplementedError("SlotAttentionAutoEncoder.decode is not implemented")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Compute reconstruction and latent slots.

        Returns the reconstructed image and any additional outputs (e.g.
        per‑slot masks) required for training.
        """
        slots = self.encode(x)
        recon = self.decode(slots)
        return recon, slots


__all__ = ["SlotAttentionConfig", "SlotAttentionAutoEncoder"]