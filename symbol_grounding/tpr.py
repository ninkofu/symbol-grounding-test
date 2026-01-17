"""Tensor-product representation utilities."""
from __future__ import annotations

import torch


def bind(filler: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    """Compute the tensor (outer) product between filler and role."""
    return torch.einsum("...f,...r->...fr", filler, role)


def unbind(bound: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    """Recover the filler from a bound tensor via contraction with the role."""
    return torch.einsum("...fr,...r->...f", bound, role)


__all__ = ["bind", "unbind"]
