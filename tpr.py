"""Tensor‑product representation utilities.

The tensor product (outer product) offers a principled way to bind
symbols (fillers) to roles without mixing them up.  In the context of
symbol grounding, tensor products can be used to combine noun,
adjective and verb representations with their respective roles (e.g.
subject, predicate, object) and later unbind them via inner products.

This module provides simple helper functions for binding and unbinding
using PyTorch.  The operations are implemented as batch‑compatible
functions so they can be used directly in neural networks.
"""
from __future__ import annotations

import torch


def bind(filler: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    """Compute the tensor (outer) product between filler and role.

    Args:
        filler: Tensor of shape ``(..., F)`` representing the content to
            bind.  ``...`` can be any number of leading dimensions.
        role: Tensor of shape ``(..., R)`` representing the role vector.  The
            leading dimensions must be broadcastable to those of
            ``filler``.

    Returns:
        A tensor of shape ``(..., F, R)`` containing the outer products.
    """
    # Use einsum for clarity: out[i,...,f,r] = filler[i,...,f] * role[i,...,r]
    return torch.einsum("...f,...r->...fr", filler, role)


def unbind(bound: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    """Recover the filler from a bound tensor via contraction with the role.

    Args:
        bound: Tensor of shape ``(..., F, R)`` produced by :func:`bind`.
        role: Tensor of shape ``(..., R)`` representing the role vector used
            during binding.

    Returns:
        A tensor of shape ``(..., F)`` that approximates the original filler.

    Note:
        Exact recovery requires role vectors to be orthonormal.  In
        practice one may learn role vectors or use random orthogonal
        vectors so that unbinding is approximate but still distinct.
    """
    return torch.einsum("...fr,...r->...f", bound, role)


__all__ = ["bind", "unbind"]