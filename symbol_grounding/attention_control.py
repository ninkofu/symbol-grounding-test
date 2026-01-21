"""Cross-attention control utilities for prompt-to-prompt editing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class AttentionCapture:
    """Store attention probabilities for later reuse."""

    maps: list[torch.Tensor]

    def __init__(self) -> None:
        self.maps = []

    def reset(self) -> None:
        self.maps = []

    def add(self, attn: torch.Tensor) -> None:
        self.maps.append(attn.detach().clone())

    def get(self, index: int) -> Optional[torch.Tensor]:
        if index >= len(self.maps):
            return None
        return self.maps[index]


class AttentionStoreProcessor:
    """Diffusers attention processor that stores attention maps."""

    def __init__(self, store: AttentionCapture):
        self.store = store
        self.counter = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        attn_probs = attn.get_attention_scores(query, key, attention_mask)
        self.store.add(attn_probs)
        self.counter += 1

        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AttentionReplaceProcessor:
    """Replace attention maps for locked tokens with stored maps."""

    def __init__(self, store: AttentionCapture, token_mask: torch.Tensor):
        self.store = store
        self.token_mask = token_mask
        self.counter = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        attn_probs = attn.get_attention_scores(query, key, attention_mask)
        stored = self.store.get(self.counter)
        if stored is not None:
            mask = self.token_mask.to(attn_probs.device)
            while mask.dim() < attn_probs.dim():
                mask = mask.unsqueeze(0)
            attn_probs = torch.where(mask, stored, attn_probs)

        self.counter += 1
        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def build_token_mask(tokenizer, prompt: str, lock_tokens: Iterable[str]) -> torch.Tensor:
    """Build a boolean mask for token positions matching any lock token."""
    encoded = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    lock_set = {tok.lower() for tok in lock_tokens}
    mask = torch.zeros(len(tokens), dtype=torch.bool)
    for idx, tok in enumerate(tokens):
        if any(lock in tok.lower() for lock in lock_set):
            mask[idx] = True
    return mask


def apply_attention_processors(pipe, processor_cls, **kwargs) -> None:
    """Apply an attention processor to all cross-attention layers in a diffusers pipe."""
    processors = {}
    for name, module in pipe.unet.attn_processors.items():
        if "attn2" in name:
            processors[name] = processor_cls(**kwargs)
        else:
            processors[name] = module
    pipe.unet.set_attn_processor(processors)


def prompt_to_prompt_edit(
    pipe,
    base_prompt: str,
    edit_prompt: str,
    lock_tokens: Iterable[str],
    **kwargs,
):
    """Run a base prompt pass and reuse locked attention maps in the edit pass."""
    store = AttentionCapture()
    apply_attention_processors(pipe, AttentionStoreProcessor, store=store)
    _ = pipe(prompt=base_prompt, **kwargs)

    token_mask = build_token_mask(pipe.tokenizer, edit_prompt, lock_tokens)
    apply_attention_processors(pipe, AttentionReplaceProcessor, store=store, token_mask=token_mask)
    return pipe(prompt=edit_prompt, **kwargs)


__all__ = [
    "AttentionCapture",
    "AttentionStoreProcessor",
    "AttentionReplaceProcessor",
    "build_token_mask",
    "apply_attention_processors",
    "prompt_to_prompt_edit",
]
