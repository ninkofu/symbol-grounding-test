"""Simple text encoder utilities for multimodal conditioning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


@dataclass
class TextEncoderConfig:
    vocab: Sequence[str]
    embedding_dim: int = 64
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"


class SimpleTextEncoder(nn.Module):
    """Bag-of-words style encoder with mean pooling."""

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config
        vocab = [config.pad_token, config.unk_token, *config.vocab]
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.embedding = nn.Embedding(len(vocab), config.embedding_dim)

    def tokenize(self, text: str) -> List[int]:
        tokens = text.lower().replace(",", " ").split()
        ids = [
            self.token_to_id.get(token, self.token_to_id[self.config.unk_token])
            for token in tokens
        ]
        if not ids:
            ids = [self.token_to_id[self.config.unk_token]]
        return ids

    def forward(self, texts: Iterable[str]) -> torch.Tensor:
        ids_list = [self.tokenize(text) for text in texts]
        max_len = max(len(ids) for ids in ids_list)
        pad_id = self.token_to_id[self.config.pad_token]
        batch_ids = [
            ids + [pad_id] * (max_len - len(ids))
            for ids in ids_list
        ]
        ids_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.embedding.weight.device)
        embeddings = self.embedding(ids_tensor)
        mask = (ids_tensor != pad_id).float().unsqueeze(-1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled


def build_shape_color_vocab() -> List[str]:
    return [
        "circle",
        "square",
        "triangle",
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "and",
        "a",
    ]


__all__ = ["TextEncoderConfig", "SimpleTextEncoder", "build_shape_color_vocab"]
