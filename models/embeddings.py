"""
Turning integer ids into vectors.

Every feature in this model is categorical — item ids, store ids, and the
quantile buckets standing in for price, rating, popularity and time gaps. That
uniformity is deliberate: the non-sequence tower learns *interactions between
field embeddings*, so every field has to arrive as an embedding of the same
width and they have to stack into one `[B, F, D]` tensor.

Item-side fields are not carried in the batch. The batch holds an item id, and
the item's attributes are looked up from a table held as a buffer. That keeps
the collate function cheap and means a history of L events costs one gather
rather than L dictionary lookups.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PAD = 0


class FieldEmbeddings(nn.Module):
    """
    One embedding table per categorical field, all sharing a width.

    `padding_idx=0` matters everywhere: id 0 is PAD by convention across the
    whole pipeline, and its embedding must stay at zero so padded positions
    contribute nothing to a sum and receive no gradient.
    """

    def __init__(self, vocab_sizes: dict[str, int], dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.fields = list(vocab_sizes)
        self.tables = nn.ModuleDict({
            name: nn.Embedding(size, dim, padding_idx=PAD)
            for name, size in vocab_sizes.items()
        })
        for table in self.tables.values():
            nn.init.normal_(table.weight, std=0.01)
            with torch.no_grad():
                table.weight[PAD].zero_()

    def forward(self, ids: dict[str, torch.Tensor]) -> torch.Tensor:
        """`{field: [B]}` -> `[B, F, D]`, fields in a fixed order."""
        return torch.stack([self.tables[f](ids[f]) for f in self.fields], dim=1)

    def lookup(self, field: str, ids: torch.Tensor) -> torch.Tensor:
        return self.tables[field](ids)

    @property
    def n_fields(self) -> int:
        return len(self.fields)


class ItemFeatureTable(nn.Module):
    """
    Static item attributes, indexed by item id.

    Registered as a buffer rather than kept in the batch so that both the
    candidate and every event in a user's history can be enriched with one
    gather. Not a parameter — these are observed facts, not learned.
    """

    def __init__(self, table: torch.Tensor, columns: list[str]) -> None:
        super().__init__()
        self.register_buffer("table", table, persistent=False)
        self.columns = columns

    def forward(self, item_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """`[...]` of item ids -> `{column: [...]}` of attribute ids."""
        gathered = self.table[item_ids]                      # [..., n_columns]
        return {c: gathered[..., i] for i, c in enumerate(self.columns)}


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    `[B]` of lengths -> `[B, L]` boolean, True where the position is real.

    Every attention module downstream takes its mask from here rather than
    inferring one from `ids != PAD`. A real item id can never be 0, but relying
    on that couples the model to the vocabulary layout; passing lengths
    explicitly keeps the contract visible.
    """
    device = lengths.device
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the sequence axis, ignoring padded positions. `[B, L, D] -> [B, D]`."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
