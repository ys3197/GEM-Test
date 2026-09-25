"""
The sequence tower: one embedding per event, then a summary of the sequence.

Two stages, following the structure Meta describes for event-based features:

**Event model.** Each history event carries several attributes — which item, its
store, category, price band, and how long ago it happened. Every attribute gets
its own embedding; they are concatenated and passed through one linear layer.
That projection is doing real work and is not a formality: different event
streams would arrive with different attribute schemas, and the projection is
where they are forced to a common width so they can enter the same sequence.
The time-gap embedding is added afterwards rather than concatenated, exactly as
positional information is added to token embeddings in a language model — which
is also why no positional encoding appears anywhere below. Order and recency
already live inside each event vector.

**Sequence model.** A fixed number M of learnable queries attend over the N
events, so cost is O(M*N) rather than O(N^2). N is the knob that would otherwise
make long histories unaffordable; M is a tunable budget independent of it.

The queries are **conditioned on the candidate item**. This is the detail that
changes what the tower is: it does not produce "a user vector". It produces a
summary of the part of this user's history that is relevant to *this* candidate,
so the same history reads differently against different candidates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.embeddings import FieldEmbeddings, ItemFeatureTable, sequence_mask


class EventModel(nn.Module):
    """`[B, L]` of item ids and time buckets -> `[B, L, D]` event embeddings."""

    def __init__(
        self,
        embeddings: FieldEmbeddings,
        item_features: ItemFeatureTable,
        dim: int,
        time_field: str = "time_gap",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.item_features = item_features
        self.time_field = time_field

        # item_id 本身 + 它的每个属性
        self.attribute_fields = ["item_id"] + list(item_features.columns)
        self.compress = nn.Linear(len(self.attribute_fields) * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, items: torch.Tensor, time_buckets: torch.Tensor) -> torch.Tensor:
        attrs = self.item_features(items)                       # {col: [B, L]}
        attrs["item_id"] = items

        parts = [self.embeddings.lookup(f, attrs[f]) for f in self.attribute_fields]
        fused = self.compress(torch.cat(parts, dim=-1))         # [B, L, D]

        time_emb = self.embeddings.lookup(self.time_field, time_buckets)
        return self.norm(self.dropout(fused) + time_emb)


class AttentionPooling(nn.Module):
    """
    Summarise N events into M vectors, with the queries keyed by the candidate.

    Cost is O(M*N): the events never attend to each other, so doubling history
    length doubles the cost instead of quadrupling it.
    """

    def __init__(
        self,
        dim: int,
        n_queries: int = 4,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.base_queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.candidate_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        events: torch.Tensor,        # [B, L, D]
        mask: torch.Tensor,          # [B, L]  True = real position
        candidate: torch.Tensor,     # [B, D]
    ) -> torch.Tensor:
        # 每个查询槽 = 一个可学习的基向量 + 候选商品的投影。
        # 基向量负责"看历史的哪个侧面"，候选投影负责"针对这条广告"。
        queries = self.base_queries.unsqueeze(0) + self.candidate_proj(candidate).unsqueeze(1)

        # MultiheadAttention 的 key_padding_mask 约定 True=忽略，与本仓库相反
        summary, _ = self.attn(queries, events, events,
                               key_padding_mask=~mask, need_weights=False)
        return self.norm(summary)                              # [B, M, D]


class SequenceTower(nn.Module):
    """Event model and attention pooling, wired together."""

    def __init__(
        self,
        embeddings: FieldEmbeddings,
        item_features: ItemFeatureTable,
        dim: int,
        n_queries: int = 4,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.event_model = EventModel(embeddings, item_features, dim, dropout=dropout)
        self.pooling = AttentionPooling(dim, n_queries, n_heads, dropout)
        self.dim = dim
        self.n_queries = n_queries

    @property
    def output_dim(self) -> int:
        return self.n_queries * self.dim

    def forward(
        self,
        hist_items: torch.Tensor,     # [B, L]
        hist_gap: torch.Tensor,       # [B, L]
        hist_len: torch.Tensor,       # [B]
        candidate: torch.Tensor,      # [B, D]
    ) -> torch.Tensor:
        mask = sequence_mask(hist_len, hist_items.size(1))
        events = self.event_model(hist_items, hist_gap)
        return self.pooling(events, mask, candidate)           # [B, M, D]
