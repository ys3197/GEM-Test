"""
The two structures claim A compares.

GEM's post makes a specific complaint about the conventional design:

    Existing approaches compress user behavior sequences into compact vectors
    for downstream tasks, which risks losing critical engagement signals.

That is exactly what `PoolThenInteract` does, and it is the structure the 2024
sequence-learning post describes: pool N events into M summaries once, then let
those summaries interact with the non-sequence features. Whatever the pooling
discarded is gone for every layer that follows.

`InterFormer` is the alternative GEM describes — "parallel summarization with an
interleaving structure that alternates between sequence learning and
cross-feature interaction layers" — and the property that matters is that the
**full sequence survives every layer**. Each block refines the sequence, takes a
fresh summary of it, and lets the field side interact with that summary. Depth
therefore buys repeated access to the complete history rather than repeated
processing of one early compression.

The prediction claim is about depth: if pooling really destroys signal, the
pooled variant should saturate earlier as layers are added, because it is
refining a fixed summary while the interleaved one keeps returning to the
source.

**Both variants must be comparable in size**, or the experiment measures
capacity instead of structure. `parameter_count` is exposed so runs can report
it, and the depth sweep in M2 should tune widths until the two are within a few
percent of each other.

A caveat this repo will state rather than hide: the M0 measurement found a
median history of 6-7 events. Pooling 7 events into 4 summaries is barely a
compression, so this data may not be able to separate the two structures at all.
That would be a fact about the dataset, not a refutation of GEM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.embeddings import FieldEmbeddings, ItemFeatureTable, sequence_mask
from models.sequence import AttentionPooling, EventModel
from models.wukong import Wukong, WukongLayer


class _Base(nn.Module):
    """Shared plumbing so the two variants differ only where they should."""

    def __init__(
        self,
        embeddings: FieldEmbeddings,
        item_features: ItemFeatureTable,
        dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.event_model = EventModel(embeddings, item_features, dim, dropout=dropout)
        self.dim = dim

    @property
    def parameter_count(self) -> int:
        """Excludes embedding tables, which are shared and dominate the total."""
        return sum(p.numel() for n, p in self.named_parameters()
                   if not n.startswith("event_model.embeddings"))


class PoolThenInteract(_Base):
    """
    Compress the sequence once, then interact. The conventional structure.

    The sequence is summarised a single time, up front. Every subsequent layer
    sees only that summary, so any signal the pooling dropped is unrecoverable
    no matter how much depth follows.
    """

    def __init__(
        self,
        embeddings: FieldEmbeddings,
        item_features: ItemFeatureTable,
        n_fields: int,
        dim: int,
        n_layers: int = 3,
        n_queries: int = 4,
        n_heads: int = 4,
        n_fmb: int = 16,
        n_lcb: int = 16,
        rank: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(embeddings, item_features, dim, dropout)
        self.pooling = AttentionPooling(dim, n_queries, n_heads, dropout)
        self.wukong = Wukong(
            n_fields=n_fields + n_queries, dim=dim, n_layers=n_layers,
            n_fmb=n_fmb, n_lcb=n_lcb, rank=rank, dropout=dropout,
        )
        self.n_out = self.wukong.n_out

    def forward(self, fields, hist_items, hist_gap, hist_len, candidate):
        mask = sequence_mask(hist_len, hist_items.size(1))
        events = self.event_model(hist_items, hist_gap)
        summary = self.pooling(events, mask, candidate)          # [B, M, D]
        return self.wukong(torch.cat([fields, summary], dim=1))


class InterFormer(_Base):
    """
    Alternate between refining the sequence and interacting with it.

    Each block does three things, and the order is the point:

      1. self-attention over the **full** sequence — never compressed
      2. a fresh candidate-keyed summary of the refined sequence
      3. a field-side interaction layer over (fields + that summary)

    Because step 1 always operates on the complete history, the summary handed
    to the field side at layer 3 is drawn from a sequence representation that
    has been refined twice — not from a vector frozen at layer 0.
    """

    def __init__(
        self,
        embeddings: FieldEmbeddings,
        item_features: ItemFeatureTable,
        n_fields: int,
        dim: int,
        n_layers: int = 3,
        n_queries: int = 4,
        n_heads: int = 4,
        n_fmb: int = 16,
        n_lcb: int = 16,
        rank: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(embeddings, item_features, dim, dropout)

        seq_layers, pools, field_layers = [], [], []
        n_in = n_fields
        for _ in range(n_layers):
            seq_layers.append(nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            ))
            pools.append(AttentionPooling(dim, n_queries, n_heads, dropout))
            field_layers.append(
                WukongLayer(n_in + n_queries, dim, n_fmb, n_lcb, rank, dropout))
            n_in = n_fmb + n_lcb

        self.seq_layers = nn.ModuleList(seq_layers)
        self.pools = nn.ModuleList(pools)
        self.field_layers = nn.ModuleList(field_layers)
        self.n_out = n_in

    def forward(self, fields, hist_items, hist_gap, hist_len, candidate):
        mask = sequence_mask(hist_len, hist_items.size(1))
        events = self.event_model(hist_items, hist_gap)

        for seq_layer, pool, field_layer in zip(
                self.seq_layers, self.pools, self.field_layers):
            # 序列在这里被精炼，但**长度不变**——完整历程始终保留
            events = seq_layer(events, src_key_padding_mask=~mask)
            summary = pool(events, mask, candidate)
            fields = field_layer(torch.cat([fields, summary], dim=1))

        return fields


VARIANTS = {"pooled": PoolThenInteract, "interleaved": InterFormer}


def build_variant(name: str, **kwargs) -> _Base:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[name](**kwargs)
