"""
The assembled model.

One config object drives every size in the network, because claim C needs to
sweep model scale and a sweep is only meaningful if a single number moves all
the widths together in a fixed ratio. `MiniGEMConfig.scaled` exists for that.

The forward pass is deliberately thin. All the structure lives in the variant
(`pooled` or `interleaved`), so the two arms of claim A share every other part
of the model — embeddings, features, head, loss — and differ only where the
claim says they should.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from data.dataset import ITEM_FEATURE_COLUMNS, N_TIME_BUCKETS
from models.embeddings import FieldEmbeddings, ItemFeatureTable
from models.interformer import build_variant

# 进入非序列塔的字段，顺序固定——Wukong 的字段轴依赖它稳定
NON_SEQUENCE_FIELDS = ["user_id", "item_id"] + ITEM_FEATURE_COLUMNS


@dataclass
class MiniGEMConfig:
    variant: str = "pooled"          # "pooled" | "interleaved"
    dim: int = 32
    n_layers: int = 3
    n_queries: int = 4               # 序列摘要的数量 M
    n_heads: int = 4
    n_fmb: int = 16
    n_lcb: int = 16
    rank: int = 8
    head_hidden: tuple[int, ...] = (256, 64)
    dropout: float = 0.1

    def scaled(self, factor: float) -> "MiniGEMConfig":
        """
        One knob that moves every width together.

        Claim C asks whether quality is log-linear in compute. That question is
        only well posed if the models on the curve are the same shape at
        different sizes — otherwise the curve mixes scale with architecture.
        Depth is kept fixed here; widths carry the scaling.
        """
        def s(v: int, lo: int = 1) -> int:
            return max(lo, int(round(v * factor)))

        return MiniGEMConfig(
            variant=self.variant,
            dim=s(self.dim, 8),
            n_layers=self.n_layers,
            n_queries=s(self.n_queries, 2),
            n_heads=self.n_heads,
            n_fmb=s(self.n_fmb, 4),
            n_lcb=s(self.n_lcb, 4),
            rank=s(self.rank, 2),
            head_hidden=tuple(s(h, 16) for h in self.head_hidden),
            dropout=self.dropout,
        )


class PredictionHead(nn.Module):
    """Flattened interaction fields -> one logit."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MiniGEM(nn.Module):
    """
    Sequence tower and non-sequence tower, joined by the chosen variant.

    `vocab_sizes` comes from the domain's vocab.json; the time-gap vocabulary is
    added here because it is produced by the dataset rather than by feature
    encoding.
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        item_feature_table: torch.Tensor,
        config: MiniGEMConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MiniGEMConfig()
        cfg = self.config

        sizes = dict(vocab_sizes)
        sizes["time_gap"] = N_TIME_BUCKETS
        # FieldEmbeddings 的字段顺序即 dict 顺序；显式固定，避免上游改动悄悄换轴
        self.embeddings = FieldEmbeddings(
            {k: sizes[k] for k in NON_SEQUENCE_FIELDS + ["time_gap"]}, cfg.dim)
        self.item_features = ItemFeatureTable(item_feature_table, ITEM_FEATURE_COLUMNS)

        self.body = build_variant(
            cfg.variant,
            embeddings=self.embeddings,
            item_features=self.item_features,
            n_fields=len(NON_SEQUENCE_FIELDS),
            dim=cfg.dim,
            n_layers=cfg.n_layers,
            n_queries=cfg.n_queries,
            n_heads=cfg.n_heads,
            n_fmb=cfg.n_fmb,
            n_lcb=cfg.n_lcb,
            rank=cfg.rank,
            dropout=cfg.dropout,
        )
        self.head = PredictionHead(self.body.n_out * cfg.dim,
                                   cfg.head_hidden, cfg.dropout)

    def build_fields(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """`[B, F, D]` of non-sequence field embeddings for the candidate."""
        ids = self.item_features(batch["item"])
        ids["user_id"] = batch["user"]
        ids["item_id"] = batch["item"]
        return torch.stack(
            [self.embeddings.lookup(f, ids[f]) for f in NON_SEQUENCE_FIELDS], dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns logits; the loss applies the sigmoid."""
        fields = self.build_fields(batch)
        candidate = self.embeddings.lookup("item_id", batch["item"])
        out = self.body(fields, batch["hist"], batch["hist_gap"],
                        batch["hist_len"], candidate)
        return self.head(out.flatten(start_dim=1))

    def embedding_output(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The representation just before the head.

        M3 needs this: representation transfer hands a teacher's internal
        features to a student as input, so the teacher has to be able to expose
        them without running its own head.
        """
        fields = self.build_fields(batch)
        candidate = self.embeddings.lookup("item_id", batch["item"])
        out = self.body(fields, batch["hist"], batch["hist_gap"],
                        batch["hist_len"], candidate)
        return out.flatten(start_dim=1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_dense_parameters(self) -> int:
        """Excluding embedding tables, which scale with the vocabulary, not the model."""
        return self.n_parameters - sum(p.numel() for p in self.embeddings.parameters())
