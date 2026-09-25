"""
Non-sequence feature interactions: a Wukong-style stack.

Wukong (Meta, ICML 2024) stacks factorization-machine blocks so that feature
interactions compose with depth instead of being fixed at second order. Each
layer runs two paths in parallel and concatenates them:

    FMB — explicit pairwise interactions between field embeddings
    LCB — a linear compression of the same fields, which gives the block a
          path that does not pass through the quadratic term

Stacking matters because one FM layer can only express order-2 interactions.
Two layers reach order 4, three reach order 8: depth buys interaction order,
which is what "scale vertically for deeper interactions" refers to in the GEM
post, while the FMB/LCB widths are the horizontal knob. Those two are the
levers the scaling-law experiment (claim C) turns.

The naive FM computes an n x n interaction matrix. `OptimizedFM` compresses the
field axis to k first, so cost is O(n*k) rather than O(n^2) — irrelevant at the
7 fields this repo starts with, and the reason the block survives if that grows.

**Cross-layer attention is an interpretation, not a specification.** The GEM
post says its Wukong blocks use "cross-layer attention connections" and says
nothing further. What is implemented here lets each layer attend over the
outputs of all previous layers. It is behind a flag and off by default so that
the published architecture and the guess never get conflated in a result.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearCompressBlock(nn.Module):
    """`[B, n_in, D] -> [B, n_out, D]` by mixing along the field axis only."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("on,bnd->bod", self.weight, x)


class OptimizedFM(nn.Module):
    """
    Explicit pairwise interactions, with the field axis compressed first.

    Compressing n fields to k before the outer product turns an n x n
    interaction matrix into n x k. The embedding dimension is contracted away by
    the product itself, so what reaches the MLP is a matrix of *interaction
    strengths* between fields, not a pile of concatenated embeddings.
    """

    def __init__(self, n_in: int, dim: int, rank: int, n_out: int) -> None:
        super().__init__()
        self.n_out, self.dim = n_out, dim
        self.compress = LinearCompressBlock(n_in, rank)
        self.norm = nn.LayerNorm(n_in * rank)
        self.project = nn.Linear(n_in * rank, n_out * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = self.compress(x)                       # [B, k, D]
        interactions = torch.bmm(x, compressed.transpose(1, 2))   # [B, n, k]
        flat = self.norm(interactions.flatten(start_dim=1))
        return self.project(flat).view(-1, self.n_out, self.dim)


class WukongLayer(nn.Module):
    """One stacked-FM layer: FMB and LCB in parallel, concatenated, residual."""

    def __init__(
        self,
        n_in: int,
        dim: int,
        n_fmb: int,
        n_lcb: int,
        rank: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fmb = OptimizedFM(n_in, dim, rank, n_fmb)
        self.lcb = LinearCompressBlock(n_in, n_lcb)
        self.n_out = n_fmb + n_lcb
        # 残差通道：输入和输出的字段数一般不同，用一个线性映射对齐字段轴
        self.residual = LinearCompressBlock(n_in, self.n_out)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.fmb(x), self.lcb(x)], dim=1)
        return self.norm(self.dropout(out) + self.residual(x))


class CrossLayerAttention(nn.Module):
    """
    Let a layer look back at what earlier layers produced.

    Interpretation of GEM's "cross-layer attention connections"; see the module
    docstring. Queries are the current layer's fields, keys and values are every
    field produced so far.
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, memory, memory, need_weights=False)
        return self.norm(x + attended)


class Wukong(nn.Module):
    """
    `[B, n_fields, D]` of field embeddings -> `[B, n_out, D]` of interactions.

    Args:
        n_fields:  how many categorical fields arrive
        dim:       shared embedding width
        n_layers:  depth — the vertical scaling knob
        n_fmb/n_lcb: per-layer widths — the horizontal scaling knob
        rank:      field-axis compression inside the FM
        cross_layer_attention: enable the interpreted GEM addition
    """

    def __init__(
        self,
        n_fields: int,
        dim: int,
        n_layers: int = 3,
        n_fmb: int = 16,
        n_lcb: int = 16,
        rank: int = 8,
        dropout: float = 0.0,
        cross_layer_attention: bool = False,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.cross_layer_attention = cross_layer_attention

        layers, attns = [], []
        n_in = n_fields
        for _ in range(n_layers):
            layers.append(WukongLayer(n_in, dim, n_fmb, n_lcb, rank, dropout))
            n_in = n_fmb + n_lcb
            attns.append(CrossLayerAttention(dim, n_heads)
                         if cross_layer_attention else nn.Identity())

        self.layers = nn.ModuleList(layers)
        self.attns = nn.ModuleList(attns)
        self.n_out = n_in
        self.dim = dim

    @property
    def output_dim(self) -> int:
        return self.n_out * self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = [x] if self.cross_layer_attention else None

        for layer, attn in zip(self.layers, self.attns):
            x = layer(x)
            if self.cross_layer_attention:
                x = attn(x, torch.cat(memory, dim=1))
                memory.append(x)

        return x

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Same, flattened for a prediction head. `[B, n_out * D]`."""
        return self.forward(x).flatten(start_dim=1)
