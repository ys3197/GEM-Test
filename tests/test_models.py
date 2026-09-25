"""
Guardrails for the model components.

Two properties here are not "does it run" checks — they are the architecture:

- **The sequence summary must change with the candidate.** If it does not, the
  tower has silently become a static user vector and the central design point of
  the sequence model ("keyed by the ad to be ranked") is gone. Nothing would
  crash; results would just be quietly worse.
- **Padding must not reach the output.** A mask bug lets fabricated events into
  a user's history, which is leakage wearing the costume of a shape mismatch.

These run on synthetic tensors and need neither the dataset nor a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.embeddings import (  # noqa: E402
    FieldEmbeddings,
    ItemFeatureTable,
    masked_mean,
    sequence_mask,
)
from models.sequence import SequenceTower  # noqa: E402
from models.wukong import Wukong  # noqa: E402

DIM = 16
N_ITEMS = 50
ITEM_COLUMNS = ["store", "category", "price_bucket"]
VOCAB = {
    "user_id": 40, "item_id": N_ITEMS, "store": 12,
    "category": 6, "price_bucket": 9, "time_gap": 10,
}


@pytest.fixture
def parts():
    torch.manual_seed(0)
    emb = FieldEmbeddings(VOCAB, DIM)
    table = torch.randint(3, 6, (N_ITEMS, len(ITEM_COLUMNS)), dtype=torch.long)
    feats = ItemFeatureTable(table, ITEM_COLUMNS)
    tower = SequenceTower(emb, feats, DIM, n_queries=3, n_heads=2).eval()
    return emb, feats, tower


@pytest.fixture
def batch():
    torch.manual_seed(1)
    b, max_len = 8, 6
    lengths = torch.tensor([6, 1, 4, 5, 2, 6, 3, 1])
    hist = torch.randint(3, N_ITEMS, (b, max_len))
    gap = torch.randint(3, 9, (b, max_len))
    mask = sequence_mask(lengths, max_len)
    hist[~mask] = 0
    gap[~mask] = 0
    return hist, gap, lengths, mask


def test_padding_embedding_stays_zero():
    emb = FieldEmbeddings(VOCAB, DIM)
    pad = emb.lookup("item_id", torch.zeros(4, dtype=torch.long))
    assert torch.equal(pad, torch.zeros_like(pad))


def test_sequence_mask_marks_exactly_the_real_positions():
    mask = sequence_mask(torch.tensor([3, 1, 0]), 4)
    assert mask.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]


def test_masked_mean_ignores_padding():
    x = torch.ones(2, 4, DIM)
    x[:, 2:] = 99.0
    out = masked_mean(x, sequence_mask(torch.tensor([2, 2]), 4))
    assert torch.allclose(out, torch.ones(2, DIM))


def test_summary_depends_on_the_candidate(parts, batch):
    """
    The point of keying on the candidate: the same history must read differently
    against different candidates.
    """
    emb, _, tower = parts
    hist, gap, lengths, _ = batch

    a = tower(hist, gap, lengths, emb.lookup("item_id", torch.full((8,), 7)))
    b = tower(hist, gap, lengths, emb.lookup("item_id", torch.full((8,), 31)))

    assert (a - b).abs().mean() > 1e-5


def test_padding_content_cannot_change_the_summary(parts, batch):
    """Garbage written into padded slots must be invisible to the output."""
    emb, _, tower = parts
    hist, gap, lengths, mask = batch
    candidate = emb.lookup("item_id", torch.full((8,), 7))

    clean = tower(hist, gap, lengths, candidate)

    dirty_hist, dirty_gap = hist.clone(), gap.clone()
    dirty_hist[~mask] = N_ITEMS - 1
    dirty_gap[~mask] = 9
    dirty = tower(dirty_hist, dirty_gap, lengths, candidate)

    assert torch.allclose(clean, dirty, atol=1e-5)


def test_sequence_output_shape_is_independent_of_history_length(parts, batch):
    """M summaries out, whatever N went in — that is the O(M*N) contract."""
    emb, _, tower = parts
    hist, gap, lengths, _ = batch
    candidate = emb.lookup("item_id", torch.full((8,), 7))

    short = tower(hist[:, :2], gap[:, :2], lengths.clamp(max=2), candidate)
    full = tower(hist, gap, lengths, candidate)

    assert short.shape == full.shape == (8, 3, DIM)


@pytest.mark.parametrize("cross_attention", [False, True])
def test_wukong_shapes_and_gradients(cross_attention):
    torch.manual_seed(2)
    x = torch.randn(4, 7, DIM, requires_grad=True)
    model = Wukong(n_fields=7, dim=DIM, n_layers=2, n_fmb=8, n_lcb=8, rank=4,
                   cross_layer_attention=cross_attention)

    out = model(x)
    assert out.shape == (4, 16, DIM)
    assert model.output_dim == 16 * DIM

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_wukong_depth_is_the_vertical_scaling_knob():
    """Depth buys interaction order; the field axis is set by the widths."""
    for depth in (1, 3, 5):
        model = Wukong(n_fields=7, dim=DIM, n_layers=depth, n_fmb=8, n_lcb=8, rank=4)
        assert model(torch.randn(2, 7, DIM)).shape == (2, 16, DIM)
        assert sum(p.numel() for p in model.parameters()) > 0
