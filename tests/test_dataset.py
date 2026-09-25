"""
Guardrails for the sample builder.

The causality tests are the load-bearing ones. A leak here does not crash and
does not look like a bug — it looks like unusually good results, and every
number downstream becomes meaningless. Both failure modes below were real:

- timestamps arrived in milliseconds while split boundaries were nanoseconds,
  so a date-based split silently put everything in train;
- events sharing the candidate's exact timestamp were being fed as history,
  affecting 1.8% of samples.

These run on real prepared data and are skipped when it is absent, so the suite
stays usable on a machine that has not downloaded 8 GB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MAX_SEQ_LEN, PROCESSED_DIR  # noqa: E402
from data.dataset import (  # noqa: E402
    InteractionDataset,
    collate,
    load_domain,
    temporal_split,
)

DOMAIN = "Software"

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / DOMAIN / "interactions.parquet").exists(),
    reason="prepared data absent — run `python -m data.prepare`",
)


@pytest.fixture(scope="module")
def data():
    return load_domain(DOMAIN)


@pytest.fixture(scope="module")
def splits(data):
    return temporal_split(data, pd.Timestamp("2021-01-01"), pd.Timestamp("2022-06-01"))


def test_timestamps_are_nanoseconds(data):
    """Millisecond timestamps compared against nanosecond boundaries split wrongly."""
    as_dates = pd.to_datetime(data.user_ts)
    assert as_dates.min() > pd.Timestamp("1995-01-01")
    assert as_dates.max() < pd.Timestamp("2030-01-01")


def test_history_is_strictly_before_candidate(data, splits):
    """No history event may occur at or after the event being predicted."""
    rng = np.random.default_rng(0)
    sample = splits["train"][rng.integers(len(splits["train"]), size=5000)]

    for user_idx, pos in sample:
        start = data.user_offsets[user_idx]
        end = data.hist_end[start + pos]
        hist_ts = data.user_ts[max(start, end - MAX_SEQ_LEN):end]
        assert len(hist_ts) > 0, "split kept an event with empty history"
        assert hist_ts.max() < data.user_ts[start + pos]


def test_history_stays_within_the_same_user(data, splits):
    """CSR slicing must never run past a user's own range."""
    rng = np.random.default_rng(1)
    sample = splits["train"][rng.integers(len(splits["train"]), size=2000)]

    for user_idx, pos in sample:
        start, stop = data.user_offsets[user_idx], data.user_offsets[user_idx + 1]
        end = data.hist_end[start + pos]
        assert start <= max(start, end - MAX_SEQ_LEN) <= end <= stop


def test_splits_are_ordered_in_time_and_disjoint(data, splits):
    """Train must end before valid begins, and valid before test."""
    def ts_of(positions):
        idx = data.user_offsets[positions[:, 0]] + positions[:, 1]
        return data.user_ts[idx]

    train, valid, test = (ts_of(splits[k]) for k in ("train", "valid", "test"))
    assert train.max() < valid.min()
    assert valid.max() < test.min()


def test_split_boundaries_are_honoured(data):
    """A date boundary must actually land on that date, not 1000x off."""
    boundary = pd.Timestamp("2021-01-01")
    splits = temporal_split(data, boundary)
    idx = data.user_offsets[splits["train"][:, 0]] + splits["train"][:, 1]
    assert pd.to_datetime(data.user_ts[idx]).max() < boundary


def test_negatives_are_not_in_the_user_history(data, splits):
    ds = InteractionDataset(data, splits["train"][:2000], n_negatives=4, seed=7)
    for i in range(0, 4000, 5):           # 每组的第 0 个是正样本，跳过
        for slot in range(1, 5):
            s = ds[i + slot]
            assert s["label"] == 0.0
            assert s["item"] not in set(s["hist"].tolist())


def test_collate_pads_to_batch_max_not_fixed_length(data, splits):
    """
    Padding to the batch's own longest sequence is the baseline M0 established;
    padding to MAX_SEQ_LEN would waste 74% of positions on this data.
    """
    ds = InteractionDataset(data, splits["train"][:500], n_negatives=1, seed=3)
    batch = collate([ds[i] for i in range(128)])

    assert batch["hist"].shape[1] == int(batch["hist_len"].max())
    assert batch["hist"].shape[1] <= MAX_SEQ_LEN
    # 补出来的位置必须是 PAD，否则会被当成真实商品喂进模型
    for row, length in zip(batch["hist"], batch["hist_len"]):
        assert (row[length:] == 0).all()


def test_positive_rate_matches_negative_sampling_ratio(data, splits):
    ds = InteractionDataset(data, splits["train"][:1000], n_negatives=4, seed=5)
    batch = collate([ds[i] for i in range(1000)])
    assert batch["label"].mean() == pytest.approx(0.2, abs=0.02)


# ── 池化数据（M3 的地基）──────────────────────────────────────

@pytest.fixture(scope="module")
def pooled():
    from data.pooled import load_pooled
    from config import DOMAINS
    have = [d for d in DOMAINS if (PROCESSED_DIR / d / "vocab.json").exists()]
    if len(have) < 2:
        pytest.skip("need at least two prepared domains")
    return load_pooled(have)


def test_pooled_ids_do_not_collide_across_domains(pooled):
    """
    Items are offset per domain because the measured overlap is zero: Amazon
    categories partition ASINs. If ranges overlapped, two unrelated products
    would share an embedding.
    """
    ranges = []
    for i in range(pooled.n_domains):
        mask = pooled.domain_of_event == i
        ranges.append((pooled.user_items[mask].min(), pooled.user_items[mask].max()))
    for (_, hi), (lo, _) in zip(ranges, ranges[1:]):
        assert hi < lo


def test_every_user_belongs_to_one_domain(pooled):
    rng = np.random.default_rng(0)
    for u in rng.integers(0, pooled.n_users, 2000):
        start, stop = pooled.user_offsets[u], pooled.user_offsets[u + 1]
        if stop > start:
            assert len(np.unique(pooled.domain_of_event[start:stop])) == 1


def test_pooling_preserves_each_domains_sample_count(pooled):
    """Pooling must add data, never silently drop or duplicate any."""
    from data.pooled import domain_positions

    boundary = pd.Timestamp("2021-01-01")
    pooled_splits = temporal_split(pooled, boundary)

    for name in pooled.domain_names:
        solo = temporal_split(load_domain(name), boundary)
        assert len(domain_positions(pooled, pooled_splits["train"], name)) == \
               len(solo["train"])


def test_causality_survives_pooling(pooled):
    splits = temporal_split(pooled, pd.Timestamp("2021-01-01"))
    rng = np.random.default_rng(3)
    for user_idx, pos in splits["train"][rng.integers(len(splits["train"]), size=3000)]:
        start = pooled.user_offsets[user_idx]
        end = pooled.hist_end[start + pos]
        assert end > start
        assert pooled.user_ts[max(start, end - MAX_SEQ_LEN):end].max() < \
               pooled.user_ts[start + pos]
