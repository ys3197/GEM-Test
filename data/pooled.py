"""
Pooling four domains into one dataset for the foundation model.

The design here follows a measurement rather than an assumption. Across the four
selected domains:

    users appearing in 2+ domains   15,135 / 346,190   =  4.37%
    items appearing in 2+ domains   0                      (categories partition ASINs)

So cross-domain transfer cannot happen the obvious way — the same entity being
seen in two places. 95.6% of users and every item are domain-exclusive, which
means a shared user/item embedding table would just be four disjoint tables
glued together. Ids are therefore **offset per domain**: pooling gives the
foundation model more data to fit its dense weights on, not a shared entity
space it does not have.

Two channels remain, and they are what the transfer experiments actually test:

- **Dense weights.** How price interacts with popularity, how quickly relevance
  decays with recency — these are learned once and apply everywhere.
- **Quantile bucket semantics.** `price_bucket=7` means "expensive for its
  category" in every domain, because the buckets were cut on within-domain
  quantiles. That makes them transferable in a way absolute dollar bins would
  not be, and it is why those four vocabularies are shared while store and
  category are offset like ids.

A `domain` field is added as a feature. GEM balances "learning from cross-surface
interactions while ensuring predictions remain tailored to each surface's
characteristics"; giving the model an explicit domain id is the smallest honest
version of that — it can share what generalises and condition on the surface
where it should not.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DOMAINS, PROCESSED_DIR  # noqa: E402
from data.dataset import (  # noqa: E402
    ITEM_FEATURE_COLUMNS,
    N_RESERVED,
    DomainData,
    load_domain,
)

# 分位数桶在各域之间语义一致（"本品类里第 k 分位"），因此共享词表。
# store / category 只是各域内部的 id，没有跨域语义，按 id 处理（加偏移）。
SHARED_FIELDS = {"price_bucket", "rating_bucket", "popularity_bucket"}
OFFSET_FIELDS = {"store", "category"}


@dataclass
class PooledData(DomainData):
    """A DomainData covering several domains, plus which domain each event is in."""

    domain_of_event: np.ndarray = None     # [n_events]  每个事件属于哪个域
    domain_of_user: np.ndarray = None      # [n_users]
    domain_names: list[str] = None
    n_domains: int = 0


def load_pooled(domains: list[str] | None = None) -> PooledData:
    """Concatenate per-domain data with ids offset so nothing collides."""
    names = domains or DOMAINS
    parts = [load_domain(d) for d in names]

    user_offset = item_offset = N_RESERVED
    field_offsets = {f: N_RESERVED for f in OFFSET_FIELDS}

    items_all, ts_all, hist_end_all = [], [], []
    offsets_all = [np.zeros(1, dtype=np.int64)]
    dom_event, dom_user = [], []
    feature_rows = [np.zeros((N_RESERVED, len(ITEM_FEATURE_COLUMNS)), dtype=np.int64)]
    shared_max = {f: N_RESERVED for f in SHARED_FIELDS}

    event_base = 0
    for d_idx, (name, part) in enumerate(zip(names, parts)):
        n_users = part.n_users
        n_items = part.n_items - N_RESERVED

        items_all.append(part.user_items - N_RESERVED + item_offset)
        ts_all.append(part.user_ts)
        hist_end_all.append(part.hist_end + event_base)

        counts = np.diff(part.user_offsets[: n_users + 1])
        offsets_all.append(np.cumsum(counts) + event_base)
        dom_event.append(np.full(len(part.user_items), d_idx, dtype=np.int64))
        dom_user.append(np.full(n_users, d_idx, dtype=np.int64))

        feats = part.item_features[N_RESERVED:].copy()
        for col_idx, col in enumerate(ITEM_FEATURE_COLUMNS):
            column = feats[:, col_idx]
            real = column >= N_RESERVED
            if col in OFFSET_FIELDS:
                column[real] += field_offsets[col] - N_RESERVED
                field_offsets[col] += part.vocab_sizes[col] - N_RESERVED
            else:
                shared_max[col] = max(shared_max[col], part.vocab_sizes[col])
            feats[:, col_idx] = column
        feature_rows.append(feats)

        user_offset += n_users
        item_offset += n_items
        event_base += len(part.user_items)

    vocab_sizes = {
        "user_id": user_offset,
        "item_id": item_offset,
        **{f: field_offsets[f] for f in OFFSET_FIELDS},
        **{f: shared_max[f] for f in SHARED_FIELDS},
    }

    return PooledData(
        domain="+".join(names),
        user_items=np.concatenate(items_all),
        user_offsets=np.concatenate(offsets_all),
        user_ts=np.concatenate(ts_all),
        hist_end=np.concatenate(hist_end_all),
        item_features=np.concatenate(feature_rows),
        vocab_sizes=vocab_sizes,
        n_users=user_offset - N_RESERVED,
        n_items=item_offset,
        domain_of_event=np.concatenate(dom_event),
        domain_of_user=np.concatenate(dom_user),
        domain_names=list(names),
        n_domains=len(names),
    )


def domain_positions(pooled: PooledData, positions: np.ndarray, domain: str) -> np.ndarray:
    """Keep only the positions belonging to one domain — how a VM gets its slice."""
    d_idx = pooled.domain_names.index(domain)
    return positions[pooled.domain_of_user[positions[:, 0]] == d_idx]


def summarise(pooled: PooledData) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(pooled.domain_names):
        mask = pooled.domain_of_event == i
        rows.append({
            "domain": name,
            "events": int(mask.sum()),
            "users": int((pooled.domain_of_user == i).sum()),
        })
    rows.append({
        "domain": "POOLED",
        "events": len(pooled.user_items),
        "users": pooled.n_users,
    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pooled = load_pooled()
    print(summarise(pooled).to_string(index=False))
    print("\nvocab sizes:")
    for k, v in pooled.vocab_sizes.items():
        shared = " (shared across domains)" if k in SHARED_FIELDS else ""
        print(f"  {k:20} {v:>9,}{shared}")


def user_negative_ranges(pooled: PooledData) -> np.ndarray:
    """
    The item-id range each user's negatives must be drawn from: `[n_users, 2)`.

    Without this, pooling silently makes the task *easier*. Uniform negatives over
    the pooled catalogue land outside the user's own domain 72-82% of the time, and
    a model only has to learn "is this item even in a category this user shops in"
    — which the category embedding answers immediately. Measured effect: AUC 0.932
    on solo data, 0.988 once pooled. Every transfer arm then sits at the ceiling and
    the comparison between them measures nothing.

    Because ids are offset per domain, each domain occupies one contiguous range, so
    the restriction is two integers per user rather than a per-user candidate set.
    """
    ranges = np.zeros((pooled.n_users + N_RESERVED, 2), dtype=np.int64)
    for d_idx in range(pooled.n_domains):
        mask = pooled.domain_of_event == d_idx
        lo = int(pooled.user_items[mask].min())
        hi = int(pooled.user_items[mask].max()) + 1
        ranges[np.flatnonzero(pooled.domain_of_user == d_idx)] = (lo, hi)
    return ranges
