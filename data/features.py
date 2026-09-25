"""
Encode item metadata into integer ids the model can embed.

Everything the model consumes is categorical: continuous fields (price, average
rating, review count) are bucketed rather than fed as floats. Two reasons, both
of which come from how the non-sequence tower works:

- Wukong-style factorization machines learn *interactions between embeddings*.
  A raw float has no embedding to interact with.
- Price and popularity are heavily skewed and full of missing values. Bucketing
  on quantiles makes the model robust to both, and gives a natural home for
  "unknown" as its own bucket rather than an imputed number pretending to be a
  measurement.

Reserved ids, consistent across every vocabulary:

    0  PAD      padding position in a sequence
    1  UNK      value not seen when the vocabulary was built
    2  MISSING  value absent in the source data

Usage:
    python -m data.features
    python -m data.features --domains Software
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DOMAINS, PROCESSED_DIR  # noqa: E402

PAD, UNK, MISSING = 0, 1, 2
N_RESERVED = 3

# store 的取值是长尾的：保留高频的，其余归入 UNK。
# 上限定在这里而不是设为无穷，是因为只出现一两次的 store 学不出有意义的 embedding，
# 只会让表变大、梯度变稀疏。
MAX_STORE_VOCAB = 5_000

PRICE_BUCKETS = 16
RATING_BUCKETS = 10
POPULARITY_BUCKETS = 16


def _quantile_bucketer(values: pd.Series, n_buckets: int) -> np.ndarray:
    """
    Bucket edges from quantiles of the observed (non-missing) values.

    Quantiles rather than equal width: price spans four orders of magnitude and
    equal-width bins would put almost everything in bucket 0.
    """
    clean = values.dropna()
    if clean.empty:
        return np.array([])
    qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
    return np.unique(np.quantile(clean, qs))


def _apply_buckets(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Map values to bucket ids; NaN becomes MISSING."""
    out = np.full(len(values), MISSING, dtype=np.int64)
    mask = values.notna().to_numpy()
    if mask.any() and len(edges) > 0:
        out[mask] = np.searchsorted(edges, values[mask].to_numpy()) + N_RESERVED
    elif mask.any():
        out[mask] = N_RESERVED
    return out


def build_vocab(values: pd.Series, max_size: int | None = None) -> dict[str, int]:
    """Frequency-ordered vocabulary. Ids start after the reserved block."""
    counts = values.dropna().astype(str).value_counts()
    if max_size is not None:
        counts = counts.head(max_size)
    return {v: i + N_RESERVED for i, v in enumerate(counts.index)}


def encode_domain(domain: str) -> dict:
    src = PROCESSED_DIR / domain
    items = pd.read_parquet(src / "items.parquet")
    inter = pd.read_parquet(src / "interactions.parquet",
                            columns=["user_id", "parent_asin"])

    # item / user 的 id 词表由**交互表**决定，而不是元数据表：
    # 只有出现在交互里的实体才需要 embedding。
    item_vocab = {a: i + N_RESERVED
                  for i, a in enumerate(sorted(inter["parent_asin"].unique()))}
    user_vocab = {u: i + N_RESERVED
                  for i, u in enumerate(sorted(inter["user_id"].unique()))}

    items = items[items["parent_asin"].isin(item_vocab)].copy()

    store_vocab = build_vocab(items["store"], MAX_STORE_VOCAB)
    category_vocab = build_vocab(items["main_category"])

    price_edges = _quantile_bucketer(items["price"], PRICE_BUCKETS)
    rating_edges = _quantile_bucketer(items["average_rating"], RATING_BUCKETS)
    # 评论数跨几个数量级，先取 log 再分桶
    log_pop = np.log1p(items["rating_number"].astype("float64"))
    pop_edges = _quantile_bucketer(log_pop, POPULARITY_BUCKETS)

    enc = pd.DataFrame({
        "item_id": items["parent_asin"].map(item_vocab).astype("int64"),
        "store": items["store"].astype(str).map(store_vocab).fillna(UNK).astype("int64"),
        "category": items["main_category"].astype(str).map(category_vocab)
                    .fillna(UNK).astype("int64"),
        "price_bucket": _apply_buckets(items["price"], price_edges),
        "rating_bucket": _apply_buckets(items["average_rating"], rating_edges),
        "popularity_bucket": _apply_buckets(log_pop, pop_edges),
    })

    # 交互表里存在、但元数据缺失的商品：补一行全 MISSING，
    # 这样下游可以按 item_id 直接索引，不必处理缺行。
    missing_ids = sorted(set(item_vocab.values()) - set(enc["item_id"]))
    if missing_ids:
        pad_rows = pd.DataFrame({
            "item_id": missing_ids,
            **{c: MISSING for c in enc.columns if c != "item_id"},
        })
        enc = pd.concat([enc, pad_rows], ignore_index=True)

    enc = enc.sort_values("item_id").reset_index(drop=True)

    vocab_sizes = {
        "item_id": len(item_vocab) + N_RESERVED,
        "user_id": len(user_vocab) + N_RESERVED,
        "store": len(store_vocab) + N_RESERVED,
        "category": len(category_vocab) + N_RESERVED,
        "price_bucket": len(price_edges) + 1 + N_RESERVED,
        "rating_bucket": len(rating_edges) + 1 + N_RESERVED,
        "popularity_bucket": len(pop_edges) + 1 + N_RESERVED,
    }

    enc.to_parquet(src / "item_features.parquet", index=False)
    (src / "vocab.json").write_text(
        json.dumps({
            "vocab_sizes": vocab_sizes,
            "item_vocab": item_vocab,
            "user_vocab": user_vocab,
            "reserved": {"PAD": PAD, "UNK": UNK, "MISSING": MISSING},
        }),
        encoding="utf-8",
    )

    coverage = {c: 100.0 * (enc[c] > MISSING).mean()
                for c in enc.columns if c != "item_id"}
    return {"domain": domain, "vocab_sizes": vocab_sizes, "coverage": coverage}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+")
    args = ap.parse_args()

    domains = [d for d in (args.domains or DOMAINS)
               if (PROCESSED_DIR / d / "items.parquet").exists()]
    if not domains:
        raise SystemExit("No prepared domains — run `python -m data.prepare` first")

    for d in domains:
        res = encode_domain(d)
        sizes = res["vocab_sizes"]
        print(f"  {d}")
        print("    vocab  " + "  ".join(f"{k}={v:,}" for k, v in sizes.items()))
        print("    filled " + "  ".join(f"{k}={v:.0f}%" for k, v in res["coverage"].items()))

    print(f"\nWrote item_features.parquet + vocab.json under {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
