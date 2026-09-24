"""
Turn raw review jsonl into per-domain interaction tables plus item features.

Three things this stage is responsible for, in order of how much they matter:

1. **Keeping the calendar.** Timestamps survive intact (converted from the
   milliseconds the raw files use). Every later experiment — especially the
   staleness sweep — slices on real dates, so a pipeline that shuffles rows or
   drops time would quietly make the central experiment impossible.

2. **k-core filtering.** Iteratively drop users and items with too few
   interactions. The point is not hygiene; it is that a user with two events has
   no sequence to learn from.

3. **Streaming.** The largest configured domain is ~900 MB of jsonl. Files are
   read line by line, never loaded whole.

Usage:
    python -m data.prepare                  # all configured domains
    python -m data.prepare --smoke
    python -m data.prepare --domains Digital_Music
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DOMAINS,
    DOMAINS_SMOKE,
    MIN_ITEM_INTERACTIONS,
    MIN_USER_INTERACTIONS,
    PROCESSED_DIR,
    RAW_DIR,
)

REVIEW_FIELDS = ("user_id", "parent_asin", "rating", "timestamp", "verified_purchase")
META_FIELDS = ("parent_asin", "main_category", "title", "average_rating",
               "rating_number", "price", "store")


def _review_path(domain: str) -> Path:
    return RAW_DIR / "raw" / "review_categories" / f"{domain}.jsonl"


def _meta_path(domain: str) -> Path:
    return RAW_DIR / "raw" / "meta_categories" / f"meta_{domain}.jsonl"


def read_reviews(domain: str) -> pd.DataFrame:
    """Stream the review jsonl into a DataFrame of just the fields we use."""
    path = _review_path(domain)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m data.download` first")

    rows, skipped = [], 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows.append((
                    r["user_id"],
                    r["parent_asin"],
                    float(r["rating"]),
                    int(r["timestamp"]),
                    bool(r.get("verified_purchase", False)),
                ))
            except (KeyError, ValueError, json.JSONDecodeError):
                skipped += 1

    if skipped:
        print(f"    skipped {skipped} malformed lines")

    df = pd.DataFrame(rows, columns=list(REVIEW_FIELDS))
    # 原始时间戳是**毫秒**。所有下游的日期切分都依赖这一步换算正确。
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop(columns=["timestamp"])


def k_core_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """
    Iteratively drop sparse users and items until the set stops shrinking.

    One pass is not enough: removing a cold item can push a user below the
    threshold, which can in turn orphan more items.
    """
    before = len(df)
    for it in range(20):
        n = len(df)
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= min_user].index)]
        ic = df["parent_asin"].value_counts()
        df = df[df["parent_asin"].isin(ic[ic >= min_item].index)]
        if len(df) == n:
            print(f"    k-core converged after {it + 1} pass(es): "
                  f"{before:,} -> {len(df):,} rows")
            break
    return df


def read_meta(domain: str) -> pd.DataFrame:
    """Item-side features. Missing prices/stores are common and kept as NaN."""
    path = _meta_path(domain)
    if not path.exists():
        print(f"    no metadata file for {domain}, skipping item features")
        return pd.DataFrame(columns=list(META_FIELDS))

    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            price = r.get("price")
            try:
                price = float(price) if price not in (None, "", "None") else None
            except (TypeError, ValueError):
                price = None
            rows.append((
                r.get("parent_asin"),
                r.get("main_category"),
                (r.get("title") or "")[:200],
                r.get("average_rating"),
                r.get("rating_number"),
                price,
                r.get("store"),
            ))
    return pd.DataFrame(rows, columns=list(META_FIELDS)).drop_duplicates("parent_asin")


def prepare_domain(domain: str) -> dict:
    print(f"  {domain}")
    df = read_reviews(domain)
    print(f"    raw: {len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, {df['parent_asin'].nunique():,} items")

    df = k_core_filter(df, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
    if df.empty:
        print("    nothing left after k-core — domain too sparse at these thresholds")
        return {}

    df = df.sort_values(["user_id", "ts"]).reset_index(drop=True)
    df["domain"] = domain

    items = read_meta(domain)
    items = items[items["parent_asin"].isin(df["parent_asin"].unique())]

    out = PROCESSED_DIR / domain
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "interactions.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)

    stats = {
        "domain": domain,
        "interactions": len(df),
        "users": df["user_id"].nunique(),
        "items": df["parent_asin"].nunique(),
        "first_ts": df["ts"].min(),
        "last_ts": df["ts"].max(),
        "span_days": (df["ts"].max() - df["ts"].min()).days,
        "median_seq_len": int(df.groupby("user_id").size().median()),
        "items_with_meta": len(items),
    }
    print(f"    kept: {stats['interactions']:,} interactions | "
          f"{stats['users']:,} users | {stats['items']:,} items | "
          f"span {stats['span_days']:,} days")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--domains", nargs="+")
    args = ap.parse_args()

    domains = args.domains or (DOMAINS_SMOKE if args.smoke else DOMAINS)
    print(f"Preparing {len(domains)} domain(s)\n")

    stats = [s for d in domains if (s := prepare_domain(d))]
    if not stats:
        return

    summary = pd.DataFrame(stats)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROCESSED_DIR / "summary.csv", index=False)

    print("\n" + summary.to_string(index=False))
    print(f"\nWrote {PROCESSED_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
