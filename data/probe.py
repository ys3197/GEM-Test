"""
Check whether a domain has sequences at all, before building anything on it.

Amazon Reviews is sparse in a way that file size does not reveal: a category can
be hundreds of megabytes and still have almost no user with more than one
interaction. Sequence models need users with history, so the question that
decides domain selection is not "how big is the file" but "how many users
survive k-core".

This probe answers that without committing to a threshold: it reports the
survival curve across several k values, so the choice is made on evidence.

Usage:
    python -m data.probe                        # every downloaded domain
    python -m data.probe --domains All_Beauty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RAW_DIR  # noqa: E402

K_VALUES = [2, 3, 5, 8, 10]


def load_pairs(domain: str) -> pd.DataFrame:
    """Only user/item/ts — enough for density, cheap to read."""
    path = RAW_DIR / "raw" / "review_categories" / f"{domain}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — download this domain first")

    users, items, ts = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                users.append(r["user_id"])
                items.append(r["parent_asin"])
                ts.append(int(r["timestamp"]))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return pd.DataFrame({"user_id": users, "parent_asin": items, "ts": ts})


def k_core_survivors(df: pd.DataFrame, k: int, max_passes: int = 20) -> tuple[int, int]:
    """Rows and users remaining after iterative k-core at threshold k."""
    cur = df
    for _ in range(max_passes):
        n = len(cur)
        uc = cur["user_id"].value_counts()
        cur = cur[cur["user_id"].isin(uc[uc >= k].index)]
        ic = cur["parent_asin"].value_counts()
        cur = cur[cur["parent_asin"].isin(ic[ic >= k].index)]
        if len(cur) == n:
            break
    return len(cur), cur["user_id"].nunique()


def probe(domain: str) -> dict:
    df = load_pairs(domain)
    per_user = df.groupby("user_id").size()
    span = (pd.to_datetime(df["ts"].max(), unit="ms")
            - pd.to_datetime(df["ts"].min(), unit="ms")).days

    row = {
        "domain": domain,
        "rows": len(df),
        "users": df["user_id"].nunique(),
        "items": df["parent_asin"].nunique(),
        "rows_per_user": len(df) / max(df["user_id"].nunique(), 1),
        "users_ge_5": int((per_user >= 5).sum()),
        "pct_users_ge_5": 100.0 * (per_user >= 5).mean(),
        "span_days": span,
    }
    for k in K_VALUES:
        rows_k, users_k = k_core_survivors(df, k)
        row[f"k{k}_rows"] = rows_k
        row[f"k{k}_users"] = users_k
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+")
    args = ap.parse_args()

    review_dir = RAW_DIR / "raw" / "review_categories"
    domains = args.domains or sorted(p.stem for p in review_dir.glob("*.jsonl"))
    if not domains:
        raise SystemExit(f"No downloaded domains in {review_dir}")

    rows = []
    for d in domains:
        print(f"  probing {d} ...", flush=True)
        rows.append(probe(d))

    res = pd.DataFrame(rows)

    print("\n=== density ===")
    print(res[["domain", "rows", "users", "items", "rows_per_user",
               "pct_users_ge_5", "span_days"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\n=== users surviving k-core ===")
    cols = ["domain"] + [f"k{k}_users" for k in K_VALUES]
    print(res[cols].to_string(index=False))

    print("\nA domain is usable for sequence modelling only if k5_users is large "
          "enough to train on — file size says nothing about this.")


if __name__ == "__main__":
    main()
