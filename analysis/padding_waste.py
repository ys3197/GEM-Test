"""
M0 result: how much compute does padding actually waste on this data?

Meta's sequence-learning post claims padding jagged user sequences to a fixed
maximum "would waste up to 50% of compute." That is a claim about *their* data,
whose sequences run "from hundreds to tens of thousands of tokens per sample."
This script measures the same quantity on ours, because the answer decides
whether jagged kernels are worth any complexity here — and because the shape of
our length distribution turns out to be nothing like theirs.

Two padding strategies, and the gap between them is the interesting part:

    fixed   — pad every sequence to `max_len`. The naive baseline, and the one
              Meta's 50% figure refers to.
    batch   — pad to the longest sequence *in each batch*. What any competent
              dataloader already does for free, so this is the real baseline
              that jagged-tensor work has to beat.

Waste is the fraction of padded positions carrying no signal:

    waste = 1 - (real tokens) / (padded tokens)

Usage:
    python -m analysis.padding_waste
    python -m analysis.padding_waste --domains Software
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DOMAINS, FIGURES_DIR, MAX_SEQ_LEN, PROCESSED_DIR, RANDOM_SEED  # noqa: E402

MAX_LENS = [8, 16, 32, 64, 128, 256]
BATCH_SIZES = [16, 32, 64, 128, 256, 512]


def raw_lengths(domain: str) -> np.ndarray:
    """Per-user event counts, uncapped — the cap is a parameter we sweep."""
    path = PROCESSED_DIR / domain / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m data.prepare` first")
    df = pd.read_parquet(path, columns=["user_id"])
    return df.groupby("user_id").size().to_numpy()


def waste_fixed(lens: np.ndarray, max_len: int) -> float:
    capped = np.minimum(lens, max_len)
    return 1.0 - capped.sum() / (len(capped) * max_len)


def waste_batched(lens: np.ndarray, max_len: int, batch_size: int,
                  rng: np.random.Generator) -> float:
    capped = np.minimum(lens, max_len).copy()
    rng.shuffle(capped)
    n_full = len(capped) // batch_size
    if n_full == 0:
        return float("nan")
    batches = capped[: n_full * batch_size].reshape(n_full, batch_size)
    return 1.0 - batches.sum() / (batches.max(axis=1).sum() * batch_size)


def analyse(domain: str, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    lens = raw_lengths(domain)
    row = {
        "domain": domain,
        "users": len(lens),
        "mean_len": lens.mean(),
        "median_len": float(np.median(lens)),
        "p95_len": float(np.percentile(lens, 95)),
        "p99_len": float(np.percentile(lens, 99)),
        "max_len_observed": int(lens.max()),
    }
    for m in MAX_LENS:
        row[f"covered_at_{m}"] = 100.0 * (lens <= m).mean()
        row[f"fixed_{m}"] = waste_fixed(lens, m)
        row[f"batch32_{m}"] = waste_batched(lens, m, 32, rng)
    for b in BATCH_SIZES:
        row[f"bs_{b}"] = waste_batched(lens, MAX_SEQ_LEN, b, rng)
    return row, lens


def plot(results: pd.DataFrame, lengths: dict[str, np.ndarray]) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))
    ax1, ax2, ax3 = axes
    accent = "#B8541A"

    # 三张图共用一套按域固定的配色，否则图例和线对不上
    palette = ["#2D6E9E", "#D98324", "#4A9B5C", "#B3453E"]
    colors = {d: palette[i % len(palette)] for i, d in enumerate(lengths)}

    # ── 1. 长度的累积分布：一个上限到底覆盖了多少用户 ────────────────
    for d, lens in lengths.items():
        xs = np.sort(lens)
        ys = np.arange(1, len(xs) + 1) / len(xs) * 100
        ax1.plot(xs, ys, linewidth=1.7, label=d, color=colors[d])
    for m in (16, 32, 64):
        ax1.axvline(m, color="#8A94A0", linestyle=":", linewidth=1.0)
        ax1.text(m, 22, f" {m}", color="#5A646E", fontsize=8.5)
    ax1.axvline(MAX_SEQ_LEN, color=accent, linestyle="--", linewidth=1.3)
    ax1.set_xscale("log", base=2)
    ax1.set_xlim(1, 512)
    ax1.set_ylim(0, 101)
    ax1.set_xlabel("sequence length (events per user)")
    ax1.set_ylabel("% of users at or below")
    ax1.set_title("Almost everyone is short", fontsize=11, loc="left")
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    # ── 2. 浪费 vs 上限：固定补齐 vs 按 batch 补齐 ──────────────────
    for _, r in results.iterrows():
        c = colors[r["domain"]]
        ax2.plot(MAX_LENS, [r[f"fixed_{m}"] * 100 for m in MAX_LENS],
                 marker="o", markersize=4, linewidth=1.6, color=c)
        ax2.plot(MAX_LENS, [r[f"batch32_{m}"] * 100 for m in MAX_LENS],
                 marker="s", markersize=3.5, linewidth=1.3, linestyle="--",
                 alpha=0.75, color=c)
    ax2.axhline(50, color=accent, linewidth=1.1)
    ax2.text(150, 43, "Meta's “up to 50%”", color=accent, fontsize=8.5,
             ha="right")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(MAX_LENS)
    ax2.set_xticklabels(MAX_LENS)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("max_len (cap)")
    ax2.set_ylabel("wasted compute (%)")
    ax2.set_title("solid = pad to fixed   ·   dashed = pad to batch max",
                  fontsize=11, loc="left")

    # ── 3. 反直觉的那条：batch 越大，浪费越多 ──────────────────────
    for _, r in results.iterrows():
        ax3.plot(BATCH_SIZES, [r[f"bs_{b}"] * 100 for b in BATCH_SIZES],
                 marker="o", markersize=4, linewidth=1.6, label=r["domain"],
                 color=colors[r["domain"]])
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(BATCH_SIZES)
    ax3.set_xticklabels(BATCH_SIZES)
    ax3.set_xlabel("batch size")
    ax3.set_ylabel("wasted compute (%)")
    ax3.set_title(f"Bigger batches waste more (cap={MAX_SEQ_LEN})",
                  fontsize=11, loc="left")
    ax3.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.22, linewidth=0.6)

    fig.tight_layout()
    out = FIGURES_DIR / "m0_padding_waste.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+")
    args = ap.parse_args()

    available = [d for d in (args.domains or DOMAINS)
                 if (PROCESSED_DIR / d / "interactions.parquet").exists()]
    if not available:
        raise SystemExit("No prepared domains — run `python -m data.prepare` first")

    rng = np.random.default_rng(RANDOM_SEED)
    rows, lengths = [], {}
    for d in available:
        row, lens = analyse(d, rng)
        rows.append(row)
        lengths[d] = lens
    results = pd.DataFrame(rows)

    print("\n=== sequence lengths ===")
    print(results[["domain", "users", "mean_len", "median_len", "p95_len",
                   "p99_len", "max_len_observed"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print(f"\n=== coverage and waste by cap ===")
    for m in MAX_LENS:
        sub = results[["domain", f"covered_at_{m}", f"fixed_{m}", f"batch32_{m}"]]
        vals = sub.iloc[:, 1:].mean()
        print(f"  cap={m:4}  covers {vals.iloc[0]:5.1f}% of users  |  "
              f"fixed waste {vals.iloc[1] * 100:5.1f}%  |  batch-32 waste {vals.iloc[2] * 100:5.1f}%")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(FIGURES_DIR / "m0_padding_waste.csv", index=False)
    out = plot(results, lengths)
    print(f"\nWrote {out}")
    print(f"Wrote {FIGURES_DIR / 'm0_padding_waste.csv'}")


if __name__ == "__main__":
    main()
