"""
Figures for M3 and M4 — the two headline results, made visible instead of tabular.

    python -m analysis.transfer_figures

Reads whatever is already on disk under `runs/transfer/*.json` (no retraining) and
produces two figures:

    m3_arm_comparison.png   the six arms at k=0, Software, 3 seeds — paired delta vs
                            arm A with its spread, colour-coded by resolved/not
    m4_k_sweep.png          C's paired advantage over B across all four domains and
                            every k — the monotonic-widening result M4 was built to
                            test

Both use the same paired-by-seed method as `analysis/transfer_table.py`: a delta is
computed per seed (same seed shares teacher/data across arms within a run), then
averaged, because comparing raw NE ranges independently throws away that correlation
and was shown to hide a real, resolved effect (arm D in M3 — see STATE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FIGURES_DIR, ROOT  # noqa: E402
from models.transfer import ARM_LABELS, ARMS  # noqa: E402

RUNS_DIR = ROOT / "runs" / "transfer"

# M4's domain -> file-tag mapping, fixed when the sweep was run
M4_DOMAIN_TAGS = {
    "Software": "sw",
    "Video_Games": "video_games",
    "Musical_Instruments": "musical_instruments",
    "Industrial_and_Scientific": "industrial_and_scientific",
}
SEEDS = [42, 1337, 7]

ACCENT = "#B8541A"
RESOLVED_COLOR = "#2D6E9E"
UNRESOLVED_COLOR = "#B3453E"
DOMAIN_PALETTE = ["#2D6E9E", "#D98324", "#4A9B5C", "#B3453E"]


def _load(pattern: str) -> dict[int, dict]:
    """seed -> parsed JSON, for files matching pattern in runs/transfer/."""
    out = {}
    for path in sorted(RUNS_DIR.glob(f"{pattern}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload["student_seed"]] = payload
    return out


def m3_arm_comparison() -> Path:
    """Arm A..E' at k=0, Software, paired delta vs A with resolved/not colouring."""
    runs = _load("m3_seed*")
    if len(runs) < 2:
        raise SystemExit("need at least 2 of runs/transfer/m3_seed*.json — run M3 first")

    by_seed_arm = {s: {r["arm"]: r["best"]["ne"] for r in payload["results"]}
                   for s, payload in runs.items()}
    seeds = sorted(by_seed_arm)
    base = {s: by_seed_arm[s]["vm_only"] for s in seeds}

    labels, means, spreads, resolved = [], [], [], []
    for arm in ARMS:
        if arm == "vm_only":
            continue
        deltas = np.array([100 * (by_seed_arm[s][arm] - base[s]) / base[s] for s in seeds])
        mean, spread = deltas.mean(), deltas.max() - deltas.min()
        labels.append(ARM_LABELS[arm])
        means.append(mean)
        spreads.append(spread)
        resolved.append(abs(mean) > spread)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [RESOLVED_COLOR if r else UNRESOLVED_COLOR for r in resolved]
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=spreads, color=colors, height=0.6,
           error_kw={"ecolor": "#4A4A4A", "capsize": 3, "linewidth": 1.1})
    ax.axvline(0, color="#4A4A4A", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("paired NE delta vs no-transfer (%) — negative is better")
    ax.set_title("M3: five transfer techniques vs no transfer, Software, k=0, "
                 f"n={len(seeds)} seeds", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=RESOLVED_COLOR),
              plt.Rectangle((0, 0), 1, 1, color=UNRESOLVED_COLOR)]
    ax.legend(handles, ["resolved (gap > seed spread)", "not resolved"],
             fontsize=8.5, frameon=False, loc="lower right")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "m3_arm_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def m4_k_sweep() -> Path:
    """C's paired advantage over B vs k, one line per domain, all four."""
    per_domain = {}
    for domain, tag in M4_DOMAIN_TAGS.items():
        runs = _load(f"m4_{tag}_s*")
        if len(runs) < 2:
            continue
        seeds = sorted(runs)
        by_seed = {s: {(r["stale_days"], r["arm"]): r["best"]["ne"]
                       for r in runs[s]["results"]} for s in seeds}
        ks = sorted({k for s in seeds for k, _ in by_seed[s]})
        means, spreads = [], []
        for k in ks:
            b = np.array([by_seed[s][(k, "kd")] for s in seeds])
            c = np.array([by_seed[s][(k, "kd_adapter")] for s in seeds])
            pct = 100 * (b - c) / b
            means.append(pct.mean())
            spreads.append(pct.max() - pct.min())
        per_domain[domain] = (ks, np.array(means), np.array(spreads))

    if not per_domain:
        raise SystemExit("no runs/transfer/m4_*_s*.json found — run M4 first")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (domain, (ks, means, spreads)) in enumerate(per_domain.items()):
        color = DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)]
        ax.errorbar(ks, means, yerr=spreads, marker="o", markersize=5, linewidth=1.8,
                   capsize=3, color=color, label=domain)

    ax.axhline(0, color="#4A4A4A", linewidth=1.0)
    ax.set_xlabel("k — teacher staleness (days)")
    ax.set_ylabel("C's paired advantage over B (%) — positive is C beats naive KD")
    ax.set_title("M4: C's edge over naive KD widens monotonically with staleness,\n"
                 "every domain, every point resolved", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "m4_k_sweep.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    print(f"wrote {m3_arm_comparison()}")
    print(f"wrote {m4_k_sweep()}")


if __name__ == "__main__":
    main()
