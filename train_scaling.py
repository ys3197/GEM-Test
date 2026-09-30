"""
M5 — does quality scale log-linearly with compute?

    python train_scaling.py
    python train_scaling.py --factors 0.25 0.5 1 2 4 --seeds 42 1337 7

Five model sizes via `MiniGEMConfig.scaled(factor)` — the same knob M3/M4's arm D
depends on staying compatible with, which is why this sweep and the transfer arms
cannot share a config. Depth is fixed; every width moves together, so the sizes on
the curve differ only in scale, not in shape.

Each size is trained on the **same fixed dataset** (`Software`, solo, not pooled —
this is about capacity, not the transfer arms' domain-pooling machinery) with the
same schedule; only the model grows. That isolates one question — does more capacity
help on this much data — rather than mixing it with a compute-optimal data/size
co-scaling question this project has no budget to answer.

Compute is **measured**, not estimated, via `train.count_flops` (`FlopCounterMode`
tracing an actual forward+backward pass) — a hand-derived FLOPs formula would have to
track every matmul across the FM stacks, attention pooling, and head separately, and
silently get one wrong. Total training FLOPs = FLOPs/example x examples actually
seen, which already accounts for early stopping cutting some sizes off sooner.

Every arm elsewhere in this project turned out to need >=3 seeds before a result
could be trusted (A's own NE moved ~8% on seed alone in M3). This sweep is not
exempt: each size is trained at 3 seeds, and the fit is read against that spread,
not against single points.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import PROCESSED_DIR, ROOT
from models.gem import MiniGEMConfig
from train import train

RUNS_DIR = ROOT / "runs" / "scaling"


def run_one(factor: float, seed: int, args) -> dict:
    ns = argparse.Namespace(**vars(args))
    ns.scale = factor
    ns.seed = seed
    ns.flops = True
    ns.tag = f"m5_f{factor}_s{seed}".replace(".", "p")
    ns.variant = "pooled"
    ns.dim, ns.layers, ns.queries = MiniGEMConfig().dim, MiniGEMConfig().n_layers, MiniGEMConfig().n_queries

    print(f"\n{'=' * 70}\nfactor={factor}  seed={seed}\n{'=' * 70}")
    result = train(ns)
    (RUNS_DIR).mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{ns.tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarise(results: list[dict]) -> None:
    by_factor: dict[float, list[dict]] = {}
    for r in results:
        by_factor.setdefault(r["_factor"], []).append(r)

    print(f"\n{'factor':>8}  {'dense params':>13}  {'FLOPs/example':>14}  "
          f"{'eval NE (mean)':>15}  {'spread':>8}  {'n':>2}")
    print("-" * 70)
    points = []
    for factor in sorted(by_factor):
        rs = by_factor[factor]
        ne = np.array([r["eval"]["ne"] for r in rs])
        flops = rs[0]["flops_per_example"]
        params = rs[0]["dense_params"]
        mean, spread = ne.mean(), (ne.max() - ne.min() if len(ne) > 1 else float("nan"))
        print(f"{factor:>8}  {params:>13,}  {flops:>14,.0f}  {mean:>15.4f}  "
              f"{spread:>8.4f}  {len(rs):>2}")
        points.append((flops, mean, spread))

    log_flops = np.log10([p[0] for p in points])
    log_ne = np.log10([p[1] for p in points])
    slope, intercept = np.polyfit(log_flops, log_ne, 1)
    pred = slope * log_flops + intercept
    ss_res = np.sum((log_ne - pred) ** 2)
    ss_tot = np.sum((log_ne - log_ne.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    spreads = [p[2] for p in points if not np.isnan(p[2])]
    noise_floor = max(spreads) if spreads else 0.0
    ne_range = max(p[1] for p in points) - min(p[1] for p in points)

    print(f"\nlog10(NE) = {slope:+.4f} * log10(FLOPs) + {intercept:+.4f}   R^2 = {r2:.4f}")
    print(f"NE range across sizes: {ne_range:.4f}   largest single-size seed spread: "
          f"{noise_floor:.4f}")
    if ne_range < noise_floor:
        print("-> the NE range across 5 sizes is SMALLER than one size's own seed "
              "noise. Not resolved: this sweep cannot distinguish a scaling trend "
              "from noise at this data/compute budget.")
    elif r2 > 0.8 and slope < 0:
        print("-> a clean negative log-log trend, well above the single-size noise "
              "floor: consistent with log-linear scaling at this range.")
    else:
        print("-> the trend is not clean (low R^2, wrong-sign slope, or a floor "
              "effect) even though it clears the noise floor — report as such, not "
              "as confirmed log-linearity.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factors", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 7])
    ap.add_argument("--domain", default="Software")

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--negatives", type=int, default=4)

    ap.add_argument("--train-end", default="2021-01-01")
    ap.add_argument("--valid-end", default="2022-06-01")
    ap.add_argument("--max-train", type=int, default=300_000,
                    help="fixed across every size on purpose — this sweep isolates "
                         "capacity, not a compute-optimal data/size co-scaling")
    ap.add_argument("--max-valid", type=int, default=50_000)
    ap.add_argument("--max-eval", type=int, default=50_000)

    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1_000_000)
    ap.add_argument("--no-early-stop", action="store_true",
                    help="disable early stopping (on by default — bigger sizes "
                         "overfitting fast on a fixed data budget shouldn't burn "
                         "the rest of their epoch budget)")
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    args.early_stop = not args.no_early_stop
    if not (PROCESSED_DIR / args.domain / "interactions.parquet").exists():
        raise SystemExit(f"{args.domain} not prepared — run `python -m data.prepare`")

    results = []
    for factor in args.factors:
        for seed in args.seeds:
            r = run_one(factor, seed, args)
            r["_factor"] = factor
            r["_seed"] = seed
            results.append(r)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "m5_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    summarise(results)
    print(f"\nwrote runs/scaling/m5_summary.json ({len(results)} runs)")


if __name__ == "__main__":
    main()
