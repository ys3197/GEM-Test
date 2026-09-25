"""
Aggregate transfer runs across seeds.

    python -m analysis.transfer_table
    python -m analysis.transfer_table --pattern "m4_*"

A single run cannot answer whether one arm beats another here. Measured on
`Software` at `k = 0`, holding the teacher and everything else fixed and changing
only the student's initialisation seed:

    seed 42     A  NE 0.2288      C  NE 0.2356    (+2.99%)
    seed 1337   A  NE 0.2357      C  NE 0.2355    (-0.10%)

A's own NE moves 3.0% on the seed alone, and the effect being measured is 1-3%.
Switching from CPU fp32 to GPU bf16 — same seed, same teacher, same data — was
enough to flip which of the two won. So every number in M3 and M4 is reported as a
mean over seeds with the observed spread next to it, and a difference smaller than
that spread is reported as "not resolved", not as a result.

This reads whatever `runs/transfer/*.json` holds and groups by (k, domain, arm), so
it works the same for one seed or for five — it will just say so.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import ROOT
from models.transfer import ARM_LABELS, ARMS

RUNS_DIR = ROOT / "runs" / "transfer"


def load(pattern: str = "*") -> list[dict]:
    rows = []
    for path in sorted(RUNS_DIR.glob(f"{pattern}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = payload.get("student_seed")
        for r in payload["results"]:
            rows.append({
                "seed": seed, "file": path.stem,
                "k": r["stale_days"], "domain": r["domain"], "arm": r["arm"],
                "ne": r["best"]["ne"], "auc": r["best"]["auc"],
                "teacher_ne": r["teacher_ref"]["ne"],
                "trainable": r["trainable"],
                "selected_epoch": r.get("selected_epoch"),
            })
    return rows


def summarise(rows: list[dict]) -> None:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["k"], r["domain"], r["arm"])].append(r)

    if not grouped:
        print(f"no runs found under {RUNS_DIR}")
        return

    for k, domain in sorted({(k, d) for k, d, _ in grouped}):
        teacher = np.mean([r["teacher_ne"] for a in ARMS
                           for r in grouped.get((k, domain, a), [])])
        print(f"\n{domain}   k={k}d   teacher alone: NE {teacher:.4f}")

        header = (f"  {'arm':<26} {'n':>2}  {'NE mean':>8} {'spread':>8}  "
                  f"{'AUC':>7}  {'vs A':>8}  {'resolved?':>10}  {'params':>8}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        base = grouped.get((k, domain, "vm_only"), [])
        base_ne = np.array([r["ne"] for r in base]) if base else None

        for arm in ARMS:
            runs = grouped.get((k, domain, arm), [])
            if not runs:
                continue
            ne = np.array([r["ne"] for r in runs])
            auc = np.mean([r["auc"] for r in runs])
            spread = ne.max() - ne.min() if len(ne) > 1 else float("nan")

            if base_ne is None or arm == "vm_only":
                delta, verdict = "", ""
            else:
                gap = ne.mean() - base_ne.mean()
                delta = f"{100 * gap / base_ne.mean():+7.2f}%"
                # 判据：arm 之间的差必须超过两边各自的种子波动
                noise = max(spread if len(ne) > 1 else 0.0,
                            base_ne.max() - base_ne.min() if len(base_ne) > 1 else 0.0)
                verdict = ("yes" if abs(gap) > noise else "no") if noise > 0 else "1 seed"

            print(f"  {ARM_LABELS[arm]:<26} {len(runs):>2}  {ne.mean():>8.4f} "
                  f"{spread:>8.4f}  {auc:>7.4f}  {delta:>8}  {verdict:>10}  "
                  f"{runs[0]['trainable']:>8,}")

    n_seeds = len({r["seed"] for r in rows})
    if n_seeds < 3:
        print(f"\n{n_seeds} seed(s). The seed spread on this data is ~3% of NE, "
              f"which is the size of the effects being measured — run at least 3 "
              f"(`--student-seed`) before reading any row as a result.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="*", help="glob over runs/transfer/*.json")
    summarise(load(ap.parse_args().pattern))


if __name__ == "__main__":
    main()
