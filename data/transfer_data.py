"""
Windows of time, for teacher and student.

M3 and M4 need three slices of the calendar, not the two that `temporal_split`
provides:

    teacher  ├──── sliding 730d ────┤            k = 0, 90, 365, 730, 1095
    student                  ├──── 365d ────┤
    eval                                    ├── 180d ──┤
                                            T

The teacher is a foundation model: pooled across all four domains, trained on a
fixed-width window ending `k` days before `T`. The student is a vertical model: one
domain, always the same fresh window `[T-365, T]`. Both are scored on `[T, T+180]`,
which neither has seen.

`k` is the staleness dial. As it grows the teacher's knowledge ages while the
student's data stays fixed, which is the regime GEM's Student Adapter exists for. If
C's advantage over B does not widen with `k`, claim B is wrong — at least at this
scale, on this data.

The student's window is a fixed width rather than `[T-k, T]`, which was the first
design and was wrong twice over: at `k = 0` it was empty, and at `k = 3` it held 355
samples. Staleness is the teacher's lag, not the student's data volume, and the two
are independent in production. `staleness_windows` documents the sliding-versus-
expanding teacher choice that follows from the same concern.

**Everything lives in the pooled id space.** A student loaded through
`load_domain` would number the same product differently from a teacher trained on
`load_pooled` output, and the teacher would score the batch confidently against
unrelated items without raising anything. So the student's slice is cut out of the
pooled arrays with `domain_positions`, and both models are built against pooled
vocabularies.

That inflates the student's embedding tables — it allocates rows for 97k items and
only ever touches its own domain's 26k. Two reasons that is the right trade: rows
never indexed receive no gradient and stay at initialisation, so they change no
result; and `n_dense_parameters`, which is what the arms are compared on, excludes
embedding tables entirely. The inflation is identical across all five arms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    EVAL_WINDOW_DAYS,
    MAX_SEQ_LEN,
    RANDOM_SEED,
    SPLIT_DATE,
    STALENESS_DAYS,
    STUDENT_WINDOW_DAYS,
    TEACHER_WINDOW_DAYS,
)
from data.dataset import (  # noqa: E402
    DomainData,
    InteractionDataset,
    collate,
)
from data.pooled import (  # noqa: E402
    PooledData,
    domain_positions,
    load_pooled,
    user_negative_ranges,
)

_POOLED_CACHE: dict[tuple[str, ...], PooledData] = {}


def pooled_cached(domains: list[str] | None = None) -> PooledData:
    """
    Load the pooled data once per process.

    M4 runs 4 domains x 5 arms x 5 staleness values = 100 student trainings. At
    ~40 s to build the pooled arrays, reloading per run would cost over an hour of
    pure parquet parsing.
    """
    key = tuple(domains) if domains else ("__default__",)
    if key not in _POOLED_CACHE:
        _POOLED_CACHE[key] = load_pooled(domains)
    return _POOLED_CACHE[key]


def window_positions(
    data: DomainData,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> np.ndarray:
    """
    All valid positions whose candidate timestamp falls in `[start, end)`.

    Same two rules as `temporal_split`: positions are (user, offset) pairs, and an
    event with no legitimate history is dropped — which `hist_end` decides, not
    the position index, because events tied at the candidate's exact timestamp do
    not count as history.
    """
    n_users = data.n_users
    starts = data.user_offsets[:n_users]
    repeat = data.user_offsets[1:n_users + 1] - starts

    user_idx = np.repeat(np.arange(n_users), repeat)
    global_idx = np.arange(len(data.user_items))
    pos = global_idx - np.repeat(starts, repeat)

    keep = data.hist_end[global_idx] > np.repeat(starts, repeat)
    user_idx, pos, ts = user_idx[keep], pos[keep], data.user_ts[keep]

    mask = np.ones(len(ts), dtype=bool)
    if start is not None:
        mask &= ts >= np.int64(start.value)
    if end is not None:
        mask &= ts < np.int64(end.value)

    return np.stack([user_idx[mask], pos[mask]], axis=1)


def staleness_windows(
    split_date: pd.Timestamp,
    stale_days: int,
    student_days: int = STUDENT_WINDOW_DAYS,
    teacher_days: int | None = TEACHER_WINDOW_DAYS,
    eval_days: int = EVAL_WINDOW_DAYS,
) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp]]:
    """
    Name the three windows, so a run logs exactly what it trained on.

    The student's window has a **fixed width** and does not move with `k`. An
    earlier version tied it to `k` — student on `[T-k, T]` — which made the
    experiment meaningless in two ways at once: at `k = 0` the window was empty,
    and at `k = 3` it held 355 samples. Staleness is about the teacher's *lag*,
    not the student's data volume; those are independent in production and have to
    be independent here.

    `teacher_days` chooses between two teachers, and the choice is a real
    methodological fork:

    - **sliding** (`teacher_days` set, the default): `[T-k-W, T-k]`. Every teacher
      sees a fixed-width window, just from a different era. Measured volumes are
      430k-524k positions across the whole `k` sweep, and they are *larger* for
      older windows — so if an older teacher performs worse, less data cannot be
      the reason. The confound runs against the hypothesis.
    - **expanding** (`teacher_days=None`): everything up to `T-k`, which is what a
      production FM actually trains on. More realistic, but confounded: `k = 1095`
      also removes 28% of the data, so staleness and volume move together and a
      degradation cannot be attributed to either.

    Both are available; the mechanism test uses sliding and reports expanding as
    the realistic variant.
    """
    t = pd.Timestamp(split_date)
    teacher_end = t - pd.Timedelta(days=stale_days)
    teacher_start = (teacher_end - pd.Timedelta(days=teacher_days)
                     if teacher_days is not None else None)
    return {
        "teacher": (teacher_start, teacher_end),
        "student": (t - pd.Timedelta(days=student_days), t),
        "eval": (t, t + pd.Timedelta(days=eval_days)),
    }


def make_loader(
    data: DomainData,
    positions: np.ndarray,
    batch_size: int,
    shuffle: bool,
    n_negatives: int = 4,
    limit: int | None = None,
    workers: int = 0,
    seed: int = RANDOM_SEED,
) -> DataLoader:
    """
    A loader whose negatives stay inside the user's own domain.

    That restriction is not a refinement, it is a correction. Uniform negatives over
    the pooled catalogue land outside the user's domain 72-82% of the time, and the
    resulting task — "is this item even in a category this user shops in" — is one
    the category embedding answers immediately. It showed up as AUC rising from
    0.932 on solo data to 0.988 once pooled, with all five transfer arms inside a
    1.5% band because every one of them was at the ceiling.
    """
    if limit is not None and len(positions) > limit:
        rng = np.random.default_rng(seed)
        positions = positions[rng.choice(len(positions), limit, replace=False)]

    ranges = (user_negative_ranges(data) if isinstance(data, PooledData) else None)
    ds = InteractionDataset(data, positions, n_negatives=n_negatives,
                            max_seq_len=MAX_SEQ_LEN, negative_ranges=ranges, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=workers,
                      pin_memory=True, drop_last=shuffle)


def transfer_splits(
    domain: str,
    stale_days: int = 0,
    split_date: str | pd.Timestamp = SPLIT_DATE,
    teacher_days: int | None = TEACHER_WINDOW_DAYS,
    student_days: int = STUDENT_WINDOW_DAYS,
    eval_days: int = EVAL_WINDOW_DAYS,
    domains: list[str] | None = None,
) -> dict:
    """
    Everything a transfer run needs, already cut and already in one id space.

    Returns the pooled data (so a model can be built from its vocabularies and
    feature table), the teacher's pooled positions, and the student's and
    evaluation positions restricted to one domain.
    """
    pooled = pooled_cached(domains)
    if domain not in pooled.domain_names:
        raise ValueError(f"{domain!r} not in pooled data {pooled.domain_names}")

    windows = staleness_windows(pd.Timestamp(split_date), stale_days,
                               student_days, teacher_days, eval_days)
    cut = {name: window_positions(pooled, *bounds) for name, bounds in windows.items()}

    def named(bounds) -> tuple[str | None, str]:
        lo, hi = bounds
        return (str(lo.date()) if lo is not None else None, str(hi.date()))

    return {
        "pooled": pooled,
        "stale_days": stale_days,
        "windows": {k: named(v) for k, v in windows.items()},
        "teacher": cut["teacher"],                                    # 全部四个域
        "student": domain_positions(pooled, cut["student"], domain),   # 只此一域
        "eval": domain_positions(pooled, cut["eval"], domain),
        "item_features": torch.from_numpy(pooled.item_features),
        "vocab_sizes": pooled.vocab_sizes,
    }


if __name__ == "__main__":
    # 先看清每个 k 下三个窗口各有多少样本。学生窗口太小的话整个扫描无从谈起，
    # 教师窗口随 k 变化太小的话扫描测不出任何东西——两种失败都不会报错。
    pooled = pooled_cached()
    ref = len(window_positions(pooled, None, pd.Timestamp(SPLIT_DATE)))

    print(f"split {SPLIT_DATE}   student window {STUDENT_WINDOW_DAYS}d   "
          f"teacher window {TEACHER_WINDOW_DAYS}d (sliding)   eval {EVAL_WINDOW_DAYS}d\n")

    header = (f"{'k':>6}  {'teacher span':>23}  {'teacher n':>10}  "
              f"{'vs k=0':>7}  " + "  ".join(f"{d[:11]:>12}" for d in pooled.domain_names))
    print(header)
    print("-" * len(header))

    base = None
    for k in STALENESS_DAYS:
        s = transfer_splits(pooled.domain_names[0], k)
        n = len(s["teacher"])
        base = base or n
        lo, hi = s["windows"]["teacher"]
        per_domain = [len(transfer_splits(d, k)["student"]) for d in pooled.domain_names]
        print(f"{k:>6}  {lo} .. {hi}  {n:>10,}  {100 * (n / base - 1):>+6.1f}%  "
              + "  ".join(f"{m:>12,}" for m in per_domain))

    print(f"\nstudent window is fixed, so those four columns must not vary with k.")
    print("eval positions (neither model sees these):")
    for d in pooled.domain_names:
        print(f"  {d:30} {len(transfer_splits(d, 0)['eval']):>9,}")
    print(f"\nreference: all positions before {SPLIT_DATE} = {ref:,}")
