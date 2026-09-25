"""
Sample construction and batching.

A sample is one (user, candidate item) pair with a binary label, plus the
user's history *strictly before* that candidate's timestamp. Two properties of
that definition carry the whole experiment:

- **History is causal.** Slicing the user's sequence at the candidate's position
  means no sample can see its own future. Get this wrong and every downstream
  number is meaningless in a way that looks like success.
- **Splits are by calendar date, not by position.** The staleness sweep in M4
  trains the teacher on data up to `T - k` and the student on `[T - k, T]`, so
  the split boundary has to be a real date shared across users.

Sequences are stored CSR-style — one flat array of items plus per-user offsets —
so building a sample is a slice rather than a lookup, and 5M samples cost a few
hundred MB instead of tens of gigabytes.

Batches are padded to the **longest sequence in the batch**, never to a fixed
maximum. M0 measured why: at cap=32 that difference is 6 points of wasted
compute, and pad-to-fixed would have thrown away 74%.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MAX_SEQ_LEN, PROCESSED_DIR, RANDOM_SEED  # noqa: E402

PAD = 0

ITEM_FEATURE_COLUMNS = ["store", "category", "price_bucket",
                        "rating_bucket", "popularity_bucket"]


@dataclass
class DomainData:
    """Everything one domain contributes, already encoded and indexable."""

    domain: str
    user_items: np.ndarray       # 所有用户的商品序列，首尾相接
    user_offsets: np.ndarray     # 第 i 个用户的区间是 [offsets[i], offsets[i+1])
    user_ts: np.ndarray          # 与 user_items 对齐，**纳秒** int64
    hist_end: np.ndarray         # 每个事件的历史截止位置（不含），已排除同刻事件
    item_features: np.ndarray    # [n_items, n_feature_columns]
    vocab_sizes: dict[str, int]
    n_users: int
    n_items: int

    def history(self, user_idx: int, pos: int, max_len: int) -> np.ndarray:
        """
        The user's items strictly before this event, capped to the most recent.

        The cut uses `hist_end` rather than the position itself: events sharing
        the candidate's exact timestamp are excluded. At prediction time you do
        not get to see what else happened in the same instant, and 1.8% of
        samples in this data have such ties.
        """
        start = self.user_offsets[user_idx]
        end = self.hist_end[start + pos]
        return self.user_items[max(start, end - max_len):end]


def load_domain(domain: str) -> DomainData:
    src = PROCESSED_DIR / domain
    vocab = json.loads((src / "vocab.json").read_text(encoding="utf-8"))
    item_vocab, user_vocab = vocab["item_vocab"], vocab["user_vocab"]

    inter = pd.read_parquet(src / "interactions.parquet",
                            columns=["user_id", "parent_asin", "ts"])
    # prepare.py 已按 (user_id, ts) 排过序；这里再排一次是为了让本文件
    # 不依赖上游的隐式约定——顺序错了会静默地破坏因果性。
    inter = inter.sort_values(["user_id", "ts"], kind="stable")

    inter["u"] = inter["user_id"].map(user_vocab).astype("int64")
    inter["i"] = inter["parent_asin"].map(item_vocab).astype("int64")
    inter = inter.sort_values(["u", "ts"], kind="stable")

    counts = inter.groupby("u", sort=True).size()
    offsets = np.zeros(len(user_vocab) + 1, dtype=np.int64)
    # counts 的索引是词表 id（从 N_RESERVED 起），减去偏移得到稠密下标
    base = min(user_vocab.values())
    offsets[counts.index.to_numpy() - base + 1] = counts.to_numpy()
    offsets = np.cumsum(offsets)

    feats = pd.read_parquet(src / "item_features.parquet").sort_values("item_id")
    n_items = vocab["vocab_sizes"]["item_id"]
    item_features = np.zeros((n_items, len(ITEM_FEATURE_COLUMNS)), dtype=np.int64)
    item_features[feats["item_id"].to_numpy()] = feats[ITEM_FEATURE_COLUMNS].to_numpy()

    # 强制纳秒。parquet 往返后 pandas 可能保留毫秒精度，而 pd.Timestamp.value
    # 永远是纳秒——两者直接比较会差 1000 倍，且不会报错，只会静默切错。
    ts_ns = inter["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")

    # 每个事件的历史截止位置：同一 (user, ts) 的连续事件构成一个 run，
    # 历史在 run 的起点处截断，因此同刻事件不会进入历史。
    u_arr = inter["u"].to_numpy()
    idx = np.arange(len(ts_ns))
    new_run = np.empty(len(ts_ns), dtype=bool)
    new_run[0] = True
    new_run[1:] = (u_arr[1:] != u_arr[:-1]) | (ts_ns[1:] != ts_ns[:-1])
    hist_end = np.maximum.accumulate(np.where(new_run, idx, 0))

    return DomainData(
        domain=domain,
        user_items=inter["i"].to_numpy(),
        user_offsets=offsets,
        user_ts=ts_ns,
        hist_end=hist_end,
        item_features=item_features,
        vocab_sizes=vocab["vocab_sizes"],
        n_users=len(user_vocab),
        n_items=n_items,
    )


def temporal_split(
    data: DomainData,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp | None = None,
) -> dict[str, np.ndarray]:
    """
    Split *positions* (not users) by the timestamp of the candidate event.

    Events with an empty history are dropped. That means more than just the
    first event per user: an event whose only predecessors share its exact
    timestamp also has nothing legitimate to condition on, so `hist_end` decides
    rather than the position index.
    """
    n_users = data.n_users
    starts = data.user_offsets[:n_users]
    ends = data.user_offsets[1:n_users + 1]

    repeat = ends - starts
    user_idx = np.repeat(np.arange(n_users), repeat)
    global_idx = np.arange(len(data.user_items))
    pos = global_idx - np.repeat(starts, repeat)

    keep = data.hist_end[global_idx] > np.repeat(starts, repeat)
    user_idx, pos = user_idx[keep], pos[keep]
    ts = data.user_ts[keep]

    t_train = np.int64(train_end.value)
    t_valid = np.int64((valid_end or train_end).value)

    splits = {
        "train": ts < t_train,
        "valid": (ts >= t_train) & (ts < t_valid) if valid_end else np.zeros_like(keep[keep]),
        "test": ts >= t_valid,
    }
    return {k: np.stack([user_idx[m], pos[m]], axis=1) for k, m in splits.items()}


class InteractionDataset(Dataset):
    """
    Positives are observed interactions; negatives are sampled items.

    Uniform negative sampling is the conventional baseline. Popularity-weighted
    sampling is closer to what an ads system actually serves, and is available
    behind a flag so the choice stays visible rather than baked in.
    """

    def __init__(
        self,
        data: DomainData,
        positions: np.ndarray,
        n_negatives: int = 4,
        max_seq_len: int = MAX_SEQ_LEN,
        popularity_negatives: bool = False,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.data = data
        self.positions = positions
        self.n_negatives = n_negatives
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)

        if popularity_negatives:
            counts = np.bincount(data.user_items, minlength=data.n_items).astype("float64")
            counts[:3] = 0.0                      # 保留位不参与采样
            self.neg_p = counts / counts.sum()
        else:
            self.neg_p = None

    def __len__(self) -> int:
        return len(self.positions) * (1 + self.n_negatives)

    def __getitem__(self, idx: int) -> dict:
        row, slot = divmod(idx, 1 + self.n_negatives)
        user_idx, pos = self.positions[row]

        hist = self.data.history(user_idx, pos, self.max_seq_len)
        true_item = self.data.user_items[self.data.user_offsets[user_idx] + pos]

        if slot == 0:
            item, label = true_item, 1.0
        else:
            item, label = self._sample_negative(hist, true_item), 0.0

        return {
            "user": int(user_idx) + 3,          # 对齐词表里的保留位偏移
            "item": int(item),
            "hist": torch.from_numpy(hist.astype(np.int64)),
            "label": label,
        }

    def _sample_negative(self, hist: np.ndarray, true_item: int) -> int:
        """Reject items the user already interacted with, but bound the retries."""
        seen = set(hist.tolist()) | {true_item}
        for _ in range(10):
            if self.neg_p is not None:
                cand = int(self.rng.choice(self.data.n_items, p=self.neg_p))
            else:
                cand = int(self.rng.integers(3, self.data.n_items))
            if cand not in seen:
                return cand
        return cand      # 罕见：连续碰撞，接受一个假阴性而不是无限重试


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Pad histories to the longest in *this batch*.

    `lengths` is returned alongside so the model can build its own mask; the
    padded positions hold PAD (0) and must never reach the loss.
    """
    lengths = torch.tensor([len(b["hist"]) for b in batch], dtype=torch.long)
    max_len = int(lengths.max())

    hist = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    for i, b in enumerate(batch):
        hist[i, : len(b["hist"])] = b["hist"]

    return {
        "user": torch.tensor([b["user"] for b in batch], dtype=torch.long),
        "item": torch.tensor([b["item"] for b in batch], dtype=torch.long),
        "hist": hist,
        "hist_len": lengths,
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }
