"""
Training loop for one MiniGEM run.

The headline metric is **normalized entropy**, the metric Meta reports for
these models ("neutral NE" in the BlockAttention section of the GEM training
post). It is log loss divided by the log loss of always predicting the base
rate:

    NE = logloss(model) / logloss(base rate)

Raw log loss is not comparable across setups here, because the positive rate is
set by the negative sampling ratio and differs between experiments. Dividing by
the entropy of the base rate removes that, so NE < 1 means "better than knowing
nothing", and runs with different sampling ratios can be read on one axis.

AUC is reported alongside, and the pair matters more than either alone. AUC is
rank-only; NE responds to calibration as well. Later milestones lean on exactly
that split — a stale teacher degrades a student's calibration long before it
degrades its ranking.

Usage:
    python train.py --domain Software --variant pooled --epochs 2
    python train.py --domain Software --variant interleaved --scale 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import MAX_SEQ_LEN, PROCESSED_DIR, RANDOM_SEED, ROOT
from data.dataset import (
    InteractionDataset,
    collate,
    load_domain,
    temporal_split,
)
from models.gem import MiniGEM, MiniGEMConfig

RUNS_DIR = ROOT / "runs"


def normalized_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Log loss divided by the log loss of predicting the base rate."""
    p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-7, 1 - 1e-7)
    model_ll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    base = float(np.mean(labels))
    if base <= 0.0 or base >= 1.0:
        return float("nan")
    base_ll = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    return float(model_ll / base_ll)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC; no sklearn dependency for one formula."""
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 并列分数取平均秩，否则大量相同预测会让 AUC 偏高
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


@torch.no_grad()
def evaluate(model: MiniGEM, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(batch)
        logits.append(out.float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    return {
        "ne": normalized_entropy(logits, labels),
        "auc": roc_auc(logits, labels),
        "n": len(labels),
    }


def make_loaders(domain: str, args) -> tuple[DataLoader, DataLoader, DataLoader, dict, torch.Tensor]:
    data = load_domain(domain)
    splits = temporal_split(data, pd.Timestamp(args.train_end),
                            pd.Timestamp(args.valid_end))

    def loader(positions: np.ndarray, shuffle: bool, limit: int | None) -> DataLoader:
        if limit is not None and len(positions) > limit:
            rng = np.random.default_rng(RANDOM_SEED)
            positions = positions[rng.choice(len(positions), limit, replace=False)]
        ds = InteractionDataset(data, positions, n_negatives=args.negatives,
                                max_seq_len=MAX_SEQ_LEN, seed=RANDOM_SEED)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=args.workers,
                          pin_memory=True, drop_last=shuffle)

    return (
        loader(splits["train"], True, args.max_train),
        loader(splits["valid"], False, args.max_valid),
        loader(splits["test"], False, args.max_eval) if len(splits["test"]) else None,
        data.vocab_sizes,
        torch.from_numpy(data.item_features),
    )


def count_flops(model: MiniGEM, batch: dict[str, torch.Tensor]) -> float:
    """
    Measured, not estimated: FLOPs/example for one forward+backward pass.

    A hand-derived formula would have to track every matmul across the FM
    stacks, the attention pooling, and the head separately, and get it wrong
    silently. `FlopCounterMode` traces the actual ops instead — the same
    measure-don't-guess standard the rest of this project holds itself to.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.train()
    fc = FlopCounterMode(display=False)
    with fc:
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
        loss.backward()
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return fc.get_total_flops() / len(batch["label"])


def train(args) -> dict:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)

    train_loader, valid_loader, eval_loader, vocab_sizes, item_features = make_loaders(
        args.domain, args)

    cfg = MiniGEMConfig(variant=args.variant, dim=args.dim, n_layers=args.layers,
                        n_queries=args.queries, dropout=args.dropout)
    if args.scale != 1.0:
        cfg = cfg.scaled(args.scale)

    model = MiniGEM(vocab_sizes, item_features, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    flops_per_example = None
    if args.flops:
        probe_batch = next(iter(train_loader))
        probe_batch = {k: v.to(device, non_blocking=True) for k, v in probe_batch.items()}
        flops_per_example = count_flops(model, probe_batch)

    print(f"{args.domain} | {cfg.variant} | dim={cfg.dim} layers={cfg.n_layers} "
          f"M={cfg.n_queries}")
    print(f"params: {model.n_parameters / 1e6:.2f}M "
          f"({model.n_dense_parameters / 1e6:.2f}M dense)"
          + (f"  |  {flops_per_example / 1e6:.2f}M FLOPs/example (fwd+bwd)"
             if flops_per_example else ""))
    print(f"train batches: {len(train_loader):,}  valid batches: {len(valid_loader):,}\n")

    history, best, best_state, seen_total = [], {"ne": float("inf")}, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits.float(), batch["label"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            running += loss.item() * len(batch["label"])
            seen += len(batch["label"])
            if step % args.log_every == 0:
                rate = seen / (time.time() - t0)
                print(f"  epoch {epoch} step {step:>6,}/{len(train_loader):,}  "
                      f"loss {running / seen:.4f}  {rate:,.0f} samples/s")

        seen_total += seen
        metrics = evaluate(model, valid_loader, device)
        metrics.update(epoch=epoch, train_loss=running / seen,
                       seconds=round(time.time() - t0, 1))
        history.append(metrics)
        print(f"  epoch {epoch}: valid NE {metrics['ne']:.4f}  "
              f"AUC {metrics['auc']:.4f}  ({metrics['seconds']}s)\n")

        if metrics["ne"] < best["ne"]:
            best = dict(metrics)
            if eval_loader is not None:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif args.early_stop:
            print("  NE stopped improving — early stop")
            break

    result = {
        "domain": args.domain,
        "config": asdict(cfg),
        "params": model.n_parameters,
        "dense_params": model.n_dense_parameters,
        "flops_per_example": flops_per_example,
        "train_examples_seen": seen_total,
        "train_flops": flops_per_example * seen_total if flops_per_example else None,
        "best": best,
        "history": history,
    }

    # 用 valid 选出的那一版权重，在从未参与过选择的 test 窗口上只报一次——
    # 否则更容易过拟合的大模型会在"哪个 epoch 最好"这个选择本身上占便宜，
    # M3 已经踩过这个坑（见 STATE.md §3.4）。
    if eval_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        result["eval"] = evaluate(model, eval_loader, device)
        print(f"selected epoch {best['epoch']} (valid NE {best['ne']:.4f})  ->  "
              f"eval NE {result['eval']['ne']:.4f}  AUC {result['eval']['auc']:.4f}")

    RUNS_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"{args.domain}_{cfg.variant}_d{cfg.dim}_l{cfg.n_layers}"
    (RUNS_DIR / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"best valid NE {best['ne']:.4f} | wrote runs/{tag}.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="Software")
    ap.add_argument("--variant", default="pooled", choices=["pooled", "interleaved"])

    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--queries", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every width — the knob for the scaling-law sweep")

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--negatives", type=int, default=4)

    ap.add_argument("--train-end", default="2021-01-01")
    ap.add_argument("--valid-end", default="2022-06-01")
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap training positions — for smoke runs")
    ap.add_argument("--max-valid", type=int, default=50_000)
    ap.add_argument("--max-eval", type=int, default=50_000,
                    help="cap on the held-out window after --valid-end, used for "
                         "a once-only report with the epoch selected on --valid; "
                         "see train's eval_loader")

    ap.add_argument("--workers", type=int, default=0,
                    help="0 on Windows: the dataset holds large arrays that "
                         "spawned workers would copy")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--flops", action="store_true",
                    help="measure FLOPs/example (fwd+bwd) via torch's FlopCounterMode "
                         "— the M5 scaling-law x-axis")

    args = ap.parse_args()
    if not (PROCESSED_DIR / args.domain / "interactions.parquet").exists():
        raise SystemExit(f"{args.domain} not prepared — run `python -m data.prepare`")
    train(args)


if __name__ == "__main__":
    main()
