"""
Train one foundation model, then five students from it.

    python train_transfer.py --domain Software
    python train_transfer.py --domain Software --arms vm_only kd_adapter --k 365
    python train_transfer.py --all-domains --k 0 365 1095      # the M4 sweep

The foundation model is trained once per staleness value and cached, because M4
needs 5 teachers shared across 4 domains x 5 arms = 100 student runs. Retraining
the teacher each time would multiply the sweep's cost by twenty for no
information.

**Teacher and student have the same embedding width and differ in depth.** That is
forced by the `param_share` arm: copying an embedding table requires the shapes to
match. So the student is made smaller along depth and inner widths (`n_layers`,
`n_fmb`, `n_lcb`, `n_queries`) while keeping `dim`. This is a constraint the
experiment inherits from the mechanism it is testing, not a free choice — and it
is the honest version, because a `dim`-scaled student would make arm D
unimplementable and silently turn it into arm A.

The teacher's own score on the eval window is reported alongside the students. It
is the reference that makes the numbers readable: a student that beats its teacher
learned something from fresh data the teacher lacks, and a student that trails it
failed to absorb what was offered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ADAPTER_HOLDOUT_FRAC, DOMAINS, RANDOM_SEED, ROOT, SPLIT_DATE, STALENESS_DAYS
from data.transfer_data import make_loader, transfer_splits
from models.gem import MiniGEM, MiniGEMConfig
from models.transfer import ALL_ARMS, ARM_LABELS, ARMS, TransferConfig, TransferModel
from train import normalized_entropy, roc_auc

RUNS_DIR = ROOT / "runs" / "transfer"
TEACHER_DIR = ROOT / "runs" / "teachers"

# 样本构造方案的版本号，参与教师 checkpoint 的哈希。
# 改过负采样方式之后忘了动它，于是一次运行加载了在**旧任务**上训练的教师，
# 数字看起来完全正常。任何影响样本构造的改动都要 +1。
#   v1  负样本从整个池化目录里均匀采（72-82% 落在用户根本不逛的域里）
#   v2  负样本限制在用户自己的域内
SAMPLE_SCHEME = "v2-domain-restricted-negatives"

# 教师：基础模型。四个域合池，深而宽。
TEACHER_CONFIG = MiniGEMConfig(variant="pooled", dim=32, n_layers=3,
                               n_queries=4, n_fmb=16, n_lcb=16, rank=8)

# 学生：垂直模型。dim 必须和教师一致（arm D 要拷嵌入表），
# 所以只在深度和内部宽度上缩小。
STUDENT_CONFIG = MiniGEMConfig(variant="pooled", dim=32, n_layers=1,
                               n_queries=2, n_fmb=8, n_lcb=8, rank=4,
                               head_hidden=(128, 32))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """Works for a bare MiniGEM and for a TransferModel, which returns a dict."""
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(batch)
        if isinstance(out, dict):
            out = out["student"]
        logits.append(out.float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

    logits, labels = np.concatenate(logits), np.concatenate(labels)
    return {"ne": normalized_entropy(logits, labels),
            "auc": roc_auc(logits, labels),
            "n": int(len(labels))}


def teacher_tag(stale_days: int, args) -> str:
    """A checkpoint name that changes whenever anything affecting the teacher does."""
    payload = json.dumps({
        "stale_days": stale_days,
        "config": asdict(TEACHER_CONFIG),
        "epochs": args.teacher_epochs,
        "lr": args.lr,
        "negatives": args.negatives,
        "split": args.split_date,
        "max_train": args.max_teacher,
        "domains": sorted(args.domains),
        "seed": args.seed,
        "sample_scheme": SAMPLE_SCHEME,
    }, sort_keys=True)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    return f"fm_k{stale_days}_{digest}"


def train_teacher(stale_days: int, args, device: str) -> tuple[MiniGEM, dict]:
    """Train the pooled foundation model, or load it if this exact one exists."""
    splits = transfer_splits(args.domains[0], stale_days,
                             split_date=args.split_date, domains=args.domains)
    model = MiniGEM(splits["vocab_sizes"], splits["item_features"],
                    TEACHER_CONFIG).to(device)

    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    path = TEACHER_DIR / f"{teacher_tag(stale_days, args)}.pt"
    if path.exists() and not args.retrain_teacher:
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        print(f"teacher: loaded {path.name}  (trained on {state['n_positions']:,} "
              f"positions, {state['windows']['teacher']})")
        return model, state["meta"]

    loader = make_loader(splits["pooled"], splits["teacher"], args.batch_size,
                         shuffle=True, n_negatives=args.negatives,
                         limit=args.max_teacher, workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    lo, hi = splits["windows"]["teacher"]
    print(f"teacher: k={stale_days}d  window {lo}..{hi}  "
          f"{len(splits['teacher']):,} positions  "
          f"{model.n_dense_parameters / 1e6:.2f}M dense")

    for epoch in range(1, args.teacher_epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(batch).float(), batch["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            running += loss.item() * len(batch["label"])
            seen += len(batch["label"])
            if step % args.log_every == 0:
                print(f"  teacher epoch {epoch} step {step:>5,}/{len(loader):,}  "
                      f"loss {running / seen:.4f}  {seen / (time.time() - t0):,.0f}/s")
        print(f"  teacher epoch {epoch}: loss {running / seen:.4f}  "
              f"({time.time() - t0:.0f}s)")

    meta = {"stale_days": stale_days, "windows": splits["windows"],
            "n_positions": len(splits["teacher"]),
            "dense_params": model.n_dense_parameters,
            "config": asdict(TEACHER_CONFIG)}
    torch.save({"model": model.state_dict(), "meta": meta,
                "n_positions": len(splits["teacher"]),
                "windows": splits["windows"]}, path)
    print(f"  saved {path.name}")
    return model, meta


def pretrain_adapter(model: TransferModel, splits: dict, args, device: str) -> None:
    """
    Fit the Student Adapter to convergence on `adapter_fit`, then freeze it.

    This is what makes `kd_adapter_holdout` different from `kd_adapter`: the
    adapter here never sees the labels it will later be asked to produce
    distillation targets for. It is trained standalone against the teacher's
    (frozen) output plus item context — exactly the inputs it uses in
    `TransferModel.forward` — so switching it into the main loop afterwards
    changes nothing about its interface, only that its weights stop moving.
    """
    loader = make_loader(splits["pooled"], splits["adapter_fit"], args.batch_size,
                         shuffle=True, n_negatives=args.negatives,
                         workers=args.workers)
    opt = torch.optim.AdamW(model.adapter.parameters(), lr=args.student_lr,
                            weight_decay=args.wd)

    model.adapter.train()
    print(f"    adapter pretrain: {len(splits['adapter_fit']):,} positions "
          f"(held out from the student's {len(splits['student']):,})")
    for epoch in range(1, args.adapter_epochs + 1):
        t0, running, seen = time.time(), 0.0, 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                teacher_out = model.teacher(batch)
                feats = model.teacher.item_features(batch["item"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                adapter_out = model.adapter(teacher_out, feats["popularity_bucket"],
                                            feats["price_bucket"], batch["hist_len"])
                loss = F.binary_cross_entropy_with_logits(adapter_out.float(),
                                                          batch["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n = len(batch["label"])
            running += loss.item() * n
            seen += n
        print(f"      adapter epoch {epoch}: loss {running / seen:.4f}  "
              f"({time.time() - t0:.0f}s)")

    model.adapter.requires_grad_(False)
    model.adapter.eval()


def train_student(arm: str, domain: str, teacher: MiniGEM | None,
                  splits: dict, args, device: str) -> dict:
    """
    One arm, one domain.

    The epoch is chosen on `valid` and the chosen epoch's weights are then scored
    once on `eval`. Selecting and reporting on the same window would favour whichever
    arm has the noisiest trajectory, since a noisier curve gets a luckier minimum —
    and stability is one of the things distillation is supposed to change, so the
    contamination would land directly on the comparison being made.
    """
    torch.manual_seed(args.student_seed)   # 同一 k 下所有 arm 从同一初始化出发

    cfg = TransferConfig(arm=arm, alpha=args.alpha, temperature=args.temperature,
                         representation_dim=args.representation_dim,
                         freeze_shared=not args.tune_shared)
    student = MiniGEM(splits["vocab_sizes"], splits["item_features"], STUDENT_CONFIG)
    vocab = splits["vocab_sizes"]
    model = TransferModel(student, teacher if cfg.needs_teacher else None, cfg,
                          n_popularity_buckets=vocab["popularity_bucket"],
                          n_price_buckets=vocab["price_bucket"]).to(device)

    if arm == "kd_adapter_holdout":
        if "adapter_fit" not in splits:
            raise ValueError(
                "arm 'kd_adapter_holdout' needs splits built with "
                "adapter_holdout_frac set (see data/transfer_data.transfer_splits); "
                "without it there is nothing to hold the adapter's fit out from, "
                "and running it anyway would silently repeat the exact bug this "
                "arm exists to test.")
        pretrain_adapter(model, splits, args, device)
        train_positions = splits["distill"]
    elif args.train_window == "distill":
        if "distill" not in splits:
            raise ValueError("--train-window distill needs --adapter-holdout-frac set")
        train_positions = splits["distill"]
    else:
        train_positions = splits["student"]

    train_loader = make_loader(splits["pooled"], train_positions, args.batch_size,
                               shuffle=True, n_negatives=args.negatives,
                               workers=args.workers)
    valid_loader = make_loader(splits["pooled"], splits["valid"], args.batch_size,
                               shuffle=False, n_negatives=args.negatives,
                               limit=args.max_eval, workers=args.workers)
    eval_loader = make_loader(splits["pooled"], splits["eval"], args.batch_size,
                              shuffle=False, n_negatives=args.negatives,
                              limit=args.max_eval, workers=args.workers)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.student_lr, weight_decay=args.wd)

    window_note = (f"   training on {len(train_positions):,} positions"
                  if len(train_positions) != len(splits["student"]) else "")
    print(f"\n  {ARM_LABELS[arm]}   {model.n_trainable:,} trainable dense params"
          + (f"   shared: {', '.join(model.shared_fields)}" if model.shared_fields else "")
          + window_note)

    history, best_valid, best_state = [], float("inf"), None
    for epoch in range(1, args.student_epochs + 1):
        model.train()
        t0, sums, seen = time.time(), {}, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(batch)
            parts = model.loss({k: v.float() for k, v in out.items()}, batch["label"])

            opt.zero_grad(set_to_none=True)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step()

            n = len(batch["label"])
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.item() * n
            seen += n

        metrics = evaluate(model, valid_loader, device)
        metrics.update(epoch=epoch, seconds=round(time.time() - t0, 1),
                       **{f"loss_{k}": round(v / seen, 4) for k, v in sums.items()})
        history.append(metrics)
        losses = "  ".join(f"{k} {v / seen:.4f}" for k, v in sums.items() if k != "total")
        print(f"    epoch {epoch}: valid NE {metrics['ne']:.4f}  "
              f"AUC {metrics['auc']:.4f}  | {losses}  ({metrics['seconds']}s)")

        if metrics["ne"] < best_valid:
            best_valid, best_epoch = metrics["ne"], epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()
                          if not k.startswith("teacher.")}

    # 用 valid 选出的那一版权重，在 eval 上只跑一次
    model.load_state_dict(best_state, strict=False)
    test = evaluate(model, eval_loader, device)
    print(f"    selected epoch {best_epoch} (valid NE {best_valid:.4f})  ->  "
          f"eval NE {test['ne']:.4f}  AUC {test['auc']:.4f}")

    return {"arm": arm, "domain": domain, "config": asdict(cfg),
            "trainable": model.n_trainable, "shared_fields": model.shared_fields,
            "selected_epoch": best_epoch, "valid_ne": best_valid,
            "n_train_positions": len(train_positions),
            "best": test, "history": history}


def run(args) -> dict:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for stale_days in args.k:
        teacher, teacher_meta = train_teacher(stale_days, args, device)

        for domain in args.run_domains:
            splits = transfer_splits(domain, stale_days, split_date=args.split_date,
                                     domains=args.domains,
                                     adapter_holdout_frac=args.adapter_holdout_frac)
            ref = evaluate(teacher, make_loader(splits["pooled"], splits["eval"],
                                                args.batch_size, shuffle=False,
                                                n_negatives=args.negatives,
                                                limit=args.max_eval,
                                                workers=args.workers), device)
            print(f"\n{domain}  k={stale_days}d  "
                  f"student window {splits['windows']['student'][0]}.."
                  f"{splits['windows']['student'][1]}  "
                  f"{len(splits['student']):,} positions  |  "
                  f"teacher alone on eval: NE {ref['ne']:.4f} AUC {ref['auc']:.4f}")

            for arm in args.arms:
                r = train_student(arm, domain, teacher, splits, args, device)
                r.update(stale_days=stale_days, teacher_ref=ref,
                         windows=splits["windows"],
                         n_student_positions=len(splits["student"]))
                results.append(r)

    payload = {"split_date": args.split_date, "domains": args.domains,
               "teacher_config": asdict(TEACHER_CONFIG),
               "student_config": asdict(STUDENT_CONFIG),
               "teacher_meta": teacher_meta, "results": results}
    payload["student_seed"] = args.student_seed
    tag = args.tag or f"transfer_k{'-'.join(map(str, args.k))}"
    (RUNS_DIR / f"{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'k':>6}  {'domain':<26}  {'arm':<26}  {'NE':>7}  {'AUC':>7}  {'vs A':>7}")
    print("-" * 88)
    baseline: dict[tuple, float] = {}
    for r in results:
        key = (r["stale_days"], r["domain"])
        if r["arm"] == "vm_only":
            baseline[key] = r["best"]["ne"]
        delta = (f"{100 * (r['best']['ne'] / baseline[key] - 1):+6.2f}%"
                 if key in baseline else "     --")
        print(f"{r['stale_days']:>6}  {r['domain']:<26}  {ARM_LABELS[r['arm']]:<26}  "
              f"{r['best']['ne']:>7.4f}  {r['best']['auc']:>7.4f}  {delta:>7}")
    print(f"\nwrote runs/transfer/{tag}.json")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="Software")
    ap.add_argument("--all-domains", action="store_true")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ALL_ARMS,
                    help="'kd_adapter_holdout' is an ablation of arm C and needs "
                         "--adapter-holdout-frac; it is not in the default set")
    ap.add_argument("--k", nargs="+", type=int, default=[0],
                    help=f"staleness in days; the M4 sweep is {STALENESS_DAYS}")

    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight on ground truth; 1-alpha goes to distillation")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--representation-dim", type=int, default=64)
    ap.add_argument("--tune-shared", action="store_true",
                    help="let shared embeddings keep training instead of freezing")

    ap.add_argument("--adapter-holdout-frac", type=float, default=None,
                    help=f"cuts the student window into an earlier fraction that "
                         f"fits the adapter and a later remainder the student "
                         f"trains on; required by 'kd_adapter_holdout' "
                         f"(suggested: {ADAPTER_HOLDOUT_FRAC})")
    ap.add_argument("--adapter-epochs", type=int, default=4,
                    help="epochs to pretrain the adapter for 'kd_adapter_holdout'")
    ap.add_argument("--train-window", choices=["student", "distill"], default="student",
                    help="which slice non-holdout arms train on; set to 'distill' "
                         "to give them the same (smaller) window as "
                         "'kd_adapter_holdout' for a fair paired comparison")

    ap.add_argument("--teacher-epochs", type=int, default=2)
    ap.add_argument("--student-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--student-lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--negatives", type=int, default=4)

    ap.add_argument("--split-date", default=SPLIT_DATE)
    ap.add_argument("--max-teacher", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=50_000)
    ap.add_argument("--retrain-teacher", action="store_true")

    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help="teacher seed; part of the checkpoint identity")
    ap.add_argument("--student-seed", type=int, default=None,
                    help="student init seed, independent of the teacher so a "
                         "seed-variance sweep does not retrain the teacher")
    ap.add_argument("--tag", default=None)

    args = ap.parse_args()
    if args.student_seed is None:
        args.student_seed = args.seed
    args.domains = list(DOMAINS)
    args.run_domains = list(DOMAINS) if args.all_domains else [args.domain]
    run(args)


if __name__ == "__main__":
    main()
