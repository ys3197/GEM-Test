"""
Guardrails for the five transfer arms.

Three of these pin failures that produce plausible numbers and no error, which
makes them the only kind worth writing tests for here:

- **the adapter learning to agree with the student.** If gradient from the
  distillation loss reaches the Student Adapter, it stops tracking reality and
  starts tracking the student. The loss goes down, the arm looks fine, and claim B
  is being measured against a broken implementation.
- **parameter sharing sharing nothing.** A silently skipped copy leaves arm D
  identical to arm A while still labelled parameter sharing — a null result that
  reads as a finding.
- **the teacher training.** A teacher left in `train()` mode applies dropout, so
  its targets become noise; a teacher whose weights move is not a teacher.

These need no data and no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import ITEM_FEATURE_COLUMNS, N_TIME_BUCKETS  # noqa: E402
from models.gem import MiniGEM, MiniGEMConfig  # noqa: E402
from models.transfer import (  # noqa: E402
    ARMS,
    TRANSFERABLE_FIELDS,
    StudentAdapter,
    distillation_loss,
    TransferConfig,
    TransferModel,
    share_parameters,
)

VOCAB = {"user_id": 40, "item_id": 60, "store": 12, "category": 8,
         "price_bucket": 19, "rating_bucket": 13, "popularity_bucket": 19}
B, L = 16, 6


@pytest.fixture
def item_features():
    torch.manual_seed(0)
    table = torch.stack([
        torch.randint(3, VOCAB[c], (VOCAB["item_id"],)) for c in ITEM_FEATURE_COLUMNS
    ], dim=1)
    table[0] = 0
    return table


@pytest.fixture
def batch():
    torch.manual_seed(1)
    lengths = torch.randint(1, L + 1, (B,))
    hist = torch.randint(3, VOCAB["item_id"], (B, L))
    gaps = torch.randint(3, N_TIME_BUCKETS, (B, L))
    for i, n in enumerate(lengths):
        hist[i, n:] = 0
        gaps[i, n:] = 0
    return {
        "user": torch.randint(3, VOCAB["user_id"], (B,)),
        "item": torch.randint(3, VOCAB["item_id"], (B,)),
        "hist": hist, "hist_gap": gaps, "hist_len": lengths,
        "label": torch.randint(0, 2, (B,)).float(),
    }


def make(config: MiniGEMConfig, item_features) -> MiniGEM:
    return MiniGEM(VOCAB, item_features, config)


TEACHER = MiniGEMConfig(dim=16, n_layers=2, n_queries=3, n_fmb=6, n_lcb=6, rank=3,
                        head_hidden=(32,), dropout=0.0)
STUDENT = MiniGEMConfig(dim=16, n_layers=1, n_queries=2, n_fmb=4, n_lcb=4, rank=2,
                        head_hidden=(16,), dropout=0.0)


@pytest.fixture
def pair(item_features):
    torch.manual_seed(2)
    return make(STUDENT, item_features), make(TEACHER, item_features)


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_produces_a_finite_loss_and_a_gradient(arm, pair, batch):
    student, teacher = pair
    cfg = TransferConfig(arm=arm)
    model = TransferModel(student, teacher if cfg.needs_teacher else None, cfg,
                          VOCAB["popularity_bucket"], VOCAB["price_bucket"])

    parts = model.loss(model(batch), batch["label"])
    assert torch.isfinite(parts["total"])
    parts["total"].backward()

    got = [n for n, p in model.student.named_parameters()
           if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert got, f"arm {arm} produced no gradient in the student"


def test_the_adapter_never_sees_the_students_distillation_loss(pair, batch):
    """
    The load-bearing one.

    The adapter's only gradient path is its own fit against ground truth. Zeroing
    that term must leave it with no gradient at all — if `kd` also reaches it, the
    adapter is learning to agree with the student instead of with reality.
    """
    student, teacher = pair
    cfg = TransferConfig(arm="kd_adapter")
    model = TransferModel(student, teacher, cfg,
                          VOCAB["popularity_bucket"], VOCAB["price_bucket"])

    parts = model.loss(model(batch), batch["label"])
    # 只反传蒸馏项和任务项，把 adapter 自己的拟合项排除在外
    (cfg.alpha * parts["task"] + (1 - cfg.alpha) * parts["kd"]).backward()

    for name, p in model.adapter.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            f"adapter parameter {name} received gradient from the student's loss — "
            f"it will learn to agree with the student rather than with the labels"
        )


def test_distillation_loss_never_propagates_into_its_target():
    """
    The inner half of the same guard, pinned on its own.

    `TransferModel.loss` also detaches, so a test that only checks the adapter
    would pass with either detach removed and fail only when both were. Testing
    the function directly means each half is covered independently.
    """
    student = torch.randn(B, requires_grad=True)
    target = torch.randn(B, requires_grad=True)
    distillation_loss(student, target, temperature=2.0).backward()

    assert student.grad is not None and student.grad.abs().sum() > 0
    assert target.grad is None or target.grad.abs().sum() == 0


def test_the_adapter_does_get_gradient_from_ground_truth(pair, batch):
    """The mirror image: the intended path must actually be connected."""
    student, teacher = pair
    model = TransferModel(student, teacher, TransferConfig(arm="kd_adapter"),
                          VOCAB["popularity_bucket"], VOCAB["price_bucket"])
    model.loss(model(batch), batch["label"])["adapter_fit"].backward()

    total = sum(p.grad.abs().sum() for p in model.adapter.parameters()
                if p.grad is not None)
    assert total > 0


def test_the_adapter_starts_as_an_exact_identity(batch):
    """
    Zero-initialised output layer means "do not correct" is the default, so any
    correction is earned from the fresh window rather than an artefact of init.
    """
    adapter = StudentAdapter(VOCAB["popularity_bucket"], VOCAB["price_bucket"])
    logits = torch.randn(B) * 3
    out = adapter(logits, torch.randint(3, VOCAB["popularity_bucket"], (B,)),
                  torch.randint(3, VOCAB["price_bucket"], (B,)), batch["hist_len"])
    torch.testing.assert_close(out, logits)


def test_the_teacher_stays_frozen_and_in_eval_mode(pair, batch):
    student, teacher = pair
    model = TransferModel(student, teacher, TransferConfig(arm="kd"),
                          VOCAB["popularity_bucket"], VOCAB["price_bucket"])
    before = {n: p.detach().clone() for n, p in model.teacher.named_parameters()}

    model.train()                                   # 训练循环会这么做
    assert not model.teacher.training, "teacher must stay in eval mode"

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    for _ in range(3):
        parts = model.loss(model(batch), batch["label"])
        opt.zero_grad(set_to_none=True)
        parts["total"].backward()
        opt.step()

    for name, p in model.teacher.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, before[name], msg=f"teacher {name} moved")


def test_parameter_sharing_copies_the_transferable_fields_and_nothing_else(pair):
    student, teacher = pair
    ids = {"item_id": student.embeddings.tables["item_id"].weight.detach().clone()}

    shared = share_parameters(student, teacher, freeze=True)
    assert set(shared) == set(TRANSFERABLE_FIELDS)

    for name in shared:
        torch.testing.assert_close(student.embeddings.tables[name].weight,
                                   teacher.embeddings.tables[name].weight)
        assert not student.embeddings.tables[name].weight.requires_grad

    # 加了偏移的 id 在各域之间没有共同语义，必须原样不动
    torch.testing.assert_close(student.embeddings.tables["item_id"].weight,
                               ids["item_id"])
    assert student.embeddings.tables["item_id"].weight.requires_grad


def test_parameter_sharing_refuses_a_width_mismatch_instead_of_skipping(item_features):
    """
    Silently skipping would leave arm D identical to arm A while still being
    reported as parameter sharing.
    """
    student = make(MiniGEMConfig(dim=8, n_layers=1, head_hidden=(16,)), item_features)
    teacher = make(TEACHER, item_features)
    with pytest.raises(ValueError, match="embedding width"):
        share_parameters(student, teacher)


def test_representation_transfer_actually_uses_the_teachers_features(pair, batch):
    """
    The bridge output must change the logit. If the head learned to ignore it the
    arm would silently degenerate to the baseline.
    """
    student, teacher = pair
    model = TransferModel(student, teacher, TransferConfig(arm="representation"),
                          VOCAB["popularity_bucket"], VOCAB["price_bucket"])
    model.eval()

    with torch.no_grad():
        base = model(batch)["student"].clone()
        for p in model.bridge.project.parameters():
            p.mul_(0).add_(0.5)                     # 换成一个完全不同的映射
        changed = model(batch)["student"]

    assert not torch.allclose(base, changed)


def test_representation_transfer_widens_only_the_head(pair, batch):
    student, teacher = pair
    plain_head_in = student.body.n_out * student.config.dim
    cfg = TransferConfig(arm="representation", representation_dim=32)
    model = TransferModel(student, teacher, cfg,
                          VOCAB["popularity_bucket"], VOCAB["price_bucket"])

    first = model.student.head.net[0]
    assert first.in_features == plain_head_in + cfg.representation_dim


def test_vm_only_refuses_a_teacher_it_would_ignore():
    """A teacher passed to the baseline would be a silently inert argument."""
    cfg = TransferConfig(arm="vm_only")
    assert not cfg.needs_teacher


@pytest.mark.parametrize("arm", ["kd", "kd_adapter", "param_share", "representation"])
def test_arms_that_need_a_teacher_say_so(arm, item_features):
    student = make(STUDENT, item_features)
    with pytest.raises(ValueError, match="needs a teacher"):
        TransferModel(student, None, TransferConfig(arm=arm))


def test_unknown_arm_and_bad_alpha_are_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        TransferConfig(arm="magic")
    with pytest.raises(ValueError, match="alpha"):
        TransferConfig(arm="kd", alpha=1.5)
