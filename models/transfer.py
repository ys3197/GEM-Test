"""
The ways a foundation model can hand knowledge to a vertical model, plus a control.

GEM describes three post-training techniques — knowledge distillation with a
Student Adapter, representation learning, and parameter sharing. Each transfers
something different, and the difference is *where the knowledge ends up*:

    arm  technique                what crosses           where it lives after
    ---  ----------------------   --------------------   --------------------
    A    (none)                   nothing                —
    B    knowledge distillation   the teacher's output   the student's weights
    C    + Student Adapter        a corrected output     the student's weights
    D    parameter sharing        the teacher's weights  shared tensors
    E    representation transfer  the teacher's features an input at serving time
    E'   shuffled control         nothing (same shape)   —

B and E differ more than they first appear. Distillation compresses the teacher
into the student's parameters, so the student's capacity is the ceiling — a small
vertical model can only absorb so much. Representation transfer hands the
knowledge over as an *input*, so the student never has to memorise it. In
production those features are precomputed offline and read from a table, which is
why GEM can claim the transfer adds no inference overhead. This repo computes
them live and tests the quality claim only; the latency claim is not reproduced
and is not claimed.

The Student Adapter addresses something none of the others do. A foundation model
is retrained on a cadence measured in days; by the time its knowledge reaches a
vertical model, the world has moved. GEM's framing:

    VMs often suffer from stale supervision caused by delays in FM training and
    evaluation ... these outdated or misaligned signals can degrade the accuracy
    and adaptability of student models over time.

The adapter refines the teacher's outputs using the most recent ground truth. Its
premise is that staleness damages a teacher's *calibration* before its *ranking*,
which is why NE and AUC are reported separately, and why claim B is falsifiable:
if the adapter's advantage does not widen as the teacher ages, the premise is
wrong.

Two implementation details are load-bearing and silent when wrong.

**1. Both models must share an id space.** The teacher is trained on pooled data,
where ids are offset per domain; a student trained on solo-loaded data uses
different integers for the same product (Video_Games items are 3..26,356 solo and
17,888..44,241 pooled). Feeding a solo batch to a pooled teacher produces
confident nonsense and no error. Every arm therefore builds both models against
the pooled vocabularies, and the student simply never sees ids outside its own
domain — see `data/transfer_data.py`.

**Arm E needs a control, not a caveat.** Widening the student's head to accept the
teacher's representation also gives it more capacity: 165k trainable dense parameters
against 91k for the other arms. So `representation_shuffled` feeds the same teacher
features with the batch order permuted — identical marginal distribution, identical
parameter count, zero alignment to the sample. If E does not beat that control, its
advantage was capacity and not transferred knowledge.

**2. The adapter must be trained against ground truth alone.** If gradient from
the student's distillation loss reaches it, it learns to make the teacher agree
with the *student* rather than with reality — both sides nod at each other and the
distillation signal collapses toward zero. Nothing errors out. The separation is
enforced in `TransferModel.loss` and pinned by a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gem import MiniGEM, PredictionHead

ARMS = ["vm_only", "kd", "kd_adapter", "param_share", "representation",
        "representation_shuffled"]

ARM_LABELS = {
    "vm_only": "A  no transfer",
    "kd": "B  naive KD",
    "kd_adapter": "C  KD + Student Adapter",
    "param_share": "D  parameter sharing",
    "representation": "E  representation transfer",
    "representation_shuffled": "E' shuffled control",
}

# 只有分位数桶和时间桶在各域之间语义一致（"本品类里第 k 分位"），因此只有
# 它们值得共享。user/item/store/category 是加了偏移的 id，共享等于搬运噪声。
TRANSFERABLE_FIELDS = ("price_bucket", "rating_bucket", "popularity_bucket", "time_gap")


@dataclass
class TransferConfig:
    arm: str = "vm_only"
    alpha: float = 0.5           # 真实标签损失的权重；(1-alpha) 给蒸馏
    temperature: float = 2.0     # 软化教师输出，露出它对相对可能性的判断
    adapter_hidden: int = 32
    adapter_context: int = 8
    representation_dim: int = 64
    freeze_shared: bool = True   # 参数共享时冻结借来的张量

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {ARMS}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    @property
    def needs_teacher(self) -> bool:
        return self.arm != "vm_only"

    @property
    def uses_representation(self) -> bool:
        return self.arm in ("representation", "representation_shuffled")


def distillation_loss(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Binary distillation: match the softened target probability.

    The `T^2` factor keeps the gradient magnitude comparable across temperatures,
    so `alpha` means the same thing whatever `T` is set to.

    The target is detached here, and detached again at the `kd_adapter` call site.
    That duplication is deliberate and should not be tidied away: this guard makes
    the function safe for any caller, and the one at the call site states the
    intent where a reader is looking. Each is pinned by its own test, because a
    test that only catches both removals at once would pass while half the
    protection was gone.
    """
    soft_target = torch.sigmoid(target_logits.detach() / temperature)
    loss = F.binary_cross_entropy_with_logits(student_logits / temperature, soft_target)
    return loss * (temperature ** 2)


class StudentAdapter(nn.Module):
    """
    Correct a stale teacher's output using fresh ground truth.

    Deliberately small, and deliberately conditioned on more than the logit. A
    logit-only version would be Platt scaling, which can only shift and sharpen
    the whole distribution at once. Drift is not uniform — a teacher goes stale
    faster on popular items than on the long tail, and faster on cheap impulse
    purchases than on considered ones — so the correction is given the context it
    needs to vary. Each vertical model gets its own adapter, so domain is not a
    feature here: it is constant within a student.

    The output is a residual on the teacher's logit and the last layer is
    zero-initialised, so the module starts as an exact identity. "Do not correct"
    is the default, and any correction has to be earned from the fresh window.
    """

    def __init__(
        self,
        n_popularity_buckets: int,
        n_price_buckets: int,
        hidden: int = 32,
        context_dim: int = 8,
    ) -> None:
        super().__init__()
        self.popularity_emb = nn.Embedding(n_popularity_buckets, context_dim)
        self.price_emb = nn.Embedding(n_price_buckets, context_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + 2 * context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        teacher_logits: torch.Tensor,
        popularity: torch.Tensor,
        price: torch.Tensor,
        hist_len: torch.Tensor,
    ) -> torch.Tensor:
        logits = teacher_logits.detach()
        context = torch.cat([
            logits.unsqueeze(-1),
            torch.log1p(hist_len.float()).unsqueeze(-1),
            self.popularity_emb(popularity),
            self.price_emb(price),
        ], dim=-1)

        scale, shift = self.net(context).chunk(2, dim=-1)
        return logits * (1.0 + scale.squeeze(-1)) + shift.squeeze(-1)


class RepresentationBridge(nn.Module):
    """Project the teacher's pre-head representation into the student's head."""

    def __init__(self, teacher_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.project = nn.Sequential(
            nn.Linear(teacher_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, teacher_repr: torch.Tensor) -> torch.Tensor:
        return self.project(teacher_repr.detach())


def share_parameters(student: MiniGEM, teacher: MiniGEM,
                     freeze: bool = True) -> list[str]:
    """
    Copy the teacher's transferable embedding tables into the student.

    Returns the field names actually shared, so a run reports what happened
    rather than assuming it. A dimension mismatch raises: silently skipping would
    leave this arm identical to the baseline while still being labelled
    parameter sharing, which is the worst outcome available — a null result that
    reads as a finding.
    """
    shared: list[str] = []
    for name in TRANSFERABLE_FIELDS:
        if name not in student.embeddings.tables or name not in teacher.embeddings.tables:
            continue
        target, source = student.embeddings.tables[name], teacher.embeddings.tables[name]
        if target.weight.shape != source.weight.shape:
            raise ValueError(
                f"cannot share {name!r}: teacher {tuple(source.weight.shape)} vs "
                f"student {tuple(target.weight.shape)}. Parameter sharing needs the "
                f"student to use the teacher's embedding width — scale the student's "
                f"depth and inner widths instead of its dim."
            )
        with torch.no_grad():
            target.weight.copy_(source.weight)
        target.weight.requires_grad_(not freeze)
        shared.append(name)

    if not shared:
        raise ValueError("parameter sharing selected but no field was shareable")
    return shared


class TransferModel(nn.Module):
    """
    A student, plus whatever the chosen arm attaches to it.

    The teacher is held as a submodule only so `.to(device)` moves it. It is
    frozen at construction, and `train()` is overridden to keep it in eval mode no
    matter what the training loop does — dropout inside a teacher would make its
    targets random noise from the student's point of view.
    """

    def __init__(
        self,
        student: MiniGEM,
        teacher: MiniGEM | None,
        config: TransferConfig,
        n_popularity_buckets: int = 19,
        n_price_buckets: int = 19,
    ) -> None:
        super().__init__()
        self.config = config
        self.student = student
        self.shared_fields: list[str] = []

        if config.needs_teacher and teacher is None:
            raise ValueError(f"arm {config.arm!r} needs a teacher")
        self.teacher = teacher
        if teacher is not None:
            teacher.eval()
            teacher.requires_grad_(False)

        self.adapter = None
        self.bridge = None

        if config.arm == "kd_adapter":
            self.adapter = StudentAdapter(n_popularity_buckets, n_price_buckets,
                                          config.adapter_hidden, config.adapter_context)

        elif config.uses_representation:
            teacher_width = teacher.body.n_out * teacher.config.dim
            self.bridge = RepresentationBridge(teacher_width, config.representation_dim)
            # 学生的头要多吃一份输入，所以换掉——这是换接口，不是加宽
            student.head = PredictionHead(
                student.body.n_out * student.config.dim + config.representation_dim,
                student.config.head_hidden,
                student.config.dropout,
            )

        elif config.arm == "param_share":
            self.shared_fields = share_parameters(student, teacher, config.freeze_shared)

    def train(self, mode: bool = True):          # noqa: A003
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Every logit the loss needs, so the loss stays explicit about which
        tensors carry gradient.
        """
        arm = self.config.arm

        if self.config.uses_representation:
            with torch.no_grad():
                teacher_repr = self.teacher.embedding_output(batch)
                if arm == "representation_shuffled":
                    # 同样的边缘分布、同样的参数量，但和样本完全不对齐。
                    # E 若只是靠更宽的头取胜，这一支会打成平手。
                    teacher_repr = teacher_repr[torch.randperm(
                        teacher_repr.size(0), device=teacher_repr.device)]
            student_repr = self.student.embedding_output(batch)
            fused = torch.cat([student_repr, self.bridge(teacher_repr)], dim=-1)
            return {"student": self.student.head(fused)}

        out = {"student": self.student(batch)}

        if arm in ("kd", "kd_adapter"):
            with torch.no_grad():
                out["teacher"] = self.teacher(batch)

            if self.adapter is not None:
                feats = self.teacher.item_features(batch["item"])
                out["adapter"] = self.adapter(
                    out["teacher"], feats["popularity_bucket"],
                    feats["price_bucket"], batch["hist_len"])

        return out

    def loss(self, out: dict[str, torch.Tensor],
             labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Total loss plus its parts, for logging.

        The adapter's term is separate on purpose: it is fit against ground truth
        alone, and the student distils from `adapter(...).detach()`. See the
        module docstring for why mixing those two gradients is fatal and silent.
        """
        cfg = self.config
        task = F.binary_cross_entropy_with_logits(out["student"], labels)
        parts = {"task": task}

        if cfg.arm == "kd":
            parts["kd"] = distillation_loss(out["student"], out["teacher"], cfg.temperature)
            parts["total"] = cfg.alpha * task + (1 - cfg.alpha) * parts["kd"]

        elif cfg.arm == "kd_adapter":
            parts["adapter_fit"] = F.binary_cross_entropy_with_logits(out["adapter"], labels)
            parts["kd"] = distillation_loss(out["student"], out["adapter"].detach(),
                                            cfg.temperature)
            parts["total"] = (cfg.alpha * task
                              + (1 - cfg.alpha) * parts["kd"]
                              + parts["adapter_fit"])

        else:
            parts["total"] = task

        return parts

    @property
    def n_trainable(self) -> int:
        """Excludes the frozen teacher, and embeddings, which track the vocabulary."""
        embedding_ids = {id(p) for p in self.student.embeddings.parameters()}
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad and id(p) not in embedding_ids)
