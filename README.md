# GEM-Test

Testing three of the mechanism claims behind **Meta's Generative Ads Recommendation
Model (GEM)** on a single consumer GPU, using public Amazon Reviews data.

The point is not to reproduce GEM. It cannot be reproduced — it trains on thousands of
GPUs. The point is that GEM's published claims separate cleanly into two kinds, and only
one kind depends on scale:

| | Examples | Testable here? |
|---|---|---|
| **Mechanism claims** | interleaving beats pooling; a Student Adapter beats naive distillation; performance is log-linear in compute | **Yes** — these are claims about *proportions*, not absolute size |
| **Infrastructure claims** | 5D parallelism, MXFP8 kernels, SM-free collectives | **No, and deliberately not attempted** — the problems they solve (cross-zone bandwidth, thousand-GPU straggler skew) do not exist on one GPU |

So this repo tests the first kind and says so plainly. Negative results count.

---

## The three claims

| | Claim | How it is tested | Status |
|---|---|---|---|
| **A** | InterFormer's interleaved structure beats pool-then-interact, which "risks losing critical engagement signals" | Fix parameter count, swap only the structure, sweep depth | M2 — **at risk**, see below |
| **B** | A **Student Adapter** — a light module that refines a teacher's outputs using fresh ground truth — beats naive knowledge distillation when the teacher is stale | Deliberately train the teacher on data up to `T − k`, the student on `[T − k, T]`, sweep `k ∈ {0, 3, 7, 14, 30}` days | M4 |
| **C** | Performance scales log-linearly with compute | Five model sizes, NE vs FLOPs | M5 |

**B is the centre of gravity.** A and C have close analogues in the public literature;
the Student Adapter does not, and it is the only published mechanism that addresses
teacher *staleness* rather than teacher *accuracy*. Its whole premise is falsifiable at
small scale: if the gap between adapter and naive KD does not widen as the teacher ages,
the mechanism does not do what it claims.

---

## M0 — what the data actually looks like

Two findings, both of which changed the plan.

### 1. File size says nothing about whether a domain has sequences

Amazon Reviews is sparse in a way that is invisible until measured. The first domain
selection was made on file size — small categories, fast to iterate. It collapsed:

```
Digital_Music:  130,434 interactions / 100,952 users  =  1.29 per user
after k-core(5):  157 rows, 20 users
```

`data/probe.py` measures this properly before anything is built on it:

| domain | rows/user | users surviving k-core(5) | |
|---|---|---|---|
| All_Beauty | 1.11 | 357 | unusable |
| Handmade_Products | 1.13 | 89 | unusable |
| Digital_Music | 1.29 | 20 | unusable |
| **Software** | 1.88 | **149,625** | selected |
| **Video_Games** | 1.67 | **98,906** | selected |
| **Musical_Instruments** | 1.71 | **59,941** | selected |
| **Industrial_and_Scientific** | 1.51 | **54,567** | selected |

Small categories are sparse *in users*, not only in bytes — one-off purchases leave no
history to model. The four selected domains also differ in buyer type and repeat cadence
(digital goods / digital entertainment / physical hobby goods / B2B consumables), which
matters: if the domains are too similar, "cross-domain learning" and "one pooled domain"
become the same thing and the transfer experiment loses its control.

### 2. Padding waste is real here, but the lever is the cap — not jagged kernels

![padding waste](figures/m0_padding_waste.png)

Meta reports that padding jagged sequences "would waste up to 50% of compute," on data
whose sequences run "from hundreds to tens of thousands of tokens per sample." Ours have
a **median of 6–7 events**. Measured across the four domains:

| cap | users covered | waste, pad-to-fixed | waste, pad-to-batch-max |
|---|---|---|---|
| 8 | 71.0% | 17.9% | 17.9% |
| **16** | 93.0% | **51.0%** | 50.5% |
| 32 | 98.6% | 73.8% | 67.5% |
| 64 | 99.8% | 86.6% | 72.4% |
| 128 | 100.0% | 93.3% | 73.7% |

Three things fall out:

- **Meta's 50% figure reproduces almost exactly — at `cap=16`.** It is not a property of
  jagged data in general; it is a property of a particular cap against a particular
  length distribution.
- **Batch-max padding barely helps, and then plateaus.** At `cap=8` it saves nothing; at
  `cap=32` it saves 6 points; beyond that it flattens near 74%, because the waste is
  dominated by within-batch spread rather than by the cap. Jagged kernels would be
  chasing those 6 points.
- **Choosing the cap is worth ~56 points**, from 73.8% at `cap=32` down to 17.9% at
  `cap=8` — at the cost of truncating 29% of users.

So on this dataset the honest conclusion is a negative one: **jagged-tensor machinery is
not where the compute goes.** Picking `max_len` deliberately is. The repo uses `cap=32`
(covering 98.6% of users) and spends its complexity budget elsewhere.

---

## M1 — the model runs

Both structures claim A compares are built, assembled behind one config, and
training converges on a single 8 GB GPU.

```
Software, 150k positions, 2 epochs, untuned

  pooled        NE 0.5187   AUC 0.9322   36 s/epoch   0.60M dense params
  interleaved   NE 0.5262   AUC 0.9224   56 s/epoch   0.67M dense params
```

This is a smoke test, not the experiment. What it establishes is narrower: the
pipeline learns (NE well below 1, AUC in a sane range for CTR), and both
variants overfit by epoch 2, so the real sweep needs early stopping and more
data.

**Normalized entropy** is the headline metric because raw log loss is not
comparable across these runs — the positive rate is fixed by the negative
sampling ratio, which differs between experiments. Dividing by the entropy of
the base rate removes that. AUC sits beside it on purpose: AUC is rank-only,
NE also responds to calibration, and M4 depends on exactly that split, since a
stale teacher degrades a student's calibration well before its ranking.

Two things must be fixed before the claim-A comparison means anything:

- **The variants differ by 12% in dense parameters.** Compared as they stand,
  the result would measure capacity rather than structure.
- **The interleaved variant is currently slightly worse and 60% slower**, which
  is the direction M0 predicted: at a median of 7 events per user, "preserving
  the full sequence" has little to preserve. Whether that is a fact about this
  dataset or about the structure is precisely what M2 has to separate — and if
  it cannot, that is a finding about what public data can support, reported as
  such.

```bash
python train.py --domain Software --variant pooled --epochs 2
python train.py --domain Software --variant interleaved --scale 0.5
```

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m data.download           # ~7.9 GB across four domains
python -m data.probe              # density check — run this before trusting a domain
python -m data.prepare            # k-core, temporal ordering, parquet
python -m data.features           # vocabularies and quantile buckets
python -m analysis.padding_waste  # reproduces the figure above

pytest tests/ -q                  # 17 guardrails; skips cleanly without data
```

Models (M1 onward) additionally need a CUDA build of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Raw and processed data are gitignored; every table and figure in this README regenerates
from the commands above.

---

## Layout

```
config.py                   domains, k-core thresholds, sequence cap — all measured, not guessed
data/
  download.py               fetch raw jsonl from HuggingFace (no loading script, by choice)
  probe.py                  density and k-core survival, per domain
  prepare.py                k-core filter, temporal sort, parquet output
  features.py               categorical encoding: vocabularies and quantile buckets
  dataset.py                causal sample construction, jagged batching
models/
  embeddings.py             shared-width field tables, sequence masking
  wukong.py                 stacked factorization machines (non-sequence tower)
  sequence.py               event model + candidate-keyed attention pooling, O(M*N)
  interformer.py            the two structures claim A compares
  gem.py                    assembly; one config drives every width
train.py                    training loop, normalized entropy, AUC
analysis/
  padding_waste.py          M0 result
tests/                      causality and architecture guardrails — no GPU needed
figures/                    committed — the README renders them
```

---

## Sources

- [Sequence learning: A paradigm shift for personalized ads recommendations](https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/) — Meta, Nov 2024
- [Meta's Generative Ads Model (GEM): The Central Brain](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) — Nov 2025
- [GEM Training: How Meta Doubled the Efficiency of Its LLM-Scale Ads Foundation Model](https://engineering.fb.com/2026/08/03/ml-applications/training-gem-at-llm-scale-meta-ads-recommendation-foundation-model/) — Aug 2026
- [From User Sequences to Scaling Laws](https://engineering.fb.com/2026/08/05/ml-applications/from-user-sequences-to-scaling-laws-a-multi-stage-architecture-for-metas-ads-ranking/) — Aug 2026
- Data: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), McAuley Lab, UCSD

## License

MIT — see [LICENSE](LICENSE).
