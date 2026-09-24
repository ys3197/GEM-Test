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
| **A** | InterFormer's interleaved structure beats pool-then-interact, which "risks losing critical engagement signals" | Fix parameter count, swap only the structure, sweep depth | M2 |
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

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m data.download           # ~7.9 GB across four domains
python -m data.probe              # density check — run this before trusting a domain
python -m data.prepare            # k-core, temporal ordering, parquet
python -m analysis.padding_waste  # reproduces the figure above
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
analysis/
  padding_waste.py          M0 result
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
