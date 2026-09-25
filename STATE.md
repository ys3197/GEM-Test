# Project state — handoff notes

Working notes for picking this up cold. Last updated **2026-09-25**, after the M3
transfer work (commits `128615a` → `8e2a6d1`, all pushed to
`github.com/ys3197/GEM-Test`).

The README is the public account of the project. This file is the working one: the
environment traps, the decisions and the measurement behind each, what is unresolved,
and the exact commands to resume.

---

## 0. Environment traps — read this first

**There are two Pythons on this machine and only one has CUDA.**

```
.venv/Scripts/python.exe    torch 2.6.0+cu124    CUDA True     <- use this
python  (system, PATH)      torch 2.14.0+cpu     CUDA False
```

The system one is `C:\Users\yikma\AppData\Local\Programs\Python\Python311`. A bare
`python train_transfer.py` silently runs on CPU at roughly 1/8 speed. This already
cost one full run: it completed, produced plausible numbers, and only the
`pin_memory ... no accelerator is found` warning gave it away. Always:

```bash
.venv/Scripts/python.exe -u train_transfer.py ...
```

The `-u` matters too. Without it, stdout is block-buffered when redirected, so a
long run shows nothing until it exits.

**Do not pipe a training run through `tail` or `grep`.** The pipeline's exit code is
the last stage's, so a Python traceback reports as exit 0. One crash was hidden this
way. Redirect to a log file and check the code explicitly:

```bash
.venv/Scripts/python.exe -u train_transfer.py ... > runs/x.log 2>&1 || { tail -30 runs/x.log; exit 1; }
```

**Teacher checkpoints are cached and the cache key must cover everything upstream.**
`train_transfer.py` hashes the teacher's config, epochs, lr, negatives, split date,
domains, seed **and `SAMPLE_SCHEME`**. That last one was added after a run silently
loaded a teacher trained on the *old, easier* task. If you change anything about how
samples are built, bump `SAMPLE_SCHEME`. Existing checkpoints in `runs/teachers/`:

```
fm_k0_e84939d6.pt   stale - trained under v1 (unrestricted negatives). Safe to delete.
fm_k0_65945397.pt   current - v2, domain-restricted negatives, 2 epochs, k=0
```

---

## 1. What the project is

Testing the **mechanism** claims behind Meta's GEM on one consumer GPU with public
Amazon Reviews 2023 data. Not a reproduction — GEM trains on thousands of GPUs. The
split that makes it tractable:

- **Mechanism claims** are about proportions, so they survive scaling down:
  interleaving beats pooling; a Student Adapter beats naive KD; quality is log-linear
  in compute.
- **Infrastructure claims** are not attempted and the README says so: 5D parallelism,
  MXFP8, SM-free collectives all solve problems that do not exist on one GPU.

Three claims, current status:

| | claim | status |
|---|---|---|
| **A** | InterFormer's interleaved structure beats pool-then-interact | **deferred** — see §5 |
| **B** | A Student Adapter beats naive KD when the teacher is stale | M3 built, **M4 pending** |
| **C** | Quality scales log-linearly with compute | M5, not started |

**B is the centre of gravity.** A and C have close analogues in public literature; the
Student Adapter does not, and it is the only published mechanism addressing teacher
*staleness* rather than teacher *accuracy*.

---

## 2. Decisions, and the measurement behind each

The project's method is **measure before building**. Every entry below is a case where
the obvious choice would have silently produced numbers that looked fine.

### Domain selection — by measured k-core survival, not file size

First selection was by file size (small = fast to iterate). `Digital_Music` collapsed
from 130,434 interactions to **157 rows / 20 users** under k-core(5). `data/probe.py`
exists to check this before anything is built on a domain.

| domain | rows/user | users after k-core(5) | |
|---|---|---|---|
| All_Beauty | 1.11 | 357 | unusable |
| Handmade_Products | 1.13 | 89 | unusable |
| Digital_Music | 1.29 | 20 | unusable |
| Software | 1.88 | 149,625 | **selected** |
| Video_Games | 1.67 | 98,906 | **selected** |
| Musical_Instruments | 1.71 | 59,941 | **selected** |
| Industrial_and_Scientific | 1.51 | 54,567 | **selected** |

Small Amazon categories are sparse **in users**, not just small in bytes. The four
chosen domains also differ in buyer type and repeat cadence (digital goods / digital
entertainment / physical hobby goods / B2B consumables) — if they were too similar,
"cross-domain learning" and "one pooled domain" would be the same thing and the
transfer experiment would lose its control.

### `MAX_SEQ_LEN = 32` — measured, not guessed

Originally 128, which made a "93% padding waste" figure that was an artefact of the
choice. Median sequence length here is **6–7 events**; only 0.03% of users hit 128.

| cap | users covered | waste, pad-to-fixed | waste, pad-to-batch-max |
|---|---|---|---|
| 8 | 71.0% | 17.9% | 17.9% |
| 16 | 93.0% | **51.0%** | 50.5% |
| **32** | **98.6%** | 73.8% | 67.5% |
| 64 | 99.8% | 86.6% | 72.4% |
| 128 | 100.0% | 93.3% | 73.7% |

Conclusion recorded in the README as a **negative** result: Meta's "up to 50% wasted"
reproduces almost exactly, but at `cap=16` — it is a property of a cap against a
length distribution, not of jagged data in general. Jagged-tensor kernels would be
chasing the 6 points between pad-to-fixed and pad-to-batch-max. Choosing the cap is
worth ~56 points. So batch-max padding, `cap=32`, and no jagged machinery.

### Two causality bugs in the sample builder, both silent

1. **Timestamp unit mismatch.** `user_ts` came back from parquet at millisecond
   resolution while `pd.Timestamp.value` is nanoseconds — a 1000× mismatch that put
   *every* event in the train split. The first self-check missed it because the
   boundary came from a quantile of the same array. Fixed with
   `to_numpy(dtype="datetime64[ns]").astype("int64")`, pinned by a test that asserts
   the dates land between 1995 and 2030.
2. **Tie leakage, 1.83% of samples.** Events sharing the candidate's exact timestamp
   were entering its history. Fixed with a `hist_end` run-boundary array;
   `temporal_split` now filters on `hist_end > start`, not `pos > 0`.

Both have tests. These are the load-bearing tests in the suite: a leak here does not
crash, it looks like unusually good results.

### Pooling — ids offset per domain, buckets shared

Measured first:

```
users in 2+ domains    15,135 / 346,190  =  4.37%
items in 2+ domains    0                    (Amazon categories partition ASINs)
```

So cross-domain transfer cannot happen via shared entities. Ids are offset per domain;
pooling gives the foundation model more data for its **dense weights**, not a shared
entity space it does not have. Two channels remain and they are what M3/M4 test:

- dense weights (how price interacts with popularity, how relevance decays)
- **quantile bucket semantics** — `price_bucket=7` means "expensive for its category"
  in every domain, because cuts were made on within-domain quantiles. Those four
  vocabularies (`price_bucket`, `rating_bucket`, `popularity_bucket`, `time_gap`) are
  shared; `store` and `category` are offset like ids.

Pooled totals: **3,151,266 events / 363,039 users**. Id ranges, verified disjoint:

```
Software                  [    3,  17,887]
Video_Games               [17,888,  44,241]
Musical_Instruments       [44,242,  69,769]
Industrial_and_Scientific [69,770,  96,998]
```

---

## 3. M3 — the four things that would have produced fake numbers

This is the most recent work and the part most worth re-reading.

### 3.1 Teacher and student must share an id space

```
Video_Games item ids, loaded solo   :     3 .. 26,356
Video_Games item ids, inside pooled : 17,888 .. 44,241
```

A pooled-trained teacher scoring a batch built by `load_domain` looks up **entirely
unrelated products**, confidently, with no error. Fix: both models build against
pooled vocabularies, and the student simply never sees ids outside its own domain
(`domain_positions`). Cost is an inflated student embedding table — rows never indexed
get no gradient, and `n_dense_parameters` (what arms are compared on) excludes
embeddings anyway.

### 3.2 Pooling made the task *easier*

Negatives were drawn uniformly from the whole pooled catalogue, but each user belongs
to one domain:

| domain | own items | share of catalogue | negatives from a foreign domain |
|---|---|---|---|
| Software | 17,885 | 18.4% | **81.6%** |
| Video_Games | 26,354 | 27.2% | **72.8%** |
| Musical_Instruments | 25,528 | 26.3% | **73.7%** |
| Industrial_and_Scientific | 27,229 | 28.1% | **71.9%** |

Three quarters of the time the model only had to answer "is this item even in a
category this user shops in", which the category embedding answers immediately.
Visible as **AUC 0.988 pooled against 0.932 on the same solo data in M1**, with all
five arms inside a 1.5% band at the ceiling. Fixed by
`data.pooled.user_negative_ranges`; AUC returns to ~0.93.

Not a bug in the sampler — it did what it was told. Pooling changed what "a random
other item" means and nothing in the code had a reason to notice.

### 3.3 `k` in days does not survive the move from Meta's data to Amazon's

The original M4 design was teacher on `[.., T−k]`, student on `[T−k, T]`, sweeping
`k ∈ {0, 3, 7, 14, 30}` days. Measured:

| | |
|---|---|
| student positions at `k = 0` | **0** — `[T, T]` is zero-width |
| student positions at `k = 3` | **355 / 483 / 416 / 438** across the four domains |
| teacher data removed at `k = 7` | **0.14%** |
| teacher data removed at `k = 30` | **0.54%** |

Two independent failures. The design tied the student's *data volume* to the teacher's
*lag*, which are independent in production; and a week of Amazon reviews is 0.14% of
the teacher's training set, so the sweep would have returned five near-identical
numbers — a false negative dressed as a result. Meta's week is enormous and its
creatives turn over fast; here the catalogue barely moves.

**Corrected design.** Student window fixed; teacher slides.

```
 teacher  ├──── sliding 730d ────┤        k = 0, 90, 365, 730, 1095
 student                  ├──── 365d ────┤
 valid                                   ├─90d─┤          epoch selection
 eval                                          ├──── 180d ────┤   reported once
                                         T = 2022-01-01
```

| `k` | teacher window | teacher positions | vs `k=0` |
|---|---|---|---|
| 0 | 2020-01-02 .. 2022-01-01 | 429,890 | — |
| 90 | 2019-10-04 .. 2021-10-03 | 460,611 | +7.1% |
| 365 | 2019-01-02 .. 2021-01-01 | 499,802 | +16.3% |
| 730 | 2018-01-02 .. 2020-01-02 | 523,716 | +21.8% |
| 1095 | 2017-01-02 .. 2019-01-02 | 522,439 | +21.5% |

Per-domain positions (invariant in `k`, which is the point):

| domain | student (365d) | valid (90d) | eval (180d) |
|---|---|---|---|
| Software | 49,807 | 10,108 | 18,053 |
| Video_Games | 44,198 | 10,625 | 19,369 |
| Musical_Instruments | 55,617 | 11,488 | 23,116 |
| Industrial_and_Scientific | 52,985 | 12,444 | 28,092 |

**Why sliding rather than expanding.** An expanding teacher (everything up to `T−k`,
what a production FM actually trains on) loses 28% of its data at `k = 1095`, so
staleness and volume move together and a degradation cannot be attributed to either.
Sliding keeps volume flat and *slightly higher* for older windows, because these
categories were marginally busier in 2018 than 2021 — so the confound runs **against**
the hypothesis. Both modes exist; `teacher_days=None` selects expanding, and running
it as the "realistic variant" is a reasonable M4 extension.

### 3.4 Selecting the epoch and reporting on the same window — my bug

The first trainer picked the best epoch by NE on the eval window and then reported that
same number: model selection on the test set. The bias is **not constant across arms**
— a noisier trajectory gets a luckier minimum, so the contaminated metric rewards
instability, which is one of the axes distillation is supposed to change. Now there is
a separate 90-day `valid` window for selection and `eval` is scored once with the
selected weights. The gap is large: arm A, seed 42 — valid NE 0.4757, eval NE 0.5966.

---

## 4. Results so far

### 4.1 Seed noise is the binding constraint

Holding the teacher, data and everything else fixed and changing only the student's
initialisation seed:

```
on the corrected task:   A  seed 42  eval NE 0.5966      spread 0.053  (~9%)
                         A  seed 1337 eval NE 0.5438

on the old easy task:    A 0.2288 / C 0.2356  (+2.99%)  seed 42, GPU bf16
                         A 0.2357 / C 0.2355  (-0.10%)  seed 1337, GPU bf16
                         A 0.2315 / C 0.2285  (-1.29%)  seed 42, CPU fp32
```

The last two lines are the same seed and the same teacher, differing only in numeric
precision — and the winner flips. **No single run can be read as a result.**
`analysis/transfer_table.py` reports mean and spread and prints `resolved? no` when a
gap falls inside the noise.

### 4.2 Six arms, `Software`, `k = 0`

4 epochs, selection on `valid`, reported once on `eval`. Teacher alone on `eval`:
**NE 0.5574, AUC 0.9260**.

| arm | trainable dense | seed 42 | seed 1337 | AUC (s42) | same direction? |
|---|---|---|---|---|---|
| **A** no transfer | 91,333 | 0.5966 | 0.5438 | 0.9127 | baseline |
| **B** naive KD | 91,333 | 0.5690 ✓ | 0.5527 ✗ | 0.9258 | **no** |
| **C** KD + Student Adapter | 92,311 | **0.7767** | **0.7613** | 0.9276 | yes, consistently worse |
| **D** parameter sharing | 91,333 | 0.5551 ✓ | 0.4749 ✓ | 0.9294 | yes, consistently better |
| **E** representation transfer | 165,253 | 0.6194 | — | 0.9303 | n=1 |
| **E'** shuffled control | 165,253 | 0.5519 | — | 0.9213 | n=1 |

Valid NE for reference (selection basis): seed 42 — A 0.4757, B 0.4184, C 0.5286,
D 0.4315, E 0.4887, E' 0.4804. Seed 1337 — A 0.4771, B 0.4136, C 0.5142, D 0.4078.

**Readings, descending confidence:**

1. **C is 30% worse than no transfer, consistently, and it is not a selection
   artefact** — the same ordering holds on `valid`. Hypothesis: the adapter turns
   distillation from a regulariser into an *overfitting amplifier*. It fits ground
   truth on the student's own training window and fits it better than the student does
   (`adapter_fit` 0.096 against `task` 0.15), so the distillation target becomes a
   high-fidelity copy of the training labels. At `k = 0` this is all cost — the teacher
   is not stale, so there is nothing to correct and the training labels are just
   relayed twice.

   **This is not evidence against claim B.** The claim is that C's advantage *widens
   with `k`*, so C at its worst when the teacher is current is the baseline the sweep
   needs. What it does expose is a question the GEM post leaves open — it says the
   adapter uses "the most recent ground-truth data", and if that is the student's own
   training window the amplification is structural. **Untested.** See §6.

2. **E loses to its own shuffled control** (0.6194 against 0.5519). At n=1 nothing is
   resolved, but there is no evidence E transfers anything useful, and its extra 74k
   parameters are unearned. This is why the control was added: without it, 0.6194 reads
   as "E is slightly behind" rather than "E cannot beat its own noise".

3. **D is the only arm consistent in direction across both seeds**, and the cheapest —
   copy four quantile-bucket embedding tables and freeze them. Matches the reason those
   four were the only shared vocabularies.

**NE and AUC disagree**, which matters because M4 rests on it: C keeps a respectable
AUC (0.9276) while its NE collapses (0.7767); A has the worst AUC (0.9127) and a
middling NE. C's *ranking* is intact and its *calibration* is what broke — the
calibration-before-ranking split claim B's premise depends on, appearing as a
measurement rather than an argument.

### 4.3 M1 smoke test (claim A, for reference)

```
Software, 150k positions, 2 epochs, untuned, solo (not pooled)
  pooled        NE 0.5187   AUC 0.9322   36 s/epoch   0.60M dense
  interleaved   NE 0.5262   AUC 0.9224   56 s/epoch   0.67M dense
```

Not the experiment. Two blockers recorded: the variants differ by **12% in dense
parameters**, so as they stand the comparison measures capacity rather than structure;
and interleaved is slightly worse and 60% slower, which is the direction M0 predicted
(at a median of 7 events, "preserving the full sequence" has little to preserve).

---

## 5. What was running when this was written

A background run of 6 arms × 3 seeds was **stopped by the system while the session was
idle, because the machine was low on memory**. Not a failure of the run, and nothing in
it to debug.

State at the stop:

```
seed 42     complete   -> runs/transfer/m3_seed42.json written
seed 1337   A,B,C,D done, E in progress -> no JSON (log only: runs/m3_seed1337.log)
seed 7      never started
```

Seed 1337's numbers in §4.2 come from the log, not the JSON. **To finish, the missing
work is seed 1337's E and E', plus all of seed 7** — per-seed logs are independent, so
re-running seeds 1337 and 7 from scratch is simplest.

---

## 6. Open questions and next steps

**Immediate — finish M3's seeds** (~20 min, GPU; teacher is cached so it loads):

```bash
cd E:/projects/GEM-Test
for s in 1337 7; do
  .venv/Scripts/python.exe -u train_transfer.py --domain Software --k 0 \
    --teacher-epochs 2 --student-epochs 4 --student-seed $s --tag m3_seed$s \
    > runs/m3_seed$s.log 2>&1 || { echo "seed $s FAILED"; tail -30 runs/m3_seed$s.log; break; }
done
.venv/Scripts/python.exe -m analysis.transfer_table
```

**Untested hypothesis worth resolving before M4 — does the adapter need held-out
data?** §4.2 reading 1 says C's damage comes from the adapter fitting the student's own
training window. The test is cheap: split the student's 365-day window, fit the adapter
on one part and train the student on the other, and see whether C's 30% penalty at
`k = 0` shrinks. If it does, the adapter needs its own slice and M4 should use that
version — otherwise M4 measures a self-inflicted wound at every `k`, which would mask
the staleness effect it is looking for. **This is the single highest-value next
experiment**, because it decides whether M4's C arm is even implemented correctly.

**M4 — the staleness sweep.** 4 domains × 5 `k` values × 6 arms × ≥3 seeds = 360 student
runs plus 5 teachers. At ~60 s per student that is roughly 6–7 GPU-hours. Worth running
`Software` across all `k` at 3 seeds first (90 runs, ~1.5 h) to see whether the `k`
trend exists at all before committing to the full grid.

```bash
.venv/Scripts/python.exe -u train_transfer.py --domain Software \
  --k 0 90 365 730 1095 --student-seed 42 --tag m4_sw_s42 > runs/m4_sw_s42.log 2>&1
```

The falsification criterion, stated so it cannot be moved later: **claim B holds only
if C's gap over B widens monotonically with `k`.** A flat or noisy relationship is a
negative result and gets reported as one.

**M5 — scaling law.** Five model sizes via `MiniGEMConfig.scaled`, NE against FLOPs.
Not started. Note the constraint from arm D: `scaled()` moves `dim`, which breaks
parameter sharing, so M5 and the transfer arms cannot share a config sweep.

**M2 / claim A — deferred, deliberately.** M1 measured why it is the weaker bet: at a
median of 7 events per user, "preserving the full sequence" has little to preserve, so
the likely outcome is a null result about *this data* rather than about the structure.
To make it conclusive it needs the 12% dense-parameter gap closed first. The README
states this as a choice rather than an omission.

**M6 — README polish, figures for M3/M4, CI.** The README already carries M0/M1/M3;
what is missing is a figure for the `k` sweep and for the arm comparison.

---

## 7. File map

```
config.py                   domains, k-core, sequence cap, transfer windows, seeds
                            - every constant has the measurement that set it in a comment
data/
  download.py               raw jsonl from HuggingFace
  probe.py                  density / k-core survival - run before trusting a domain
  prepare.py                k-core filter, temporal sort, parquet
  features.py               vocabularies and quantile buckets
  dataset.py                causal sample construction, jagged batching, negatives
  pooled.py                 four domains in one id space + user_negative_ranges
  transfer_data.py          teacher / student / valid / eval windows; the staleness dial
models/
  embeddings.py             shared-width field tables, sequence masking
  wukong.py                 stacked factorization machines (non-sequence tower)
  sequence.py               event model + candidate-keyed attention pooling, O(M*N)
  interformer.py            the two structures claim A compares
  gem.py                    assembly; one config drives every width
  transfer.py               the six arms (five techniques + shuffled control)
train.py                    single-model training loop, normalized entropy, AUC
train_transfer.py           pooled FM once (cached), then students under each arm
analysis/
  padding_waste.py          M0 figure
  transfer_table.py         cross-seed aggregation with resolved?/no
tests/                      44 guardrails, no GPU needed, skip cleanly without data
runs/                       gitignored - checkpoints, logs, result JSON
figures/                    committed, the README renders them
```

**Metric convention.** Headline metric is **normalized entropy** (`NE = logloss(model) /
logloss(base rate)`), which is what Meta reports. Raw log loss is not comparable across
these runs because the positive rate is set by the negative sampling ratio. AUC sits
beside it deliberately: AUC is rank-only, NE also responds to calibration, and claim B
depends on exactly that split.

---

## 8. Commit history worth knowing

```
8e2a6d1  README: record M3's first numbers, and what they do not support
406b72c  M3: add the controls the first run showed were missing
38af625  M3: five FM-to-VM transfer arms, and three measurements that reshaped M4
128615a  M3 (1/2): pooled multi-domain data, shaped by a measurement
8a41684  README: cover M1, and record the two things that must change before claim A is tested
```

---

## Appendix — unrelated open items on the other project

`E:/projects/Travel-Agent` (pushed to `github.com/ys3197/Travel-Agent`) has two
loose ends from the same session, noted here only so they are not lost:

- `wheelchair_accessible` is null for all 32 enabled POIs in
  `agents/tools/poi_table.json`, so the accessibility pre-filter is inert.
- The GitHub repo description and topics still need setting via the web UI.
