# Phase 18: Highest-Impact Fixes Before Submission (v5)

## Changelog vs v5 (mid-execution amendment, 2026-05-06 14:39 UTC)

Original v5 pre-registered `n=24` over 9 K's (`{8,16,24,32,48,64,80,96,128}`).
Mid-execution finding: actual per-example wall-clock on Qwen2.5-7B at
32K with the 8-condition contrast set is ~6.3 minutes (vs preflight
estimate of 38s based on a smaller smoke). 9 K's at n=24 would take
~7.2 hours of GPU time, far over the 130-min ceiling and the 2-day
workshop deadline.

Reduced scope (committed before re-launch):

- `n=12` per partition × 3 partitions = 36 example-K cells per K.
- `K ∈ {32, 64, 80, 96, 128}` (5 K's).
- Same 8 binding/floor conditions as v5.
- Same Wilcoxon + HL CI + signed-rank TOST, but Holm correction now
  over a 15-test family (3 binding contrasts × 5 K's) instead of 27.
- HL singleton-collapse risk acknowledged: secondary paired-bootstrap
  CI reported alongside HL.

Pre-registration commit for the original v5 plan: `601d807`.
Pre-registration commit for this mid-execution amendment: forthcoming
(this edit + git commit hash to be cited in the paper alongside the
original).

## Changelog vs v4

Final verification panel (code, devil's advocate, statistical) drove
these v5 changes:

- **Burst-expansion ablation added** as a *blocking* new condition.
  Devil's advocate v2 flagged: RepairKV uniquely uses
  burst expansion `(L=2, R=20)` while every other condition in W1
  does point selection. If burst is the whole effect, the "lifecycle
  slot" framing is wrong. ~25 min, single condition at K=96 — too
  cheap to skip.
- **Statistical pre-registration tightened.** Drop "K=96 as single
  single confirmatory configuration" (soft HARKing — smoke informed the choice).
  Treat all 9 K's × 3 binding contrasts = 27 tests as confirmatory
  with **family-wise Holm-27**. Smoke Δ ≈ 0.5+ comfortably survives.
- **Wilcoxon tie handling pinned** to `zero_method='pratt'`,
  `method='exact'`.
- **TOST written out as two one-sided signed-rank tests** on shifted
  differences d ± 0.20 (not t-based, to match the primary test).
- **PageSummary-budgeted renamed to PageSummary-Quest-inspired**
  with explicit disclaimer that it is a two-stage cheap-then-fine
  scorer adapted to the lifecycle slot, not a Quest reproduction.
- **T_repair stability rule** added: Step 1 smoke must report σ/μ on
  per-example T_repair; if σ/μ > 0.10 the budget multiplier bumps
  from 1.05 to 1.20.
- **Step 6 descriptive, not gated.** Multi-model runs get no pass
  threshold. They are reported with HL CI and Wilcoxon p; the
  abstract qualifier "consistent on models tested" only fires if all
  three contrasts have HL lower bound > 0 on each model. Step 6 in
  the *weak-pass branch* is a downgrader (any reversal → fail
  branch), not an upgrader.
- **Code amber flags addressed:**
  - Per-layer chunked Refresh-K must divide by `scored_layer_count`,
    not `n_layers`.
  - `q2_query_s` is already an emitted runner field; v5 just adds
    explicit `cuda.synchronize()` and renames to `idlekv_q2_proj_ms`.
  - TM-Recompute-BM25 uses **one forward pass per BM25 window**, not
    interleaved non-contiguous position_ids in a single pass.
  - `host_pool_coverage < 1.0` guard at all four call sites.
  - Per-K warmup: randomize K order per example with `seed=index`.
- **Two attacks the panel raised that v5 does NOT defuse** are
  flagged for your call as the only remaining open questions
  (non-NIAH run + the "much smaller" wording).

## Changelog vs v3

Five-agent critique panel drove these v4 changes:

- **B1 fix.** Refresh-K's scoring is per-layer over a single matmul, so a wall-clock cap on the outer loop cannot stop "after some fraction of positions." `score_evicted_positions` is rewritten to break each layer's `Q · K^T` into position-chunks with a wall-clock check between chunks. Honest "scored some positions, did not score others." The paper §W4.1 explains this choice.
- **B2 fix.** `runtime_capacity.py` has a BF16→FP32 dtype upcast in `.to(...)` that silently demotes async H2D to blocking, and the existing `host_pool_coverage=1.0` was effectively faked by re-reading a 16K pool 64× at 1M. Both bugs likely affect Phase 17 Figure 6 numbers. Phase 18 W2 re-runs from scratch.
- **F decision (both reframe + add baseline).** Add a 6th condition: **PageSummary-budgeted** (Quest/ShadowKV-style two-stage scorer). Reframe the headline ratio claim from "RepairKV is 380× faster than full reselection" to "RepairKV approaches Refresh-K quality without the persistent low-rank index that page-summary scorers need."
- **A1 (multi-model) reduced to Llama appendix.** With a 2-day workshop deadline, full multi-model symmetric coverage (Llama + Mistral on the same task suite as Qwen) is moved to Phase 20. Phase 18 keeps a Llama-only K=96 4Q run as appendix-only descriptive evidence. The existing Phase 16 Mistral smoke stays in the appendix.
- **A2 (recency partition).** If easy at the 4Q-sweep launch time, add a 12→34 partition control. Result goes to appendix only.
- **Statistical tightening.** Wilcoxon signed-rank as primary test. Hodges-Lehmann CI on the median paired difference. TOST equivalence margin **fixed at 0.20** (under one Q2 needle, given 4Q's 0.25-grain grid). Intent-to-treat for budget overage. Git-commit this v4 plan before any headline run starts; cite the commit hash in the paper.
- **GPU optimization.** Reorder W2 concurrent-stage priority: chunk-N+1 H2D ∥ chunk-N scan is the highest-leverage overlap (~40-50% of scan wall-clock), higher than Q2-proj ∥ promote. Probe FA-2 and FA-3 on the recompute reference.
- **ArkVale, InfiniGen, EM-LLM, and ShadowKV-reselection are added to the §W4.1 novelty paragraph.** They are the closest published prior art for "evict, recall by query"; not citing them is a positioning hole.

---

## Goal (unchanged from v3)

Close five reviewer-style objections from the Phase 17 audit:

1. Sharpen the novelty boundary (post-compression, pre-resume lifecycle slot).
2. Wall-clock-matched recompute / reselection baseline.
3. Full repair-cost decomposition with `Q_2` projection.
4. Reduce overclaiming on agentic workloads.
5. Cost accounting (host store, scan/index, transfer, projection).

The single sentence we want a reviewer to walk away with, **revised** per the F-decision reframe **and** the post-Step-0d AdaptFM-reviewer note that softens the recompute clause to a one-sided cost claim (since TM-Recompute-BM25 is a single-K add-on, not part of the K-sweep):

> RepairKV approaches `Q_2`-aware reselection quality without paying for
> a persistent low-rank index or a Q_2-time full reselection scan, and
> without paying the wall-clock cost of full-prefix recompute.

This now has three falsifiable thresholds, pre-registered below:
- (a) CI-above-zero against TM-Recompute-BM25 (recompute camp).
- (b) CI-above-zero against PageSummary-budgeted (Quest/ShadowKV camp).
- (c) TOST equivalence with Condition A (full-cache reference) at margin 0.20.

Refresh-K-budgeted stays in but is repositioned: it is the
"matched-protocol full reselection scorer at the lifecycle slot" —
not a representative of the Q_2-aware retrieval literature.

---

## Hardware envelope and sequencing

1× RTX PRO 6000 Blackwell, 96 GB HBM, 8 vCPU, 64 GB RAM, single
tenant. Sequential GPU work in one tmux session.

**Why sequential, even for Step 6 multi-model runs.** Two 7-8B BF16
models *do* fit in 96 GB HBM, so co-loading is technically possible.
But every condition that depends on per-example `T_repair` (Refresh-
K-budgeted, TM-Recompute-BM25, PageSummary-Quest-inspired) needs the
RepairKV timing to be a clean single-tenant measurement. If two
models share the GPU, both models' RepairKV `T_repair` values are
contaminated by SM and PCIe contention from the other model. The
~70 minutes saved by parallelizing is not worth invalidating the
fairness contract that the entire W1 design rests on. Step 6 is
sequential: Llama runs to completion, then Mistral.

Budget shape:

| Stage | Wall-clock | What runs |
|---|---|---|
| **Step 0a — bug-fix pass** | ~30 min | Fix `runtime_capacity.py` dtype-upcast and pool-coverage bugs. Unit-test full-pool warmup at 1M. |
| **Step 0b — chunk-position Refresh-K rewrite** | ~45 min dev + ~10 min smoke | Rewrite `score_evicted_positions` to chunked matmul. Unit-test against the unchunked baseline (must match within fp16 tolerance when no cap fires). |
| **Step 0c — page-summary baseline dev** | ~45 min dev + ~10 min smoke | Implement two-stage page-summary scorer. Unit-test summary aggregation. |
| **Step 1 — W1 timing-only smoke** | ~15 min | 4Q n=4 K=64,96,128 with full instrumentation. Per-example T_repair distribution. |
| **Step 2 — W2 paper-quality run (post-bugfix)** | ~50 min | Stage decomposition; FA-2 + FA-3 prefill references; full-pool warmup at 1M and 4M. Headline = sequential; concurrent in table. |
| **Step 3 — W1 4Q K-sweep paper-quality run** | ~110 min | n=24, 6 conditions (incl. PageSummary-budgeted), K∈{8,16,24,32,48,64,80,96,128}. |
| **Step 4 — Audit gate (3-tier; pre-registered §W1.acceptance)** | <5 min | Strong / Weak / Fail. |
| **Step 5 — W1 6Q K-sweep** *(strong-pass only)* | ~75 min | Same shape, B_base=18432, 4Q→6Q. |
| **Step 5.5 — Burst-expansion ablation** *(after Step 5 for strong-pass; required before paper edits)* | ~25 min | RepairKV-no-burst at K=96 on 4Q, n=24. Single-condition run. |
| **Step 6 — Llama appendix** *(reduced from v5 multi-model after deadline reality check)* | ~45 min | Llama-3.1-8B-Instruct, 4Q K=96 only, 4 conditions (B_match, RepairKV, PageSummary-Quest-inspired, A), n=24. **Appendix-only, descriptive. Mistral deferred to Phase 20; existing Phase 16 Mistral smoke stays as-is in appendix.** |
| **Step 7 — Recency-favorable partition** *(if easy)* | ~30 min | 12→34 partition, 4Q K=96, n=24, 6 conditions. Appendix-only. |
| **Step 8 — W4 paper edits in green; awaiting your per-paragraph approval** | hours | Five paragraphs + abstract branch. |

Total GPU floor: ~4.5 hr (no multi-model). Ceiling: ~7.5 hr (full
plan including A1+A2). Paper edits asynchronous after numbers land.

**Failure handling:** Three-tier gate at Step 4 (4Q K=96):

- **Strong pass.** Run Step 5 + Step 6 + Step 7 in order.
- **Weak pass.** Run Step 6 *only* (multi-model robustness becomes
  even more important with weaker effects on 4Q). Skip Steps 5 and 7.
  Rewrite abstract per pre-registered weak-pass text in §W4.5.
- **Fail.** Stop. Rewrite abstract per fail text in §W4.5. No
  further headline runs without re-planning.

---

## W1 — Time-matched comparators (6-condition MVP)

### Conditions

| Condition | Source | Role |
|---|---|---|
| **Condition A** *(repo)* | Single forward pass over full uncompressed prefix + Q1 + Q2. | Quality ceiling. Doubles as Full-Recompute quality reference (both have unevicted KV). Anchors the TOST equivalence test. |
| **B_match** *(repo)* | Base SnapKV at budget `B_base + K`, no repair. | The no-repair floor. |
| **RepairKV** *(repo)* | Score evicted under exact `Q_2` → top-K with burst expansion → promote. | Thing under test. Defines per-example `T_repair`. |
| **Refresh-K-budgeted** *(new — chunk-position rewrite)* | Same exact-Q2 scorer over active+offloaded keys, but with the matmul broken into position-chunks and a wall-clock cap between chunks. Take top `B_base + K` from the subset that got scored before the cap; unscored positions tie at score=0 and fall back to the runner's existing position-ascending tiebreaker. | Matched-protocol full-reselection scorer. The Δ vs RepairKV isolates the lifecycle-operator vs naive-reselection effect at the same KV substrate, scorer, and compute budget. |
| **TM-Recompute-BM25** *(new)* | At Q2 time: BM25 over 256-token windows of original `D` against Q2 → greedy-pack windows that fit `T_repair − t_BM25` → prefill at original RoPE positions → append+retruncate to `B_base + K`. | The "just recompute once Q2 is known" objection in concrete form. |
| **PageSummary-Quest-inspired** *(new — two-stage scorer at lifecycle slot)* | Two-stage scorer: (1) precompute per-chunk max-key summaries during the post-Q1 compression step (free — done once); (2) at Q2 time, cheap-score Q2 against the summary vectors, pick top-N chunks; (3) within those chunks, full-score positions and pick top `K`; total time capped at per-example `T_repair`. **Disclaimer:** this is a two-stage cheap-then-fine scorer adapted to the lifecycle slot and the 18 ms budget, *not* a Quest reproduction. Quest operates per-decoding-step on a preserved cache with min/max page envelopes; this baseline operates once per pause boundary on an evicted host store with max-key summaries. ArkVale and InfiniGen are the closer prior art. | The "Quest/ShadowKV-style two-stage scoring would also fit in 18 ms" objection. The Δ vs RepairKV isolates RepairKV's burst expansion + content-aware top-K from a literature-standard cheap reselector. |
| **RepairKV-no-burst** *(new — burst-expansion ablation)* | RepairKV with `L=R=0` (point selection only, no burst expansion). Same scorer, same KV substrate, same `T_repair` budget — differs only in the burst expansion that v3 conflated with "lifecycle operator." | Falsifies or confirms whether the lifecycle-slot framing survives without burst. **Binding gate:** if `Score(RepairKV-no-burst) ≥ Score(PageSummary-Quest-inspired) − 0.05`, the lifecycle-slot framing holds. If RepairKV-no-burst collapses to ≈ PageSummary level, we reframe to "burst expansion at the lifecycle slot" — narrower contribution. |

We keep Random-K and Oldest-K from the existing runner since they are
free; they serve as content-agnostic floor controls in the figure.

The runner's existing `Refresh-K` (unbudgeted) is also reported as a
**ceiling reference** — it tells us the scorer's full-budget headroom
even though it is not a fair time-matched comparator.

### Per-example `T_repair`

For each example `i`:

1. Run RepairKV with full instrumentation and record
   `T_repair(i, K) = t_q2_proj(i) + t_scan(i) + t_topk(i) + t_promote(i) + t_merge(i)`.
   Note: `t_q2_proj` was hidden in the runner's setup before;
   v4 surfaces it as an explicit field (`idlekv_q2_proj_ms`) and adds
   it to the budget so TM baselines see the full repair envelope.
2. For each TM baseline, allocate budget = `T_repair(i, K) × 1.05`
   and stop the baseline at that wall-clock.
3. **Intent-to-treat reporting.** Examples that exceed the budget by
   >10% are *kept* in the headline number with a `budget_overage`
   flag; an "as-treated" sensitivity row excludes them.
4. Per-example T_repair distribution histogram in the appendix.

### Append+retruncate merge (no position-replace)

SnapKV evicts non-contiguously. Suffix-prefilled and chunk-prefilled
rows have no slot-correspondence to evicted positions. The merge
path for both TM-Recompute-BM25 and PageSummary-budgeted: append the
recomputed/promoted rows at their original RoPE positions, then
re-truncate to `B_base + K` under the active eviction policy.

### Pre-registered acceptance criteria (v5)

**Statistical tests (pinned):**

- **Primary test** for each pairwise contrast: paired **Wilcoxon
  signed-rank** on per-example score differences. Pinned exactly:
  `scipy.stats.wilcoxon(diffs, zero_method='pratt', method='exact')`.
  Pratt keeps zero-difference examples (mid-rank); the exact null
  distribution is used at n=24.
- **Companion CI:** **Hodges-Lehmann** at 95% (Walsh-averages
  inversion of the signed-rank test). At n=24 with a discrete grid,
  the achievable level may not be exactly 95%; report the
  conservative wider interval and the achieved coverage explicitly.
  If the HL CI collapses to a singleton (`L == U`), report the
  closed interval as such.
- **TOST for equivalence vs Condition A** at margin 0.20:
  *signed-rank TOST* (two one-sided Wilcoxon tests on shifted
  differences `d − 0.20` and `d + 0.20`):
  - `H0_lower:` median(d) ≤ −0.20, tested by `wilcoxon(d − (−0.20), alternative='greater', zero_method='pratt', method='exact')`.
  - `H0_upper:` median(d) ≥ +0.20, tested by `wilcoxon(d − (+0.20), alternative='less', zero_method='pratt', method='exact')`.
  - Both must reject at α=0.05 → conclude equivalence.
- **Multi-comparison correction:** all 9 K's × 3 binding contrasts
  (vs TM-Recompute-BM25, PageSummary-Quest-inspired,
  Refresh-K-budgeted) are confirmatory. Apply **family-wise Holm**
  across the 27 tests at α=0.05 (`scipy.stats.false_discovery_control`
  with `method='by'` is FDR; we want FWER, so use Holm via
  `statsmodels.stats.multitest.multipletests(..., method='holm')`).
  This *removes* the K=96 single-confirmatory framing — which was
  soft HARKing because the smoke informed the choice.

**Gate at the K-sweep level (not a single (task, K) configuration):**

- **Strong pass:**
  - At K=96 specifically, `Score(RepairKV) − Score(PageSummary-Quest-inspired) ≥ 0.10`,
    Holm-adjusted Wilcoxon p < 0.01, HL CI lower bound > 0.03.
  - Holm-adjusted significant Δ ≥ 0.10 against TM-Recompute-BM25 and
    Refresh-K-budgeted at K=96.
  - Burst-ablation gate: `Score(RepairKV-no-burst) ≥ Score(PageSummary-Quest-inspired) − 0.05` at K=96.
  - TOST signed-rank equivalence vs Condition A at margin 0.20: both
    one-sided tests reject.
  - At least 5 of 9 K's show Holm-adjusted significant Δ > 0 against
    PageSummary-Quest-inspired (frontier robustness).

- **Weak pass:**
  - At K=96, `Δ` against PageSummary-Quest-inspired ∈ [0.05, 0.10),
    Holm-adjusted Wilcoxon p < 0.05.
  - Burst-ablation gate not violated.
  - TOST equivalence margin loosened to 0.30 (>1-needle tolerance,
    explicit).

- **Fail:**
  - Any K=96 contrast Holm-adjusted p ≥ 0.05, or any Δ < 0.05
    against PageSummary-Quest-inspired, or burst-ablation gate
    violated, or TOST rejects equivalence at margin 0.30.

PageSummary-Quest-inspired is the **binding contrast** for the
abstract sentence; the other contrasts are reported but
PageSummary is what determines strong/weak/fail.

### T_repair stability rule (new in v5)

Step 1 smoke must report `σ(T_repair) / μ(T_repair)` per (task, K) configuration. If
σ/μ > 0.10 on the K=96 run, the budget multiplier bumps from 1.05
to 1.20 for all TM conditions. Pin the multiplier in writing before
headline runs start. We do not change the multiplier post-hoc.

### Per-K wall-clock budget guard (new post-Step-0d, from senior ML researcher's mid-execution review)

Before launching Step 3, the orchestrator MUST:

1. Read Step 1 smoke σ/μ T_repair.
2. Set the runner's TM budget multiplier (1.05 default; 1.20 if σ/μ > 0.10).
3. Estimate Step 3 wall-clock from smoke per-K timings × n=24/n=4 scaling.
4. Hard-abort if estimated total > 130 min (110 budget + 20% slack).
5. Log all four numbers to the run header before any K's run.

### Pre-commit to git before headline runs

Per the statistical reviewer: this v4 plan must be committed to git
**before** Step 3 starts. The paper will cite the commit hash. This
defuses HARKing on the pre-registered abstract sentences.

### Deliverables

- `phases/phase18_pre_submission/scripts/run_w1.py` — orchestrates
  all 6 conditions, reads back per-example `T_repair`, computes
  per-example budgets, runs each TM condition.
- New conditions wired into `phases/phase6_repair/src/runner.py`:
  `Refresh-K-budgeted`, `TM-Recompute-BM25`, `PageSummary-budgeted`.
- New helper module:
  `phases/phase18_pre_submission/src/page_summary.py`.
- Per-(task, K) CSVs:
  `phases/phase18_pre_submission/results/w1_<task>_<model>_K<K>.csv`.
- Audit JSON with per-example `T_repair` distribution, budget overage
  flags, and Wilcoxon/HL stats.

---

## W2 — Stage decomposition (post-bugfix)

### Bugs to fix first (pre-headline)

1. **`runtime_capacity.py:285-289` and `:402-406`** — split the
   `.to(device=target, dtype=torch.float32)` into BF16 H2D first
   (truly async on pinned), then on-device `.float()`.
2. **`host_pool_coverage`** — refuse to write a row with coverage
   < 1.0 unless `--allow-partial-coverage` is set. Set
   `source_pool_chunks ≥ ceil(candidate_tokens / chunk_tokens)` for
   the headline run.
3. **`warmup_trials` bumped to ≥3** so first-faulted-page costs are
   excluded.
4. **Pinning a 14 GB tensor with `torch.randn` will swap an 8-vCPU /
   64 GB host.** Pre-allocate empty pinned and fill in chunks; or
   pin per-chunk.

### Stages in v4

| Stage | What it measures |
|---|---|
| Append + Q2 projection (with active rows present) | Forward pass with Q2 tokens appended to the compressed cache; attention against active rows included. |
| Score scan (full) | End-to-end chunked `Q · K^T` over offloaded BF16 keys + score aggregation `mean_{l,h} max_t`. |
| Top-K select | Per-chunk argpartition + global merge. |
| Host→GPU promote | Pinned-DRAM → device copy. |
| Merge / re-layout | Inserting promoted rows into the active KV layout. |
| **Sequential total** | Sum of the above. |
| **Concurrent total v4 (priority order)** | (a) chunk-N+1 H2D ∥ chunk-N scan [highest leverage]; (b) Q2-proj ∥ promote; (c) score-aggregate ∥ next-chunk scan if cheap. |
| *Reference: prefill of evicted prefix, FA-2 / FA-3* | `t_prefill(n_evicted)` — recompute floor. |
| *Reference: prefill of full prefix, FA-2 / FA-3* | `t_prefill(B_base + n_evicted)` — full prefill. |

### Configurations

Offload sizes: 32K, 256K, 1M, 4M. K∈{96, 5000}. `query_len = 64`.
`chunk_tokens = 16384`, `trials = 80`. Pinned warmup over the
entire candidate pool (full pool, not a 1-chunk re-read).

FA-3: probe with `attn_implementation="flash_attention_3"` at smoke
time; if dispatch silently falls back to FA-2, log and report only
FA-2. Headline FA = FA-2 (broadly available); FA-3 in appendix.

### Sequential headline; explanation

- Sequential is what the prototype literally pays. Reproducible
  from this repo's code. A reviewer can re-run the script.
- Concurrent is what a deployed runtime would achieve. We never
  built the concurrent path, so claiming it as the headline is an
  extrapolation.
- Headline phrasing: "our prototype takes X ms (sequential); a
  deployment-realistic concurrent path achieves X' ms (Table 4)."

---

## W3 — Repo diagnostic (conditional)

Demoted to conditional per v3. The primary action is W4.3 wording
softening. TM-TextRetrieval-K BM25 condition runs only if Steps 1-7
finish on schedule.

---

## W4 — Paper edits (no GPU; **in green; awaiting per-paragraph approval**)

### Editing protocol (unchanged)

- All Phase 18 paper edits use `\textcolor{green!50!black}{...}`.
- Nothing committed to `paper/main.tex` until you approve each green
  block in this conversation.

### W4.1 — Novelty paragraph (revised v4)

Slot after Method "Overview." Includes ArkVale / InfiniGen / EM-LLM
prior art that v3 was missing, and explicitly distinguishes RepairKV
from per-step retrievers like Quest/ShadowKV/ParisKV. Explains the
chunk-position-budgeted Refresh-K choice for the careful reader.

> "Query-aware KV retrieval methods (Quest, FIER, ShadowKV, ParisKV)
> reselect rows during *active* attention from a preserved or
> low-rank cache. Page-eviction-and-recall systems (ArkVale,
> InfiniGen, EM-LLM) recall query-relevant evicted pages at decode
> time. Preservation-and-resume systems (InferCept, Continuum,
> CachedAttention) keep KV across pause boundaries so future turns
> do not re-prefill. RepairKV occupies a distinct slot: it applies
> query-aware scoring to the *post-compression, pre-resume* boundary
> in the cache lifecycle — once per pause, against an evicted store
> created by the compressor itself, with no persistent low-rank
> index. The matched-active-cache controls (`B_match`, Random-K,
> Oldest-K, StaleQ-K, WrongQ-K, Refresh-K-budgeted, PageSummary-
> budgeted) hold the resumed footprint constant so the comparison
> isolates this lifecycle operator from "keep more rows," "use a
> different scorer at active attention," or "use a per-step page
> recaller."

### W4.2 — Cost-accounting paragraph (unchanged from v3)

Discussion / Limitations:

- **Active-cache GPU rows:** matched (`B_base + K`).
- **Host-memory store size:** not matched; `|W_N|` reported.
- **Q2 projection compute:** in W2 stage table.
- **Scan + top-K + transfer compute:** in W2 stage table.
- **Cache merge / re-layout compute:** in W2 stage table.
- **Total wall-clock:** matched in W1 against TM-Recompute-BM25,
  PageSummary-budgeted, and Refresh-K-budgeted.
- **FLOPs:** not matched; reported separately as a derived row.

### W4.3 — Repo-diagnostic wording softening (unchanged from v3)

### W4.4 — Runtime paragraph (revised for ratio reframe)

> "We run a separate GPU runtime probe for the repair operation. The
> probe measures Q2 projection on the compressed cache, chunked scan,
> top-K selection, KV movement over pinned host-memory BF16 tensors,
> and the cache-merge step, plus FlashAttention-2 reference timings
> for full-prefix and evicted-prefix prefill (FA-3 in appendix). At
> K=5000 and 32K offloaded candidates, the full sequential p95
> RepairKV cost including Q2 projection and merge is about X ms
> (the prototype path); a deployment-realistic concurrent path
> achieves X' ms (Table 4); at 1M candidates, Y s sequential / Y' s
> concurrent. RepairKV's cost stays well below full-prefix recompute
> (V ms at 32K, FA-2) and is comparable to a budgeted Quest-style
> page-summary scan (PageSummary-budgeted in Table 5). The
> mechanistic difference is RepairKV's content-aware burst expansion
> at the lifecycle slot, not raw scan speed."

### W4.5 — Pre-registered abstract branches (revised for reframe)

- **Strong pass:** "RepairKV approaches `Q_2`-aware reselection
  quality without paying for a persistent low-rank index or a
  Q_2-time full reselection scan, and at much smaller wall-clock
  cost than full-prefix recompute. On Qwen2.5-7B at 32K context,
  MQ-NIAH-4Q at K=96, RepairKV scores 0.910 vs 0.245 matched, vs
  \[fill] for the strongest time-matched alternative
  (PageSummary-budgeted) and \[fill] for full-prefix recompute."
- **Weak pass:** "Across MQ-NIAH-4Q/6Q at 32K context, RepairKV
  shows a modest but consistent advantage over both wall-clock-
  matched recompute and budgeted Q_2-aware reselection at the same
  active-cache budget; at K=96 on 4Q, RepairKV scores 0.910 vs 0.245
  matched, vs \[fill] for the strongest time-matched alternative
  (PageSummary-budgeted)."
- **Fail:** "RepairKV is competitive with wall-clock-matched
  recompute and budgeted Q_2-aware reselection at small repair
  budgets and identifies the post-compression, pre-resume lifecycle
  slot as a previously unstudied position. Further work is needed to
  determine the conditions under which idle-window repair dominates
  Q_2-time alternatives."

The branch is chosen by the gate, not by preference. The plan is
git-committed before any headline run; the paper cites the commit.

---

## Sequencing on one tmux session

```text
Step 0a — runtime_capacity.py bug fixes (~30 min, no GPU)
  Fix BF16->FP32 dtype upcast (lines 285-289, 402-406).
  Add full-pool warmup assertion. Bump warmup_trials to 3.
  Pin host pool in chunks, not as a single torch.randn allocation.

Step 0b — chunk-position Refresh-K rewrite (~45 min dev + 10 min smoke)
  Rewrite score_evicted_positions to chunk the per-layer matmul
  by position blocks. Wall-clock check between chunks.
  Unit test: chunked == unchunked when no cap fires.

Step 0c — page-summary baseline dev (~45 min dev + 10 min smoke)
  Implement two-stage scorer: per-chunk max-key summaries computed
  at compression time + Q2-time cheap scan over summaries + full
  scoring of top-N chunks. Unit-test summary aggregation.

Step 1 — W1 timing-only smoke (~15 min)
  4Q n=4 K=64,96,128 with full instrumentation including
  idlekv_q2_proj_ms (newly surfaced).

Step 2 — W2 paper-quality, post-bugfix (~50 min)
  Stage decomposition with chunk-N+1 H2D ∥ chunk-N scan, FA-2 + FA-3
  references, full-pool warmup at 1M and 4M.

Step 3 — W1 4Q K-sweep paper-quality run (~110 min)
  n=24, 6 conditions, K∈{8,16,24,32,48,64,80,96,128}.

Step 4 — Audit gate (3-tier, statistical tests pre-registered above)

Step 5 — W1 6Q K-sweep (~75 min) [Strong-pass only]
  Same shape, B_base=18432, mq_niah_6q_clean_suite.

Step 5.5 — Burst-expansion ablation (~25 min) [Strong or Weak]
  RepairKV-no-burst at K=96 on 4Q, n=24. Required before paper edits
  even on weak pass — its result determines whether the abstract says
  "lifecycle slot" or "burst expansion at the lifecycle slot."

Step 6 — Llama appendix run (~45 min) [Strong or Weak]
  Llama-3.1-8B-Instruct + Mistral-7B-Instruct-v0.3, 4Q K=96 only,
  6 conditions, n=24. Sequential preferred; opportunistic parallel
  if Step 5 leaves the GPU under-utilized.

Step 7 — Recency-favorable partition (~30 min) [if easy]
  4Q 12->34 partition, K=96, n=24. Appendix only.

Step 8 — W4 paper edits (no GPU)
  Pull X/Y/Z/V from W2 CSV. Pick abstract branch from Step 4.
  Write green-marked edits in chat. Wait for per-paragraph approval.
  Recompile and rerun the Phase 17 rg validation checks.
```

We do not parallelize Step 3 with W2; the GPU is single-tenant and
quality-run timing is sensitive. Step 6 multi-model runs *may*
parallelize because the Llama and Mistral loads fit alongside Qwen
in 96 GB and the timing claims are model-internal (`T_repair(i, K)`
is measured per-model from that model's own RepairKV row).

---

## Risks and mitigations (v4)

1. **Chunked Refresh-K's cap fires mid-aggregation** (between
   layer-position chunks, the running `mean_{l,h} max_t` is
   incomplete) → snapshot the running aggregate at chunk
   boundaries; the cap fires *between* chunks; positions in
   un-scored chunks get `score = -inf` so the existing tiebreaker
   handles them deterministically. Unit-test that
   chunked-with-no-cap matches unchunked within fp16 tolerance.
2. **PageSummary cheap-stage's max-pool over keys is a known coarse
   filter**; if it filters out the right page, recall is bad. We
   pre-register that the page-summary chunk size is **128 tokens**
   (matching the recency-window default in the runner) so the
   summaries are not so coarse that they trivially miss; sensitivity
   to chunk size is an appendix table.
3. **TM-Recompute-BM25 RoPE position correctness** → 5-example
   bit-exact unit test (existing risk; mitigation unchanged).
4. **Per-K warmup penalty** → randomize K order per example or
   discard the first K's first 2 examples as warmup (runner change).
5. **Multi-model runs fail on Llama or Mistral** → if one model
   shows a substantially weaker effect, we honest-report that
   variance in §W4 rather than relegating it; the abstract sentence
   already permits a "consistent across models tested" qualifier.
6. **Recency-favorable partition flips the result** → if RepairKV
   underperforms B_match on 12→34, it is appendix-only honest
   reporting, and the §W4.3 wording must say so explicitly.
7. **HARKing accusation** → git pre-commit + commit-hash citation in
   the paper.
8. **PageSummary-budgeted matches RepairKV** → if it does, the
   abstract sentence's `Δ ≥ 0.10` against PageSummary-budgeted
   triggers Weak or Fail; that is the correct, falsifiable
   outcome of the F-reframe and we accept it.

---

## Two open questions the verification panel raised — your call

These are not blockers; v5 is internally consistent and survives
all three verification critiques on the implementation and
statistical fronts. The two open framing/scope questions:

1. **Add a non-NIAH paper-quality run?** Devil's advocate v2's
   strongest remaining attack: every paper-quality experiment in v5
   is on an MQ-NIAH variant. The real-repository diagnostic (paper §
   Real-repository) is descriptive at K=192 and is now further
   softened in W4.3, but it is not a *confirmatory* non-NIAH run.
   Adding one (e.g., SCBench multi-turn QA, RULER variable-tracking,
   LongBench HotpotQA) at K=96 n=12-24 would convert "we tested only
   on needle tasks" into "we tested on a non-needle task with
   consistent direction." **Cost: 45-60 min GPU + several hours of
   task-harness dev** (the runner currently supports MQ-NIAH variants
   and the SWE-bench manifest path; no SCBench/HotpotQA loader
   exists). **My read:** out of Phase 18 scope. Defer to Phase 19,
   acknowledge in W4.3 that this is the most important next step,
   and lean on the multi-model runs (Step 6) as the cross-cut
   robustness for v5. Confirm or push back.

2. **The "much smaller wall-clock cost than full-prefix recompute"
   wording.** The smoke gives 18 ms RepairKV. A 7B model at 32K with
   FA-2 prefills in roughly 700–1500 ms — so the ratio is ~50–80×,
   which justifies "much smaller." But if W2's measured V (full-
   prefix prefill at 32K, FA-2) lands below 360 ms (i.e., V/X < 20),
   we should soften "much smaller" to "an order of magnitude smaller"
   in W4.5. **My read:** auto-pin: write the abstract sentence with
   "much smaller" as the strong-pass default; if V/X < 20 in the W2
   numbers, we drop to "an order of magnitude smaller" without
   re-asking. Confirm or push back.

## Other rows I need from you

3. **Confirm v5 plan as the final plan.** Once you say go, I will:
   - Git-commit this file (no other changes) so the timestamp is on
     record before Step 0a.
   - Start Step 0a immediately.
4. **Per-paragraph approval on the W4.1 / W4.4 / W4.5 green-marked
   edits when their numbers are ready** — but not before.
5. **Step 6 is sequential.** Decided: never parallel — co-loading
   two models on one GPU contaminates `T_repair` and invalidates the
   W1 fairness contract. ~150 min sequential is fine.
