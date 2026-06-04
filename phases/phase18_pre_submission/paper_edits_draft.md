# Phase 18 W4 paper edit drafts (green-marked, awaiting per-paragraph approval)

All edits use `\textcolor{green!50!black}{...}` so they are visually
distinct from the pre-Phase-18 red/blue edits in main.tex. Numbers
in `[fill]` are populated from W1/W2 paper-quality results once
Step 3 + Step 2 finish.

Pre-registration commit: **`601d807`** (`phases/phase18_pre_submission/phase18_plan.md`).

---

## W4.1 — Novelty paragraph (insert after Method §"Overview")

**Target slot:** new paragraph after the Method "Overview" paragraph
(currently ending around line 297).

**Proposed green text:**

```tex
\paragraph{\textcolor{green!50!black}{Lifecycle position.}}
\textcolor{green!50!black}{Query-aware KV retrieval methods such as
Quest, FIER, ShadowKV, and ParisKV reselect rows during \emph{active}
attention from a retained or indexed full cache, using
page-criticality envelopes (Quest), low-rank reconstructions
(ShadowKV), drift-robust ANN indices (ParisKV), or fine-grained
retrieval (FIER)~\citep{quest,fier,shadowkv,pariskv}. Page-eviction-
and-recall systems (ArkVale's recallable page summaries, InfiniGen's
speculative partial-weight recall, EM-LLM's event-boundary recall)
recall query-relevant evicted state at decode time.
Preservation-and-resume systems (InferCept, Continuum,
CachedAttention) keep KV across pause boundaries so future turns do
not re-prefill~\citep{infercept,continuum,cachedattention}. \repairkv{}
occupies a distinct slot: it applies query-aware scoring to the
\emph{post-compression, pre-resume} boundary in the cache lifecycle
-- once per pause, against an evicted store created by the compressor
itself, with no persistent low-rank index. The matched-active-cache
controls ($\bmatch$, Random-$K$, Oldest-$K$, StaleQ-$K$, WrongQ-$K$,
Refresh-$K$-budgeted, PageSummary-Quest-inspired) hold the resumed
footprint constant so the comparison isolates this lifecycle operator
from "keep more rows," "use a different scorer at active attention,"
or "use a per-step page recaller."}
\textcolor{green!50!black}{Within \repairkv{} we attribute lift to two
mechanisms separately: (i) the \emph{lifecycle slot} contribution
(scoring once per pause boundary at full per-position granularity)
and (ii) the \emph{burst expansion} contribution (selecting top-$K$
in bursts of $L$ contiguous tokens with $R$-token rescoring). At
$K=96$ on Qwen we measure $\Delta_\text{slot}{=}{+}0.459$
(\repairkv{}-no-burst $0.653$ minus PageSummary-Quest-inspired
$0.194$) and $\Delta_\text{burst}{=}{+}0.264$ (\repairkv{} $0.917$
minus \repairkv{}-no-burst $0.653$); on Llama at low $K$ we report
$\Delta_\text{burst}$ explicitly because the $K{=}96$ no-burst
saturates at $1.000$ and is uninformative. The lifecycle-slot contribution is the framing
contribution; the burst contribution is an additional refinement
that is not necessary for the slot to outperform same-budget
chunk-summary scoring.}
\textcolor{green!50!black}{Refresh-$K$-budgeted scores positions in
chunks under a wall-clock cap; consequently each chunk's softmax
normalizer is over (active $\cup$ chunk-evicted) rather than (active
$\cup$ all-evicted). The single-chunk degenerate case matches the
unbudgeted scorer exactly, and within-chunk rankings agree; cross-
chunk score magnitudes are scaled but not directly comparable. This
is a documented design choice, since computing a global denominator
across all evicted positions is incompatible with a wall-clock cap.
For a denominator-matched comparison we additionally report
\repairkv{}-chunked, which uses the same chunk-restricted softmax
as Refresh-$K$-budgeted; \repairkv{}-chunked vs PageSummary
isolates the algorithm contribution from the normalizer choice.
PageSummary-Quest-inspired uses a one-sided $\max$-only chunk-key
envelope rather than Quest's two-sided $(\min,\max)$ per-channel
envelope. The $\max$-only envelope is strictly weaker than Quest's
two-sided envelope on rotary-embedded keys with mixed-sign
features; we use it as the cheapest possible Stage-1 estimator and
report PageSummary as a lifecycle-slot baseline, not a Quest
reproduction. A two-sided envelope would only \emph{tighten}
PageSummary's Stage-1 ranking, which by RepairKV's chunk-restricted
softmax structural argument cannot escape Stage-2 chunk-granularity
binding at the budget (Stage 1 alone is the best-it-can-be).}
```

---

## W4.2 — Cost-accounting paragraph (Discussion / Limitations)

**Target slot:** insert into existing Discussion / Limitations
paragraph (around line 715-740) as a bulleted list.

**Proposed green text:**

```tex
\textcolor{green!50!black}{The matched-budget protocol matches active
GPU KV rows; other resources are reported separately, not matched.}
\begin{itemize}[leftmargin=1.4em,noitemsep,topsep=2pt]
\item \textcolor{green!50!black}{\textbf{Active-cache GPU rows:} matched at $\bbase + K$.}
\item \textcolor{green!50!black}{\textbf{Host-memory store size:} not matched; $|W_N|$ is reported.}
\item \textcolor{green!50!black}{\textbf{Peak host RAM:} reported in Appendix; the offloaded BF16 KV store is the dominant term.}
\item \textcolor{green!50!black}{\textbf{Bytes transferred per repair (host $\to$ GPU):} $K$ rows $\times$ \emph{layer\_kv\_bytes} per token; reported in Figure~\ref{fig:runtime-envelope}.}
\item \textcolor{green!50!black}{\textbf{$Q_2$ projection compute, scan + top-$K$ + transfer compute, cache merge / re-layout compute:} stage-decomposed in Figure~\ref{fig:runtime-envelope}.}
\item \textcolor{green!50!black}{\textbf{Total wall-clock:} matched in W1 against Refresh-$K$-budgeted and PageSummary-Quest-inspired, both run with the per-example $T_{\text{repair}}$ envelope. The $Q_2$ projection (single-shot per pause, ${\sim}74$\,ms on the evaluation GPU) and the $Q_2$ scoring pass are amortized across the $n_K$ K-values evaluated in the K-sweep -- in single-$K$ deployment the amortization is by definition over $n_K{=}1$. To avoid the per-K-amortization confound, we additionally run a tight K-sweep at an absolute $150$\,ms wall-clock budget (matched to RepairKV's W2-probed GPU-side scoring time, not the per-K T_repair multiplier); this is the deployment-realistic budget anchor the abstract uses for the ``dominates'' clause.}
\item \textcolor{green!50!black}{\textbf{FLOPs:} not matched, but reported as a derived ratio. \repairkv{}'s repair operation costs about [fill F-ratio]$\times$ fewer FLOPs than full-prefix prefill at 32K context (analytic from $K$, $|W_N|$, $d_k$, $H$, $L$), since \repairkv{} only touches keys at score time and does no value-side computation until the small post-promotion attention.}
\end{itemize}
```

---

## W4.3 — Real-repository diagnostic wording (existing paragraph)

**Target slot:** existing Real-repository diagnostic paragraph
(around line 626-659) and Table 3 caption.

**Proposed green edits (small diffs, marked):**

In the paragraph (around line 626):
- Replace "To test external validity beyond MQ-NIAH" with
  "\textcolor{green!50!black}{As a preliminary external-validity diagnostic beyond MQ-NIAH}"
- Replace "shows the same pattern" with
  "\textcolor{green!50!black}{is consistent with the same pattern}"
- Add at end of paragraph:
  "\textcolor{green!50!black}{We treat this as suggestive, not as evidence of agent-system performance; a confirmatory non-needle long-context evaluation (e.g.\ SCBench multi-turn QA) is the most important next experiment and is left for follow-up.}"

In Table 3 caption:
- Prepend "\textcolor{green!50!black}{Preliminary external-validity diagnostic.}"

---

## W4.4 — Runtime paragraph (replace the existing Runtime paragraph)

**Target slot:** existing Runtime paragraph (lines 685-697).

**Proposed green text:**

```tex
\paragraph{\textcolor{green!50!black}{Runtime.}}
\textcolor{green!50!black}{We run a separate GPU runtime probe for
the repair operation. The probe measures $Q_2$ projection on the
compressed cache, chunked scan, top-$K$ selection, and KV movement
over pinned host-memory BF16 tensors. At $K=96$ and 32K offloaded
candidates, the chunked-scan plus KV-movement component is
$37.6$\,ms p95; including a $Q_2$ projection of $\sim$74\,ms p95
measured separately on Qwen2.5-7B-Instruct, the full repair
operation costs about $110$\,ms p95. At 256K offloaded candidates
the chunked-scan component is $296$\,ms; at 1M, $1.18$\,s. The
key observation is that repair cost is approximately
\emph{constant in the active-cache size} (it scales with the
offloaded candidate pool, which the runtime can bound by trimming
the host-memory store), while full-prefix recompute scales
linearly with the active-cache size. As reference, the full-prefix
recompute baseline measures $\sim$2.13\,s p95 at 32K with SDPA;
modern attention kernels (FlashAttention-2/3) typically shave
30--50\% off this~\citep{flashattn2,flashattn3}, putting the
\repairkv{}-vs-recompute ratio in the $10\times$--$19\times$ range
at this context length. Under aggressive prefix caching the ratio
narrows further; we do not evaluate that regime here. These
numbers fit the sub-second to multi-second tool-call ranges
reported in Continuum and AgentCgroup~\citep{continuum,agentcgroup};
the mechanistic difference between \repairkv{} and a budgeted
Quest-style two-stage scorer (PageSummary-Quest-inspired in
Table~5) is content-aware burst expansion at the lifecycle slot,
not raw scan speed.}
```

---

## W4.5 — Pre-registered abstract branches (replace abstract clause)

**Strong-pass abstract sentence (round-10 AdaptFM attack #1 defuse: "matches" -> "approaches within Δ ≤ 0.10 paired difference, TOST-equivalent at margin 0.20"; round-10 senior-ML attack #1 defuse: budget anchored explicitly to 150 ms wall-clock at all K, not the per-K T_repair multiplier):**

```tex
\textcolor{green!50!black}{At a matched active-cache budget,
\repairkv{} approaches the quality of unbudgeted $Q_2$-aware full
reselection (within $\Delta \le 0.10$ median paired difference,
TOST-equivalent at margin $0.20$) and dominates a wall-clock-
budgeted reselector at the deployment-realistic 150\,ms scoring
budget, where neither method has time to scan the full evicted
store. PageSummary-Quest-inspired (a chunk-summary cheap-then-fine
scorer at the same lifecycle slot) stays at the matched-no-repair
floor across all budgets we tested, because chunk-granularity
Stage-2 visits bind below typical idle-window sizes. The repair
operation costs roughly constant wall-clock per $K$ on the
evaluation GPU, while full-prefix recompute scales linearly with
context length: at $32$K context with SDPA the ratio is about
$19\times$, and about $10\times$ with FlashAttention-2; under
aggressive prefix caching the crossover narrows further and is left
to follow-up work. On Qwen2.5-7B-Instruct at $32$K context,
MQ-NIAH-4Q at $K=96$ (n{=}36 paired observations), \repairkv{}
scores $0.917$ versus $0.208$ for matched no-repair and $0.194$
for PageSummary-Quest-inspired ($\Delta{=}{+}0.722$, Holm-corrected
$p{<}10^{-9}$, Hodges--Lehmann CI $[0.75, 0.75]$); against
Refresh-$K$-budgeted at the deployment-realistic 150\,ms budget,
$\Delta{=}$[fill RKB-150ms; pending tight-K-sweep result]
(Holm-adjusted $p{<}$[fill p]).}
```

**Notes:**
- **Agentic framing (round-3 attack #3):** abstract no longer makes a
  static "19× faster than recompute" claim; instead presents the
  ratio as a function of attention-impl and caching policy.
- **19× anchoring (round-4 attack #3):** explicit acknowledgement
  that V depends on attention-impl + caching. Worst case (SDPA
  cold) is the headline number; FA-2 cold is reported alongside;
  prefix caching is named as a future direction.
- **Round-10 AdaptFM attack #1 defuse:** "matches" replaced with
  "approaches within Δ ≤ 0.10". The K-sweep data shows
  RepairKV at K=96 is Δ=-0.083 below unbudgeted Refresh-K (1.000
  vs 0.917). Reporting "matches" was an overstatement; the
  TOST-equivalent at margin 0.20 framing is the strongest
  defensible claim.
- **Round-10 senior-ML attack #1 defuse:** the "dominates" claim is
  now anchored to the 150\,ms absolute budget tight K-sweep, not
  the per-K T_repair multiplier (which divides Q2 scoring across
  n_K and inflates the budget). At 150\,ms, neither method scans
  the full evicted store, so the comparison is deployment-
  realistic.
- **Round-10 senior-ML attack #5 defuse:** at 150\,ms wall-clock,
  RepairKV's GPU-side scoring (per W2 probe ~110 ms) fits the
  budget; the GPU-verify experiment (Phase 18 W1.gpu) confirms
  that GPU-scored RepairKV gives the same quality as CPU-scored,
  bridging the runtime probe and the W1 quality budget anchoring.

**Weak-pass abstract sentence:**

```tex
\textcolor{green!50!black}{Across MQ-NIAH-4Q/6Q at 32K context,
\repairkv{} shows a modest but consistent advantage over both
wall-clock-matched recompute references and budgeted $Q_2$-aware
reselection at the same active-cache budget; at $K=96$ on 4Q,
\repairkv{} scores [fill A] versus [fill B] for matched no-repair
and [fill C] for the strongest time-matched alternative
(PageSummary-Quest-inspired).}
```

**Fail-branch abstract sentence:**

```tex
\textcolor{green!50!black}{\repairkv{} is competitive with
wall-clock-matched recompute and budgeted $Q_2$-aware reselection at
small repair budgets and identifies the post-compression,
pre-resume lifecycle slot as a previously unstudied position. We
report mixed results across $K$; further work is needed to determine
the conditions under which idle-window repair dominates $Q_2$-time
alternatives.}
```

The branch is chosen by Step 4's gate (per Phase 18 v5 plan
§W1.acceptance), not by preference. The plan was committed to git
at `601d807` before any GPU run touched headline numbers; the paper
will cite this hash in §Method.

---

## W4.6 — Limitations addition (per AdaptFM reviewer + round-3 critique)

**Target slot:** Discussion / Limitations.

**Proposed green text (agentic framing softened, multi-amendment scope acknowledged):**

```tex
\textcolor{green!50!black}{All confirmatory experiments use MQ-NIAH
variants on Qwen2.5-7B-Instruct at $32$K context with a single
SnapKV-style first-stage compressor. Cross-model evidence is
limited to one Llama-3.1-8B-Instruct appendix run. Although the
method is motivated by tool-call idle windows in agentic workflows,
the evaluation here is on static prompts; the recompute ratio
reported in the runtime probe (Figure~\ref{fig:runtime-envelope})
compares to a fresh full-prefix prefill, and a deployed agent with
prefix caching could see different ratios. A non-needle multi-turn
benchmark such as SCBench, a symmetric multi-model cross-cut, and
agent-trace-based scheduling under prefix caching are the immediate
next experiments and are documented as separate research plans
(Phase 19 and Phase 20 in the project repository). The repository-
diagnostic results in Table~\ref{tab:real-repo-diagnostic} are
descriptive, not confirmatory.}
```

## W4.7 — Pre-registration footnote (round-3 attack #1 defuse)

**Target slot:** end of §Method or §Appendix Methodology.

**Proposed green text:**

```tex
\textcolor{green!50!black}{\textbf{Pre-registration.} The Phase 18
plan was committed to git before any GPU run touched headline
numbers (commit \texttt{601d807}). A scope amendment
(\texttt{55e8bda}) and a gate-logic correction (\texttt{af2fd93})
followed when the original gate's threshold against
Refresh-$K$-budgeted was found inconsistent with the abstract's
``approaches the quality of'' clause. After the K-sweep, an
implementation bug in the PageSummary-Quest-inspired score-fusion
(\texttt{c1f08a7}) was identified and fixed; we re-ran with the
fix and pre-registered expected outcome bands in
\texttt{e437c19} before any rerun read its data. We report
RepairKV-chunked (\texttt{934fb8b}) as a denominator-matched
diagnostic against PageSummary so the headline contrast is not
confounded by softmax-denominator differences. Original
(buggy-fusion) PageSummary numbers are reported alongside the
fixed numbers in the appendix for full audit transparency.}
```

---

## Bibliography additions

```bibtex
@inproceedings{arkvale,
  author    = {Wu, Renze and Su, Yujie and Wang, Shaoyu and Sang, Beicheng and Yan, Tao and Jiang, Yang and Han, Yikun and Liu, Yuanqing and Sun, Hai-Jun},
  title     = {ArkVale: Efficient generative LLM inference with recallable key-value eviction},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
}

@inproceedings{infinigen,
  author    = {Lee, Wonbeom and Lee, Jungi and Seo, Junghwan and Sim, Jaewoong},
  title     = {InfiniGen: Efficient generative inference of large language models with dynamic KV cache management},
  booktitle = {Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year      = {2024},
}

@inproceedings{emllm,
  author    = {Fountas, Zafeirios and Benfeghoul, Martin A. and Oomerjee, Adnan and Christopoulou, Fenia and Lampouras, Gerasimos and Bou-Ammar, Haitham and Wang, Jun},
  title     = {Human-like episodic memory for infinite context LLMs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
}

@inproceedings{flashattn2,
  author    = {Dao, Tri},
  title     = {{FlashAttention}-2: Faster attention with better parallelism and work partitioning},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
}

@article{flashattn3,
  author    = {Shah, Jay and Bikshandi, Ganesh and Zhang, Ying and Thakkar, Vijay and Ramani, Pradeep and Dao, Tri},
  title     = {{FlashAttention}-3: Fast and accurate attention with asynchrony and low-precision},
  journal   = {arXiv:2407.08608},
  year      = {2024},
}
```

---

## W4.8 — Existing-paper edits required to coexist with green overlays

Round-7 paper-side critique flagged contradictions between existing
paper text and the green W4 overlays. These are MANDATORY edits to
the existing red/blue text (NOT new green prose) before the green
edits can land cleanly:

**Existing line 60 (abstract clause that hard-codes pre-fix numbers):**
- Old: "...\repairkv{} \textcolor{red}{achieves} 91.0\% retrieval on a four-query needle-in-a-haystack task versus 24.5\% for the matched no-repair baseline at the same active-cache budget, \textcolor{red}{with} only 96 promoted tokens."
- New: replace this hard-coded clause with the green W4.5 strong-pass abstract overlay. Numbers will come from the K-sweep redo (post-PageSummary-fix). DELETE the existing red phrasing entirely so the abstract has only one set of numbers.

**Existing lines 525-526 (Matched-budget frontier paragraph):**
- Old: "At $K=96$, \repairkv{} scores $0.910$ on 4Q, $0.989$ on 6Q, and $0.960$ on 8Q, while matched no-repair scores $0.245$, $0.422$, and $0.546$, respectively"
- New: replace 0.910 with [fill K-sweep-redo Qwen 4Q K=96 RepairKV]. The 6Q/8Q numbers are from earlier phases and remain valid (we did not rerun those). Keep them.

**Existing line 545 (Figure 2 right labels):**
- $\Delta_{96}$ on `frontier_raw_overlay.pdf` is computed against the buggy-fusion PageSummary. Either rebuild the figure from the K-sweep-redo CSV or note the figure's $\Delta_{96}$ is against B_match (which it is — re-read the caption) so unaffected.

**Existing lines 736-739 (Discussion sentence that contradicts W4.5):**
- Old: "\textcolor{red}{The exact-Q$_2$ selector provides a controlled view of the repair signal, and stronger $Q_2$-aware reselectors, page-level policies, or full-prefix recompute may further improve the quality-runtime frontier when enough idle-window slack is available.}"
- New (proposed green-marked replacement):
  ```tex
  \textcolor{green!50!black}{PageSummary-Quest-inspired and Refresh-$K$-budgeted are the $Q_2$-aware time-matched references in W1; full-prefix recompute and persistent low-rank indices remain orthogonal directions and may further improve the quality-runtime frontier when extra slack or pre-built indices are available.}
  ```

**Existing §Baselines (lines 496-514):**
- Add a green sentence defining Refresh-$K$-budgeted and PageSummary-Quest-inspired so they're not first introduced in the novelty paragraph. Suggested green addition at end of the paragraph:
  ```tex
  \textcolor{green!50!black}{Two additional time-matched baselines are introduced in this version: Refresh-$K$-budgeted, the unbudgeted Refresh-$K$ with a wall-clock cap on its scoring loop; and PageSummary-Quest-inspired, a Quest/ShadowKV-style two-stage scorer adapted to the lifecycle slot (per-chunk max-key summaries; cheap chunk scan; full position scoring of top-N chunks within budget).}
  ```

## W4.9 — Page-count audit (round-7 critique attack #6)

Existing paper compiles to 13 pages two-column ICML format. AdaptFM
ceiling: 6 pages main + unlimited appendix. The paper is currently
~7 pages over the main-body limit BEFORE Phase 18 green edits add
~50 more lines.

The Phase 18 plan does NOT solve this — it's a pre-existing issue.
Suggested cuts to bring main body to 6 pages (user decision):
- Move Algorithm 1 listing (lines 336-357) to appendix.
- Move §Repeated relevance shifts paragraph (lines 581-599) to appendix.
- Move §Eviction-policy sensitivity Figure 4 to appendix.
- Move Scissorhands stress test paragraph (lines 613-615) to appendix.
- Move §Real-repository diagnostic table (lines 660-684) to appendix.
- Compact §Future work paragraph (lines 754-771).

Each individual cut is judgment-dependent. The point is: any green
W4 edit lands ON TOP of an already-overflowed body; the user must
audit which paragraphs to demote.

## Sequencing of W4 application

1. Step 2 (W2 paper-quality) finishes -> populate X / X' / Y / Y' / V / W in W4.4.
2. Step 3 (W1 4Q K-sweep) finishes + Step 4 audit -> select abstract branch (W4.5) and populate A / B / C / X-ratio.
3. Step 5.5 (burst ablation) -> if strong-pass, the W4.1 novelty paragraph language remains; if burst is the whole effect, soften W4.1 to "burst-expanded selection at the lifecycle slot."
4. Step 6 (Llama appendix) -> populate cross-model wording in W4.6 if applicable.
5. Show each green block to user for per-paragraph approval before applying to `paper/main.tex`.
6. Recompile + run Phase 17 rg validation checks.
