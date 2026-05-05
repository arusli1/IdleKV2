# Phase 17: Rerun headline experiments with proxy scorer

## The issue

The paper's main-text quality results (Figure 2 frontier, Table 1
specificity, Figure 3 multiturn, Figure 4 compression-policy sensitivity,
Figure 5 mechanism diagnostic, Appendix Table A1 Scissorhands stress) all
use the **exact $Q_2$ attention-query scorer**. The exact scorer extracts
$Q_2$'s post-RoPE query tensors from a forward pass that sees the full
uncompressed cache. This is **not deployable** — keeping the full
uncompressed cache around for the scoring step defeats the purpose of
compression.

The deployable alternative is the **$Q_2$-appended-state proxy scorer**:
it appends $Q_2$ to the *compressed* cache, runs a short forward pass on
those $Q_2$ tokens, and reuses the resulting cache rows as the query
tensors for scoring. Cost is bounded by $|Q_2|$ tokens of prefill plus
$O(|Q_2| \cdot N)$ attention-score work, with no need to materialize the
full uncompressed cache.

### What the existing proxy data tells us

The Latency accounting paragraph (currently line ~713 of `paper/main.tex`)
reports proxy results on paired 4Q/6Q runs at $K=96$:

- Proxy on 4Q: **0.970** (vs. exact 0.910, matched no-repair 0.245)
- Proxy on 6Q: **0.894** (vs. exact 0.989, matched no-repair 0.422)

Two facts pop out:

1. **Proxy beats exact on 4Q at $K=96$** (0.970 > 0.910). Proxy is not a
   degraded approximation of exact — it is a peer scorer whose
   compressed-cache forward pass produces $Q_2$ queries that happen to
   discriminate evicted positions at least as well as exact's
   uncompressed-cache queries.
2. **Proxy is the deployable scorer.** A production system would run
   proxy, not exact.

### Why this matters for the paper

The current framing in the paper positions exact as the headline scorer
(main curves) and proxy as a "cheaper alternative" buried in Latency
accounting and Appendix Figure A4. A reviewer who reads the abstract or
Results section reasonably assumes the headline 91.0% / 24.5% numbers are
what a real system would deliver. But those are exact-scorer numbers;
the deployable proxy gets 97.0% on the same 4Q setting. The thesis is
honest only if the reader notices the buried framing.

A clean paper would report **proxy in the headline**, with exact as a
methodological cross-check. The Discussion limitation "exact scorer is a
diagnostic rather than a production selector" would no longer be a
hedge; it would be redundant because proxy *is* the production selector
and proxy *is* what the headline measures.

## Proposed rerun plan

Switch the headline to proxy. Two scopes:

### Scope A (minimum viable rerun) --- regenerate the headline visuals

Sufficient to re-anchor the paper's main claims around the deployable
scorer.

| Float | Cells |
|---|---|
| Figure 2 (matched-budget frontier) | 4 task variants (2Q/4Q/6Q/8Q) $\times$ 9 $K$ values (8, 16, 24, 32, 48, 64, 80, 96, 128) $\times$ 4 conditions ($\bmatch$, Random-$K$, Oldest-$K$, RepairKV) $\approx$ 144 condition cells $\times$ 100 examples on 2Q/4Q/6Q + 24 examples per partition on 8Q |
| Table 1 (specificity, $K=48$ on 4Q) | 5 conditions ($\bmatch$, StaleQ-$K$, WrongQ-$K$, RepairKV, Refresh-buffered) $\times$ 72 paired example-partition cases |

Compute estimate: 4Q/6Q dominate; 4Q at 32K context with Qwen2.5-7B-Instruct
on RTX PRO 6000 Blackwell is on the order of seconds per example for the
proxy scorer (one short prefill + a chunked GPU score over pinned
host-memory keys). 144 cells $\times$ ~100 examples is order of a few hours
on one GPU.

### Scope B (full rerun) --- regenerate every exact-scorer result

Adds the secondary claims (multiturn, compression-policy sensitivity,
mechanism, Scissorhands stress).

| Float | Cells |
|---|---|
| Figure 3 (5-turn multiturn on 8Q at $K=80$, $n=24$) | 5 turns $\times$ 24 examples $\times$ ~4 conditions |
| Figure 4 (compression-policy sensitivity on 4Q, $n=24$) | sink-plus-recent + SnapKV $\times$ ~4 conditions $\times$ full $K$ sweep |
| Figure 5 (selection diagnostic) | derived from Figure 2 cells |
| Appendix Table A1 (Scissorhands persistence stress on 6Q, $n=24$) | ~5 $K$ values $\times$ ~3 conditions |
| Appendix Figure A2 (breadth/operating regime) | additional task variants and $K$ values |
| Appendix Figure A3 (partition endpoints) | derived from Figure 2 |

Compute estimate: roughly 50% more on top of Scope A.

### Scope C (out of scope here) --- real-repository diagnostic

Table 2 / Figure A1 use a separate "event-only" scorer over redacted
callsite cues, not the exact-$Q_2$ scorer. Whether to switch real-repo
to a proxy-style scorer is a separate question; not bundled into this
phase.

## Workflow

1. Confirm proxy scorer implementation is feature-complete and
   deterministic enough to regenerate the existing example sets without
   drift.
2. Run Scope A first. Compare proxy curves to existing exact curves at
   the same $K$. Expected: comparable or slightly different numbers, no
   qualitative change in the matched-budget claim.
3. If Scope A confirms thesis, decide whether Scope B is worth the
   additional compute. The mechanism diagnostic (Figure 5) and the
   Scissorhands stress test are the most reviewer-visible secondary
   experiments; the others can stay on exact with appropriate framing.
4. Update `paper/main.tex` to switch headline numbers, regenerate
   figures, and reframe the proxy-vs-exact relationship in Setup
   (Scorers paragraph) and Discussion (limitations).

## Acceptance criteria

- All Scope A figures and tables regenerated with proxy scorer.
- Headline numbers in abstract, intro, and Results section reference
  proxy (deployable) figures, not exact.
- Setup paragraph clarifies proxy is the canonical deployable scorer
  and exact is reported only as a methodological cross-check.
- Discussion no longer hedges "exact is a diagnostic" since exact is no
  longer the headline measurement.

## Open questions

- Does the proxy beat exact consistently across the 6Q and 8Q sweeps,
  or is the K=96 4Q result an anomaly? Scope A would answer this.
- Does the multiturn diagnostic survive the switch to proxy? Compressed
  cache after multiple turns may yield different $Q_2$ queries than the
  single-turn case.
- For real-repo, does the event-only scorer have an analogous "exact
  vs. proxy" structure? Worth a quick audit.
