# Phase 19: Non-NIAH Cross-Cut

## Goal

Add a paper-quality (n=24) non-NIAH confirmatory result so the
abstract sentence is supported on at least one task family that is
*not* a needle-in-a-haystack variant. Defuses Phase 18's strongest
remaining hostile-reviewer attack (devil's advocate v2, attack 2):
"You've shown RepairKV repairs needle retrieval, not attention."

## Why this is its own phase

Phase 18's headline runs (4Q / 6Q K-sweeps, multi-model Llama +
Mistral, burst ablation, recency partition) are all MQ-NIAH variants
because the runner currently supports MQ-NIAH and the SWE-bench
manifest path. The repo diagnostic at K=192 is the only non-NIAH
evidence in the paper and it is descriptive (n=48), not paper-grade
confirmatory. Adding a non-NIAH paper-quality run requires building
a new task harness — too much code to hold up Phase 18's submission
prep, but too important to leave un-done before submission.

## Candidate non-NIAH benchmarks (ranked)

### 1. SCBench multi-turn QA *(primary candidate)*

`SCBench` (Li et al., ICLR 2025; cited in our Related Work) is
explicitly KV-cache-centric and includes a *multi-turn* QA setting
where multiple questions share the same long document. That maps
naturally onto our two-turn protocol:

- Q1 = first turn question (compresses cache toward Q1 evidence).
- Q2 = second turn question (target — RepairKV needs to recover Q2
  evidence the Q1-conditioned compressor evicted).
- Shared 32K-context document = the offloaded substrate.

Why it is the best primary fit:
- Multi-turn structure is built in; no synthetic Q1/Q2 split needed.
- Already cited in our Related Work, so the cross-cut is "we tested
  on the benchmark we cite as motivation."
- Multiple subtasks (QA, summarization, in-context learning) — gives
  us choice of which is most diagnostic.

### 2. RULER variable-tracking (VT) *(secondary, cheap to build)*

VT is a chain-tracking task: `A=1, B=A, C=B, ..., what is C?` The
answer requires *composition*, not retrieval. RULER is already in
the repo's `ruler/` directory, so the loader path is short.

Pros: smallest dev cost (existing RULER infra).
Cons: still synthetic; reviewers may say "still RULER, still not
real." Useful as a *second* non-NIAH point alongside SCBench, not as
the primary.

### 3. LongBench 2WikiMultihopQA *(tertiary)*

Multi-hop QA over Wikipedia. Real, multi-hop, but single-turn —
Q1/Q2 split would have to be synthesized.

### 4. ∞Bench long-form retrieval / summarization *(tertiary)*

Strong long-context evaluation but the task formats vary; mapping
onto two-turn requires per-task adapter logic.

**Recommendation.** SCBench multi-turn QA as primary; RULER VT as
secondary if SCBench primary lands cleanly.

## Workstream

### W1 — Loader + two-turn adapter

- SCBench dataset download + tokenizer formatting.
- Two-turn adapter that picks a question from the multi-turn series
  as Q2, with prior turn(s) folded into the Q1-conditioned cache.
- Match Phase 6's existing `clean_suite` artifact schema so the
  audit / summary scripts work without changes.

Estimated cost: half-day.

### W2 — Evaluator + audit hooks

SCBench QA eval is exact-match for short answers, F1 for longer
ones. Implement both, default exact-match for the headline number,
F1 in appendix.

Estimated cost: ~2 hr.

### W3 — Smoke at n=4

Same shape as Phase 18 Step 1: B, B_match, RepairKV, K=64/96/128.
Confirm Δ_score(RepairKV − B_match) shows the same sign as MQ-NIAH.
If sign flips on the smoke, stop and investigate before paper-grade
runs.

Estimated cost: ~15 min GPU + interpretation.

### W4 — Paper-quality run

n=24, K=96 (or K-sweep if the smoke supports it), 6 conditions
(B_match, RepairKV, Refresh-K-budgeted, TM-Recompute-BM25,
PageSummary-Quest-inspired, plus Condition A as ceiling). Same
statistical pre-registration as Phase 18 (Wilcoxon, HL CI, signed-
rank TOST at margin 0.20).

Estimated cost: ~75 min GPU (single-K) to ~110 min (K-sweep).

### W5 — Paper edit

One additional row or table in Results, plus a sentence in the
abstract qualifier:

> "...consistent on a non-needle long-context benchmark (SCBench
> multi-turn QA) and across three models (Qwen, Llama, Mistral)."

Or, if results are mixed, a one-paragraph appendix note.

## Acceptance criteria

Same as Phase 18 strong / weak / fail tiers, applied to the
RepairKV vs PageSummary-Quest-inspired contrast at K=96.

## Sequencing

```
Day 1 — W1 + W2 (loader, adapter, evaluator, audit hooks)
Day 1 evening — W3 smoke; investigate if sign-flip
Day 2 — W4 paper-quality run
Day 2 — W5 paper edit + recompile + Phase 17 rg checks
```

## Open questions for next session

1. **One non-NIAH benchmark or two?** SCBench primary; should RULER VT
   be added as a confirming point, or saved as future work?
2. **Single-K or K-sweep?** Single-K (K=96) is one paper-quality run
   matching Phase 18 multi-model shape. K-sweep is more compelling
   but doubles the GPU time.
3. **Which SCBench subtask?** SCBench has retrieval, QA, summarization,
   and in-context learning sub-benchmarks. QA is the natural primary
   but smoke may steer differently.

These are deliberately open; they should be answered with smoke
results in hand, not pre-committed now.

## Risks

1. **SCBench questions may not be hard enough at 32K to show
   compression damage.** If B_match is already at ceiling, there is
   nothing for repair to fix. Mitigation: smoke checks the B vs A
   gap before paper-quality runs commit.
2. **Two-turn adapter introduces bias.** If the Q1 question is too
   close to Q2's content area, the compressor retains Q2-relevant
   rows and B_match looks higher. Mitigation: pre-register the
   Q1/Q2 selection rule (e.g., "Q1 is the earliest question whose
   subject differs from Q2's subject") and document.
3. **Eval is noisy on free-form QA.** Short-answer exact match is
   cleanest; F1 introduces tokenizer/normalization noise.
   Mitigation: report both, headline exact-match.
