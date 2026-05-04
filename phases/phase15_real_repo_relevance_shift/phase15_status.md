# Phase 15 Status

Last updated: 2026-05-04 19:05 UTC.

## Current State

Phase 15 is open, but no GPU runs have started and no tmux sessions are active.
After expert audit, the original RepoDelta line/path lookup is no longer the
paper-facing plan. It is now a smoke/fallback scaffold.

The converged paper-facing candidate is **RepoDelta-Edge**: a controlled
real-repository relevance-shift diagnostic over pinned Python repos, using a
single static edge such as `anchor function + callsite -> leaf callee
identifier`. The main repair signal must be **event-only**: repair sees the
pre-turn-2 tool/callsite cue, restores K rows from the warm evicted store, and
then all conditions decode the same final `event + Q2` prompt.

## Expert Audit Summary

Critical flaws found and resolved into the plan:

- Plain EventLoc/path-line lookup is too close to passkey retrieval over real
  code. It is useful for ability smoke, not the headline main result.
- Current Phase 6 scoring uses the full Q2 prompt. A main Phase 15 tool-event
  claim requires a separate `repair_signal_mode=event_only` path.
- Q2 must not restate path or line. The scaffold now asks about the reported
  failure location rather than repeating `path + line`.
- Frozen manifests are required before GPU. On-the-fly repo sampling is not
  publishable evidence.
- Strict identifier scoring is required. Existing substring/case-insensitive
  scoring is not valid for code identifiers.
- Token-level audits, not raw character audits, must reject tail leakage,
  duplicate answers, overlength prompts, and Q2 spans already preserved by the
  matched base cache.
- Stale/wrong cue controls are mandatory for causal specificity.
- Related-work framing must stay narrow: this is not a repository coding
  benchmark, SWE-bench slice, BFCL task, or scheduler/JCT evaluation.
- A broad creative-eval sweep did not find a stronger primary than
  RepoDelta-Edge. The best challenger is TestLog->Source, but it is riskier;
  the best non-repo fallback is DocDelta-Anchor/KubeDelta-Flag; the best
  complementary systems lane is TraceSched-Repair.

## Prior Branch Audit

- Main paper evidence is currently finalized for the MQ-NIAH frontier,
  specificity, multi-turn, retention-rule sensitivity, and runtime-capacity
  figures.
- Appendix/prose evidence is finalized for controlled proxy scoring, Llama
  portability, SpanRef-K diagnostics, and supplementary mechanism/endpoint
  diagnostics.
- Positive but not promoted: Coverage selector. It shows algorithmic selection
  headroom but lacks a dedicated full K-grid paper figure.
- Boundary evidence: Refresh-buffered shows stronger full-budget reselection can
  outperform incremental IdleKV. It is a method-boundary comparator, not a
  deployable baseline or theoretical optimum.
- Negative/deferred: quantized row-store promotion, Qwen2.5-0.5B transfer,
  exact named eviction reproductions, trace-scheduled idle-window benchmark,
  off-device retention policy, and candidate-restricted best-recovery
  references.

## Completed In This Pass

- Added Phase 15 plan/status files.
- Updated README to show Phase 15 as the active open question.
- Fixed RepoDelta file cards to include visible line numbers.
- Changed Q2 wording so it no longer restates the exact path and line after the
  tool event.
- Added/updated RepoDelta unit tests for line visibility and Q2 wording.
- Ran focused RepoDelta tests successfully.

## Immediate Next Tasks

1. Add a Phase 15 manifest package around the Phase 14 generator.
2. Implement strict identifier scoring and failure taxonomy.
3. Implement tokenizer-aware manifest audits.
4. Add Python AST extraction for the Edge candidate: single-line callsite to
   leaf callee identifier, plus minimal fallback answer types if yield is low.
5. Add stale/wrong cue construction and metadata fields for
   `repair_signal_mode` and `decode_prompt_mode`.
6. Add a Phase 15 runner or wrapper that consumes frozen manifests and supports
   event-only repair. Do not route this through Phase 6 unchanged.
7. Run CPU tests for all new code.
8. Only then build dev manifests and run full-context ability smokes in tmux.
9. If Edge fails yield or full-context ability, pivot to DocDelta-Anchor rather
   than browser/chat/notebook traces. If Edge works and time remains, consider a
   small TestLog->Source manifest as a challenger or TraceSched-Repair as a
   systems complement.

## Promotion Rule

Phase 15 enters the main paper only if a locked RepoDelta-Edge run is clean under
the written gate in `phase15_plan.md`. If Edge fails but EventLoc passes, use it
only as appendix/cautious corroboration unless the result is unexpectedly strong
and the limitations are explicit. A failed or messy real-repository result stays
out of the paper.
