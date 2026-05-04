# Real-Repository Diagnostic Design

This note records the Phase 14 scaffold that led to Phase 15. The active plan is
now maintained in `phases/phase15_real_repo_relevance_shift/phase15_plan.md`.

## Reviewer Question

Does pre-resume KV repair help when the next turn is driven by a realistic
repository cue rather than synthetic MQ-NIAH question text?

The target evidence is not SWE-bench, WebArena, patch generation, or tool-use
success. The target is a controlled mechanism diagnostic: after a repository cue
shifts relevance to a different region of an already-seen context, can IdleKV
restore the needed evicted rows better than matched no-repair and
content-agnostic restore controls?

## Current Status

The original RepoDelta-Retrieval scaffold generated exact identifier questions
from real repository file cards. Expert audit found that the path/line variant
is too close to line lookup for a main-paper claim. It remains useful for CPU
tests and full-context ability smokes, but the paper-facing Phase 15 candidate
is now RepoDelta-Edge.

RepoDelta-Edge should use a Python-only, one-hop static relation such as
`anchor function + callsite -> leaf callee identifier`. The cue names the
repository event or implicated region without leaking the answer. Q2 asks for a
short exact identifier from the already-rendered context.

## Mechanism Constraint

Paper-facing Phase 15 evidence must separate the repair cue from the downstream
question:

1. Prefill the frozen repository context.
2. Instantiate the post-Q1 active cache and offloaded warm store.
3. Score evicted rows using only the pre-turn-2 event cue.
4. Restore K rows.
5. Decode the same final `event + Q2` prompt for all conditions.

The existing Phase 6 path scores with the full Q2 prompt, so Phase 15 needs a
dedicated event-only cue path before any main-paper GPU run. If only a full-Q2
repair path is run, the result must be described as next-turn-prompt-conditioned
repair, not tool-event-conditioned repair.

## Required CPU Gates

- Pinned third-party repository snapshots and frozen manifests.
- Strict identifier scoring, not substring or case-insensitive matching.
- Tokenizer-aware rendered-length, span, uniqueness, and tail-leakage audits.
- Q1/Q2 from different files or regions.
- Q2 answer absent from the cue and final question text.
- Gold answer appears exactly once in the rendered context.
- Stale-cue and wrong-event controls available for repair specificity.

## Promotion Gate

Promote only a clean locked Phase 15 result. If RepoDelta-Edge fails and the
line-location fallback is merely positive, keep it in appendix or future-work
notes rather than weakening the main paper.
