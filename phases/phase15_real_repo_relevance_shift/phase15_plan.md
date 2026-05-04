# Phase 15 Real-Repository Relevance Shift

Last updated: 2026-05-04 19:05 UTC.

## Goal

Phase 15 targets the paper's highest-value remaining evidence gap: the main
quality results are controlled and rigorous, but they are synthetic MQ-NIAH.
The goal is one controlled real-repository diagnostic that keeps the core
IdleKV causal structure: a full context has already been prefetched, the active
cache is compressed after turn 1, a new pre-turn-2 cue shifts relevance, and
repair restores a small number of evicted rows under a matched resumed active
budget.

This is not a new repository-level coding benchmark, not SWE-bench-lite, not a
tool-use benchmark, and not an end-to-end agent evaluation. It is a mechanism
study over frozen real repository text: which evicted rows should re-enter the
active KV cache once the next-turn cue is known?

## Converged Experiment

Primary paper candidate: **RepoDelta-Edge**.

RepoDelta-Edge uses public, pinned Python repositories and constructs examples
from a simple static edge, such as caller/callsite to callee or tool-event stack
frame to implicated implementation region. The answer is a short identifier
that appears in the already-seen repository context, not in the tool event and
not in the Q2 wording. Q1 asks about an unrelated file, biasing the compressed
post-Q1 cache away from Q2. A tool-event-like cue then names the new file/region
or anchor. IdleKV may use that cue to score evicted rows and restore K rows
before Q2 decoding.

Primary implementation target:

- Python-only.
- One-hop structural relation only: `anchor function + callsite -> leaf callee
  identifier`, with `exception class` or `uppercase config constant` as reserve
  answer types if callsite yield is too low.
- Exact identifier output only.
- 32K rendered context of line-numbered file cards from a single pinned repo
  snapshot.
- Q2 does not restate path or line. It asks about the reported failure/callsite
  location or implicated helper.

Fallback/smoke candidate: **RepoDelta-EventLoc**.

EventLoc uses the existing line-location scaffold after hardening. It is useful
for ability smokes and may become appendix evidence, but it is weaker because a
path/line cue can look like passkey retrieval over real code. It should not be
the headline main-paper result unless the stronger Edge variant fails and
EventLoc is exceptionally clean.

Secondary lanes:

- TestLog->Source is mechanistically attractive but too brittle for this
  deadline unless built on top of a working Edge manifest. It is the only
  higher-upside challenger, because a failing test or stack trace is the most
  natural agentic cue, but leakage and trace alignment make it risky.
- TraceSched-Repair is a complementary systems lane, not a replacement for real
  content. It would replay repair under recorded or predeclared tool-wait
  windows to test whether score/select/promote can finish before resume.
  It answers a different question from RepoDelta-Edge and should not be used to
  close the synthetic-only quality gap.
- DocDelta-Anchor, especially a version-pinned `KubeDelta-Flag` style task over
  technical reference docs, is the best non-repo fallback if code ability or
  Edge yield fails. It is easier to audit but weaker because reviewers may read
  it as document lookup.
- RealRepo MultiTurn Revisit changes too many axes at once and has no explicit
  tool-event cue; keep it as separate multi-turn evidence/future work.
- BuildManifest Delta is a viable emergency fallback if code ability fails, but
  lower signal for the paper's code-agent motivation and leakage-prone.
- SWE-bench/WebArena/trace replay are out of scope for Phase 15 because they
  introduce retrieval, exploration, patching, tool choice, and test-execution
  confounds.

## Mechanism Protocol

The main Phase 15 result must use event-only repair.

Timeline for every condition:

1. Prefill the frozen 32K repository context.
2. Decode Q1 or otherwise instantiate the post-Q1 active state.
3. Compress/offload to `B_base`, producing active cache `C^B` and evicted warm
   store `E`.
4. Observe the pre-turn-2 cue `T` from a tool event or stack/callsite signal.
5. Repair conditions score only `E` using `T`, restore K rows, and freeze the
   repaired cache.
6. Decode the identical final `event + Q2` prompt for all conditions.

Artifact fields must make this explicit:

- `repair_signal_mode=event_only|event_plus_q2|stale_event|wrong_event|none`
- `decode_prompt_mode=event_plus_q2`
- `source_task=repodelta_edge|repodelta_eventloc|docdelta`

`Event+Q2-K` may be run as an appendix or boundary ablation, but if it is the
only positive result, the paper must call it next-turn-prompt-conditioned repair,
not tool-event-conditioned repair.

Primary estimand:

`Delta_K = E[Y(IdleKV_event,K) - Y(B_match,K)]`, where Y is strict exact Q2
identifier accuracy and `B_match` has the same resumed active budget.

Specificity estimand:

`Gamma_K = E[Y(IdleKV_event,K) - Y(WrongEvent_or_StaleEvent,K)]`.

## Relation To Prior Work

Phase 15 borrows benchmark hygiene from repository-level code evaluations, but
must not claim to replace them.

- CrossCodeEval, RepoBench, RepoEval/CodeRAG, and ReCUBE test repository-level
  context use for retrieval, completion, or generation. Phase 15 instead tests
  whether an already-seen but compressed active KV state can be repaired after a
  relevance shift.
- SWE-bench evaluates issue-to-patch success on real repositories. Phase 15
  avoids patch generation and test execution so the cache intervention remains
  identifiable.
- BFCL evaluates tool-call correctness and stateful function invocation. Phase
  15 uses a tool-event cue only as an externally supplied relevance signal.
- Continuum asks when paused KV should remain resident. Phase 15 asks which
  evicted rows should re-enter active cache after a cue.
- SideQuest studies online model-driven context pruning. Phase 15 studies
  deterministic pre-resume repair of a paused, compressed cache.

Paper wording should use "controlled real-repository relevance-shift diagnostic"
and avoid "repository-level coding benchmark," "SWE-bench-style evaluation,"
"tool-use benchmark," and broad "practical coding performance" claims.

## Prior Phase Closure

Phase 15 starts from a finalized core paper. Previous branches should remain in
one of four states: main evidence, appendix/prose evidence, future work, or
discarded negative result.

| Branch | State | Paper role |
| --- | --- | --- |
| 2Q/4Q/6Q/8Q matched-budget frontier | finalized | main Figure 2 evidence |
| Next-turn specificity controls | finalized | main Figure 3 evidence |
| Five-turn relevance-shift diagnostic | finalized | main Figure 4 evidence |
| Runtime-capacity envelope | finalized | main Figure 6 evidence |
| Controlled proxy scorer | finalized | appendix figure plus main latency prose |
| Llama same-protocol portability | finalized | cautious main prose plus appendix figure |
| Retention-rule breadth | finalized but lower priority | main Figure 5 unless Phase 15 replaces it |
| SpanRef-K terminology | finalized | appendix diagnostic only, not an upper bound |
| Coverage selector | positive but not promoted | algorithmic-selection-gap future/appendix evidence |
| Refresh-buffered | finalized as method boundary | comparator/boundary evidence, not a deployable baseline |
| Quantized row-store promotion | negative | future work only |
| Qwen2.5-0.5B transfer smoke | negative/uninterpretable | discarded |
| Exact Scissorhands/H2O reproduction | deferred | future work unless named-baseline objection dominates |
| Trace-scheduled idle-window benchmark | deferred | future systems work |
| Off-device retention policy | deferred | discussion/future work |
| Theoretical best-recovery reference | deferred | no universal maximum claim; candidate-restricted appendix only if needed |

## Existing Code Readiness

Phase 14 already contains a CPU-tested RepoDelta generator:

- `phases/phase14_critical_flaw_closure/src/repodelta.py`
- `phases/phase14_critical_flaw_closure/tests/test_repodelta.py`
- `phases/phase14_critical_flaw_closure/real_agentic_task_design.md`

Current state after audit:

- File cards are line-numbered.
- Q2 no longer restates the exact path and line after the tool event.
- The generator is still only a scaffold; it is not GPU-ready paper evidence.
- Phase 6 cannot run RepoDelta through its normal synthetic-task registry.
- The current Phase 6 repair path scores with the full Q2 prompt, so Phase 15
  needs a separate event-only cue path before any main-claim GPU run.

## Required Implementation Before GPU

No GPU run, including ability smoke, should start until these exist and pass CPU
tests:

1. `phase15_protocol.json` with model revision, tokenizer revision,
   chat-template hash, repo registry hash, `B_base`, `R_ctx`, `K_grid`, primary
   `K*=96`, strict scoring rule, bootstrap seed, and allowed conditions.
2. Public third-party repo registry with pinned `repo_url`, `commit_sha`,
   license/SPDX, snapshot date, local archive SHA256, and split assignment.
   Do not use this repository, forks, mirrors, vendored code, generated code,
   benchmark fixtures, or floating branches.
3. Offline manifest builder. Each row records repo ID, commit, source paths,
   edge type, Q1/Q2 metadata, tool cue, gold answer, rendered token count,
   Q2 token span start/end, depth bin, answer token count, leakage flags, and
   stable example ID.
4. Tokenizer-aware CPU audits: rendered length within 32K, nonempty Q2 token
   span, gold answer token sequence appears exactly once at the annotated span,
   Q2 span outside the last `R_ctx + K_max` tokens, Q2 not already preserved by
   the matched base keep plan, and no answer token occurrence in the tool cue.
5. Strict identifier scorer: first decoded line after whitespace/backtick trim
   must match the gold answer byte-for-byte. No substring, case-folding, fuzzy
   matching, or judge model.
6. Phase 15 runner or wrapper that consumes frozen manifests and reuses cache
   partition/repair/scoring code while supporting `repair_signal_mode`.
7. Cluster-bootstrap analysis script: paired two-stage bootstrap by repo, then
   examples within repo, preserving shared examples across all conditions.
8. Unit tests for manifest determinism, AST edge extraction, leakage rejection,
   token-span mapping, strict scoring, stale/wrong cue construction, metadata
   persistence, and runner dry-run plumbing.

## Conditions

Required for any paper-facing run:

- `A`: full active cache.
- `B`: compressed post-Q1 active cache.
- `B_match`: no repair, same resumed active footprint as K-row repair.
- `Random-K`: random K rows from the same evicted warm store.
- `Oldest-K`: oldest K eligible rows from the same evicted warm store.
- `IdleKV-EventOnly-K`: main method, scores evicted rows using the tool cue only.
- `StaleCue-K`: restore using Q1/stale cue.
- `WrongEvent-K`: restore using a donor/wrong event cue.

Strongly recommended if runtime permits:

- `ToolFile-K`: restore rows only from file(s) named in the tool cue. This tests
  whether IdleKV is better than a simple file-pointer heuristic.
- `Event+Q2-K`: boundary ablation; not the main tool-event claim.
- `Refresh-buffered`: method-boundary comparator, not a matched baseline.

## Run Ladder

1. Build CPU manifests for RepoDelta-EventLoc and RepoDelta-Edge over 10-15
   public pinned Python repos.
2. Inspect yield. Preferred threshold for Edge: at least 150 clean candidate
   examples total and at least 8 repos contributing candidates.
3. Run full-context ability smoke on the dev split only: `n=5` for Edge and
   EventLoc, no repair claim.
4. If Edge has `A >= 0.80` and the manifest yield is healthy, make Edge the
   primary task. If Edge fails ability/yield, freeze EventLoc as fallback or
   pivot to DocDelta only if code ability fails entirely.
5. Run repair smoke on dev split: `n=5`, `K={48,96}`, required conditions plus
   stale/wrong cue controls.
6. Pilot on dev split: `n=24`, `K={32,48,64,96,128}`.
7. Freeze locked protocol and locked manifest. No prompt/budget/repo/row changes
   after this point except objective pre-run corruption.
8. Locked run: `n=64` as `16 repos x 4 examples` if possible. Escalate to
   `n=100` only if pilot variance is high and results are clean.

All GPU runs must be in tmux with timestamped logs.

## Promotion Gate

Main-paper promotion requires a locked run satisfying all of:

- Full-cache score `A >= 0.80`.
- Matched no-repair has a real gap: `A - B_match >= 0.15` at `K*=96`.
- Mean `IdleKV-EventOnly - B_match >= 0.10` at `K*=96` and nonnegative at an
  adjacent K.
- Bootstrap 95% CI lower bound is positive for `IdleKV-EventOnly - B_match`,
  `IdleKV-EventOnly - Random-K`, and `IdleKV-EventOnly - Oldest-K` at `K*=96`.
- `IdleKV-EventOnly` beats `StaleCue-K` and `WrongEvent-K` by a positive paired
  CI or at least a predeclared practical margin.
- Repo-level median lift is positive; no single repo dominates the result.
- Zero locked-manifest leakage/audit failures.
- Full-cache misses are not dominated by formatting/truncation.
- Q2 answer is not recoverable from the cue alone, does not appear in the cue,
  appears exactly once in the rendered context, and lies outside the recent tail.

If only EventLoc passes, the result is appendix or a cautious main paragraph by
default. If only Event+Q2 passes, frame it as next-turn-prompt-conditioned
repair and do not claim idle tool-event repair.

## Paper Action If It Passes

Preferred integration:

- Add one concise result paragraph.
- Add one one-column main figure only if the result is clean and visually strong.
- Replace the lower-priority retention-rule sensitivity figure if necessary.
- Keep frontier, specificity, synthetic multi-turn, and runtime-capacity figures
  in main.
- Caption must say "controlled real-repository relevance-shift diagnostic" and
  state that it is not end-to-end issue resolution.

## Paper Action If It Fails

Do not add noisy real-repository results to the main paper. Keep current
limitations: the core evidence is controlled and synthetic, and real workflow
validation remains future work. A clean negative can be summarized in appendix
or Phase 15 notes only if it teaches a clear design lesson.
