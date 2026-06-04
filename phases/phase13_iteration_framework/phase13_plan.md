# Phase 13 Iterative Closure Framework

Last updated: 2026-05-03 16:46:00 UTC.

## Purpose

Phase 13 is not a new evidence claim. It is the execution framework for
closing live research branches without wasting GPU time or adding weak paper
material.

Every unfinished idea must move through the same loop:

1. **Hypothesis.** State the reviewer question the idea answers.
2. **Failure analysis.** If a smoke or full run failed, identify whether the
   failure is task design, budget calibration, implementation, scorer behavior,
   model ability, or an actually negative result.
3. **Code readiness.** Any implementation change needs targeted unit tests
   before GPU work.
4. **Minimal smoke.** Run the cheapest smoke that can falsify the hypothesis.
   Smokes are design signals only, not paper evidence.
5. **Result-rigor critique.** Before promoting anything, check that it is not
   smoke-only, has enough paired examples and restore-budget points, audits the
   matched active-cache budget, has a strong full-cache reference, keeps
   controls clean, avoids saturation, has an effect size large enough to
   matter, and reports paired uncertainty when the result becomes a primary
   main-text claim.
6. **Promotion gate.** Lock the smallest full run that can support a
   paper-ready graph. Promote only if the result passes a pre-written gate.
7. **Graph-quality critique.** Judge the proposed figure separately from the
   result: real data only, correct graph type for the claim, one-column fit
   unless unavoidable, no legend/data collision, readable labels at ICML
   column width, controls visible, caption scoped to the evidence, and no
   redundant low-signal panels.
8. **Integration audit.** Add the graph only if it replaces or strengthens a
   main claim. Otherwise keep it appendix or future work.

## Branch Priorities

### P0: Active Policy-Breadth Run

- **Question:** Does repair survive a structurally different first-stage
  retention rule, not only the primary SnapKV-style retention rule?
- **Current result:** `phase12_streamingllm_full`, MQ-NIAH-4Q,
  sink-plus-recent retention, `B=16384`, `K=8..128`, `n=24`, passed the
  policy-curve gate.
- **Main gate:** full-grid pass, at least two adjacent restore budgets with
  RepairKV gain at least `0.15`, Random-K/Oldest-K within `0.10` of matched
  no-repair at those budgets, and Gold-K covering RepairKV.
- **Current figure:** one-column policy-breadth plot with SnapKV,
  accumulated-attention retention, and sink-plus-recent retention. The x-axis
  is score gain over each policy's matched no-repair baseline; gray marks
  Random-K/Oldest-K controls and orange marks Gold-K headroom.
- **Figure-quality audit:** main-worthy only as narrow retention-rule breadth.
  It is a full `n=24` K-grid, but it is still one task/model family and the
  accumulated-attention and sink-plus-recent rows are protocol-matched probes,
  not faithful reproductions of the full H2O or StreamingLLM systems. Do not
  broaden the claim without an exact prior-policy branch and new uncertainty
  reporting.
- **If a future variant fails:** keep it as appendix/future benchmark evidence;
  do not claim retention-rule generality.
- **Terminology gate:** name the mechanism first and cite the source as
  inspiration unless/until the implementation reproduces the full original
  algorithmic protocol. Our accumulated-attention branch scores a frozen
  post-Q1 cache by accumulated attention over recent observation rows; our
  sink-plus-recent branch keeps sink tokens plus a recent window inside the
  matched two-turn protocol. Both are useful robustness probes, but neither
  should be presented as a canonical reproduction of every systems detail in
  H2O or StreamingLLM.

### P1: Multi-Turn Main-Readiness

- **Question:** Can repair repeatedly adapt cache state across relevance shifts
  and revisits, closer to agent-style workflows than a single Q2 handoff?
- **Current result:** locked `n=24`, `K=80` passed the main gate and is
  integrated as a main diagnostic. Non-initial RepairKV gain is `0.542`
  with paired bootstrap interval `[0.458,0.620]`; revisit-turn gain is
  `0.938` with interval `[0.875,1.000]`.
- **Residual caveat:** stale-query-only repair still gains `0.234` on
  non-initial turns. The current-query-only margin over stale-query-only is
  `0.307` with interval `[0.240,0.370]`, so the result supports repeated
  controlled repair but not broad real-agent validation.
- **Saturation audit:** displayed turn 3 in the locked figure is the `[4,5]`
  query and is a genuine no-headroom turn, not an accounting bug: matched
  no-repair scores `1.0` on all 24 examples and retains mean active annotated
  span overlap `0.841`. The reported non-initial gain includes this saturated
  turn; excluding it raises RepairKV gain from `0.542` to `0.722`, so the main
  aggregate is conservative. The MQ-NIAH-8Q span geometry places the displayed
  turn-3 pair in a region that the `B=18432,K=80` no-repair cache already
  retains well; Random-K, Oldest-K, stale-query repair, and Gold-K all also
  score `1.0` on that turn, confirming that the saturation is task/budget
  geometry rather than an RepairKV-specific effect.
- **Challenge-schedule pilot:** `mq_niah_8q_challenge_revisit` removes the easy
  `[4,5]` middle pair and alternates `[0,1]` and `[2,3]` after the full-context
  priming turn. The `n=2`, `K={64,80,96}` smoke completed on 2026-05-03 and
  should not replace the locked main diagnostic: `K=64` and `K=80` leave only
  `0.375` and `0.625` revisit gain, respectively, and `K=96` is fully closed by
  stale-query controls. Treat this as task-design evidence, not a failed
  implementation.
- **Calibration smoke:** completed `n=2`, `K in {64,80,96}` on 2026-05-03.
  The gate selected `K=80`: RepairKV non-initial gain `0.625`, revisit gain
  `1.0`, content-agnostic controls `0.0`, stale fraction `0.30`, and
  current-query-only minus stale-query-only margin `0.438`. `K=64` was too
  weak and `K=96` was too explainable by stale-query controls.
- **Locked run:** completed in tmux session `phase13_multiturn_locked` on
  2026-05-03 and wrote rows, summary, raw JSON, and paired-uncertainty CSVs.
- **Main gate:** RepairKV non-initial gain at least `0.35`, revisit gain at
  least `0.75`, content-agnostic controls near zero, StaleQ-K no more than
  `45%` of RepairKV non-initial gain, `CurrentQOnly-K` at least `0.10`
  non-initial gain above `StaleQOnly-K` when those diagnostics are present,
  and paired rows auditable for matched active-cache budget.
- **Paper integration:** promoted the multi-turn trajectory to main and moved
  the operating-regime heatmap to the appendix. Keep the caption and prose
  scoped to controlled relevance shifts, matched active-cache accounting, and
  stale-query caveats.

### P2: Llama Main-Readiness

- **Question:** Is the effect visible outside Qwen?
- **Current result:** Llama-3.1-8B-Instruct 4Q full grid passed at `n=24`, but
  the curve saturates early. It is valid appendix portability evidence, not a
  main model-family robustness claim.
- **Completed smoke:** Llama-3.1-8B-Instruct, MQ-NIAH-6Q,
  `B=18432`, `n=2`, `K={32,64,96,128}`. Full-context score is `1.0`;
  matched no-repair stays at `0.5`; RepairKV reaches `1.0` for
  `K>=64`; Random-K and Oldest-K stay at matched. Exclude `K=32` from
  locked promotion because RepairKV exceeds the metadata \goldk-style
  reference at that low budget, so the reference is not a clean ceiling.
- **Locked run:** `phase13_llama6q_locked` completed on 2026-05-03 with
  Llama-3.1-8B-Instruct, MQ-NIAH-6Q, `B=18432`, `n=12`, and
  `K={64,96,128}`. It passes the appendix portability gate: full-cache
  score `1.0`, matched no-repair `0.472`, RepairKV `1.0` at all three
  budgets, Random-K/Oldest-K at or below matched, and \goldk-style
  metadata reference covering RepairKV. Promote as appendix cross-model
  evidence, not as a broad multi-model main claim.
- **Main gate:** full-cache reference at least `0.90`, matched no-repair gap at
  least `0.20`, non-saturated RepairKV improvement over matched and controls at
  two or more K values, and Gold-K covering RepairKV.
- **If pass:** lock `n=12` or `n=24` depending on runtime and variance; render
  as a compact cross-model row plot or move model-transfer from appendix to a
  short main robustness paragraph plus figure.
- **If fail or saturate:** keep current Llama 4Q appendix figure.

### P3: Selector Variants

- **Question:** Is there an easy algorithmic improvement over the current
  exact-Q top-K selector?
- **Current status:** coverage-aware and MMR-style variants are implemented
  with CPU tests, but not promoted.
- **Next smoke:** run `K={24,48,96}`, `n=1` first. Scale only if a variant
  beats current RepairKV by at least `0.05` at mid-K without hurting high-K by
  more than `0.02`.
- **Paper gate:** only promote if it gives a clear algorithmic story and
  reduces the Gold-K gap. Otherwise avoid ablation clutter.

### P4: Quantized/Precision Repair

- **Question:** Can idle repair promote selected low-precision KV rows after
  relevance changes?
- **Current result:** first row-store quantization sweep is negative.
- **Next action:** no full GPU run until a redesigned page/channel-aware
  baseline exists and unit tests cover byte accounting. Keep as future work.

### P5: Faithful Prior-Policy Reproductions

- **Question:** Would exact H2O or StreamingLLM reproductions strengthen the
  policy-breadth story enough to justify a new implementation branch?
- **Detailed audit:** `exact_policy_audit.md`.
- **Current decision:** no immediate full run. The existing Figure 5 branches
  are protocol-matched first-stage retention rules, not full systems
  reproductions.
- **Named algorithms that can be faithful in this repo:** H2O and
  Scissorhands are the best candidates because both are fixed-budget
  eviction policies based on accumulated/persistent attention importance.
  Either requires logging actual model attention scores during Q1/answer
  generation, a unit-tested score accumulator, and a smoke that reproduces
  the expected keep-order behavior before any full grid.
- **Best exact-algorithm next branch if needed:** Scissorhands. It is closer
  to the current fixed-budget, post-compression retention setting than
  full StreamingLLM, and unlike PyramidKV it does not require layer-varying
  active-position sets. The smoke should first test a deterministic attention
  trace, then MQ-NIAH-4Q with `n=2`, `K={48,96}`, matched no-repair,
  Random-K/Oldest-K, RepairKV, and Gold-K.
- **Named algorithms that are possible but lower priority:** PyramidKV would
  add layer-varying budgets, which would complicate our matched resumed
  active-cache budget and distract from the repair claim. QUEST is
  query-aware page loading during attention, closer to our repair scorer than
  to the first-stage retention rule, so it belongs in future algorithm work unless
  we implement page-level repair. FastGen is also lower priority for this
  paper because a faithful run requires its profiling stage and head-specific
  retention patterns, while the current repair implementation assumes a single
  global retained-position set.
- **StreamingLLM path:** not a priority as a canonical reproduction.
  StreamingLLM's sink-plus-recent idea is exactly the structural control we
  need, but the full method targets rolling streaming generation and
  position/cache management outside this two-turn matched-budget protocol.
  Keep the current branch named "sink-plus-recent" or "sink-plus-recent
  retention inspired by StreamingLLM", not "StreamingLLM".
- **Promotion rule:** if we add an exact prior-policy run, call it exact only
  after tests cover the algorithmic invariant and the paper states the
  algorithm-specific budget accounting. Otherwise use "inspired by" language
  and avoid implying a faithful systems reproduction.

## Execution Rules

- Never start a full GPU run until the corresponding smoke passed.
- Never use smoke data in the main paper.
- Run long jobs in tmux with explicit result paths.
- Prefer CPU/unit-test work while GPU runs are active.
- If a branch fails a promotion gate, record why before redesigning it.
- A main result must pass both result-rigor and graph-quality gates. A clean
  result with a weak figure gets redesigned before promotion; a good-looking
  figure with weak evidence goes to appendix or is dropped.
- A main figure must answer a distinct reviewer question; otherwise put it in
  appendix or prose.
- Rebuild `paper/main.pdf` after every paper or figure edit.
- Before major paper rewrites, read `paper_guide.md` as the binding
  style/terminology/format contract, then read `paper/outline.md` as a
  directional narrative guide. The outline is not ground truth: preserve its
  flow only where it still matches the latest evidence, reviewer risks, and
  concision goals.
- Before repository handoff, write a concise top-quality `README.md` that
  explains the paper claim, code layout, how to reproduce the main figures,
  and which runs are locked paper evidence versus exploratory branches.
- Commit and push only after the active paper figures, PDF rebuild, tests, and
  long-running tmux result audits are clean enough to snapshot.

## Expert-Audit Decisions For The Current Closure Pass

- If the locked multi-turn `n=24`, `K=80` run passes the numerical gate and
  paired uncertainty is reported, promote the five-turn relevance-shift figure
  over the operating-regime heatmap. Multi-turn evidence is closer to the
  paper's dynamic workflow thesis; the heatmap is calibration context and can
  move to the appendix.
- A promoted multi-turn figure should show only the reviewer-critical traces:
  RepairKV, StaleQ-K, Gold-K, and a Random-K/Oldest-K control band. Keep
  CurrentQOnly-K and StaleQOnly-K in the audit/prose for specificity, not as
  extra plot lines unless the figure remains readable.
- The multi-turn caption/prose must report the schedule, `K`, `n`, paired
  uncertainty, stale-query separation, and matched active-cache accounting.
  It must not claim broad agent-workflow validation.
- Keep policy breadth framed as first-stage retention-rule breadth, not exact
  prior-baseline breadth. The main text may use it only as narrow robustness
  evidence; otherwise move it to the appendix.
- If there is time for one exact named eviction baseline after the current
  run, prioritize Scissorhands over exact StreamingLLM. Scissorhands better
  matches the fixed-budget two-turn protocol; exact StreamingLLM would require
  a rolling streaming/position-management setup that is outside the current
  matched-budget repair protocol.

Scissorhands branch design, if opened:

1. Add attention-trace capture during turn-N generation; do not approximate it
   with key-dot-product scores and call it Scissorhands.
2. Implement the paper's fixed-buffer update rule rather than a generic top-K
   accumulated-attention selector.
3. Unit-test the fixed-budget invariant on toy attention traces: persistent
   high-importance rows survive, sink/recency rows obey the explicit budget,
   ties are deterministic, and evicted rows enter the offloaded store.
4. CPU smoke the selector on synthetic traces before any model run.
5. GPU smoke only after tests pass: MQ-NIAH-4Q, `n=2`, `K={48,96}`, matched
   no-repair, Random-K, Oldest-K, RepairKV, and Gold-K.
6. Promote only after a locked grid shows adjacent positive RepairKV gains,
   clean content-agnostic controls, and no budget-accounting ambiguity.

Failure-response rule for `phase13_multiturn_locked`:

- If RepairKV is strong but StaleQ-K exceeds the stale-fraction gate, do not
  promote the figure. Record the result as evidence that this schedule still
  contains reusable stale-query signal, then redesign the schedule with more
  disjoint revisits or stronger stale-query distractors before any rerun.
- If Random-K/Oldest-K exceed the content-agnostic control gate, treat the
  operating point as too easy or too underconstrained. Recalibrate budget or
  task difficulty before rerunning.
- If RepairKV itself falls below the non-initial/revisit gain gate, demote
  multi-turn to future benchmark design and keep the current main package.
- Do not tune wording or graph styling to make a failed multi-turn run look
  main-ready. The gate decides promotion.

## Current Queue

1. Multi-turn locked branch: completed, passed, and integrated into the main
   paper with paired uncertainty.
2. Exact prior-policy branch: do not start unless the paper specifically needs
   one faithful named baseline; expert audit split between exact H2O for name
   recognition and Scissorhands for protocol fit.
3. Llama 6Q locked portability evidence is completed and appendix-integrated;
   selector-variant smokes remain optional follow-ups, not current-paper
   blockers.
4. Quantized/precision repair remains future work until byte-accounted
   page/channel-aware designs are implemented and tested.

## Implemented Framework Artifacts

- `src/framework.py`: CPU-only gates for first-stage-policy curves,
  multi-turn summaries, result-rigor checks, and graph-quality checks. The
  multi-turn gate now rejects a candidate when current-query-only and
  stale-query-only controls are too close.
- `scripts/audit_live_branches.py`: reads live result CSVs and prints
  branch-level actions, result-rigor decisions, paper-artifact figure-quality
  decisions, and terminology caveats for non-canonical policy variants.
  Main multi-turn candidates now require positive paired bootstrap lower
  bounds for RepairKV over matched no-repair, RepairKV over Random-K/Oldest-K, and
  CurrentQOnly-K over StaleQOnly-K on non-initial turns.
- `scripts/summarize_multiturn_uncertainty.py`: writes paired bootstrap
  intervals for multi-turn rows. Primary main-text multi-turn claims require
  this CSV before promotion.
- `scripts/postprocess_multiturn_locked.py`: locates a locked multi-turn
  summary, derives the matching rows/raw paths, and writes the uncertainty CSV
  with the filename expected by the live audit.
- `scripts/run_multiturn_hard_kcal_smoke.sh`: next multi-turn K-calibration
  smoke, parameterized by environment variables. Its default condition set now
  includes `CurrentQOnly-K` and `StaleQOnly-K` so the smoke can diagnose
  whether stale-query strength comes from the previous query itself or from the
  old importance-score tie-breaker.
- `scripts/run_multiturn_hard_locked.sh`: tmux-ready locked follow-up for the
  hard multi-turn schedule if the calibration smoke passes the query-only gate.
- `scripts/run_llama31_8b_6q_smoke.sh`: harder Llama smoke that selected the
  appendix portability branch.
- `scripts/run_llama31_8b_6q_locked.sh`: tmux-ready locked Llama 6Q follow-up;
  the completed `n=12`, `K={64,96,128}` run passed the appendix gate.
- `tests/test_framework.py`: unit tests for endpoint-only rejection,
  multi-point promotion, result-rigor gates, figure-quality gates, and
  stale-query multi-turn rejection.
- `tests/test_uncertainty.py`: unit tests for paired gain/difference matching,
  deterministic bootstrap intervals, multi-turn uncertainty summaries, the
  uncertainty CLI, and locked-run postprocessing.
- `tests/test_paper_language.py`: regression guard that prevents internal run
  nicknames and ambiguous "oracle"/"matched footprint" language from returning
  to the paper body.

Validation on 2026-05-03: repo-wide pytest passes with `202 passed,
16 warnings, 304 subtests passed`. Targeted paper/closure tests pass with
`37 passed`, and the broader CPU-side subset passes with `157 passed,
16 warnings, 240 subtests passed`.
