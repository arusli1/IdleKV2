# Phase 14 Critical-Flaw Closure Plan

Last updated: 2026-05-03.

## Purpose

Phase 14 exists to close the paper risks that would make a strong reviewer
discount the current evidence. It is not a broad benchmark expansion. Every
run must answer one of the critical-flaw questions below and must be designed
backward from a paper object or a deliberate paper edit.

## Full Idea Inventory

Phase 14 should test every idea that could materially change the paper, but it
should still test them in a compute-rational order. Each branch below has a
minimal falsifying smoke, a promotion gate, and a paper action. A branch can end
in three valid states: promote, appendix/future-work, or discard with a written
failure analysis.

## Phase 14 Test Loop

Every idea must pass through the same loop before it can affect the paper:

1. **Reviewer question.** State the precise flaw or opportunity the idea tests.
2. **Paper object.** Decide the intended figure, table, appendix diagnostic, or
   text edit before running.
3. **Code readiness.** Add or reuse a script with deterministic arguments and
   unit tests for any new logic.
4. **Minimal smoke.** Run the cheapest GPU/CPU probe that can falsify the idea.
5. **Automated gate.** Evaluate the smoke with a script, not by visual
   preference alone.
6. **Decision.** Promote to locked run, demote to appendix/future work, discard,
   or redesign the experiment and return to step 1.
7. **Locked run.** Only locked runs, not smokes, can enter the main paper.
8. **Paper audit.** A promoted result must replace or sharpen an existing claim;
   it cannot simply add another low-density figure.

The loop is deliberately conservative: Phase 14 tests all high-signal ideas, but
it should not expand the main paper unless an idea closes a named flaw better
than the current evidence.

### Priority Classes

- **Must-test before final paper claims:** scalable proxy repair, Refresh-K
  boundary, calibrated cross-model evidence.
- **Test if earlier branches leave the story under-supported:** exact named
  prior-policy reproduction, harder multi-turn schedule, runtime/idle-window
  trace evidence.
- **Keep as future work unless a smoke is surprisingly positive:** quantized
  row-store repair, selector variants, realistic agent/tool benchmark.

### Expert-Audit Constraints Incorporated

- The main paper should stay claim-first and compact. Extra evidence should
  replace weaker evidence or move to the appendix.
- The canonical system framing is a turn-conditioned promotion operator from a
  slower host/off-device KV tier into the resumed active GPU cache.
- Refresh-K should not be hidden. If it dominates, it defines a boundary:
  incremental repair is not full-budget reselection.
- Cross-model evidence must be non-saturated before it can support a main-paper
  generality claim.
- Inspired retention rules are not faithful named-policy baselines. A named
  policy claim requires an exact implementation and invariants.
- Runtime evidence should separate exact research scoring from scalable
  score/select/promote mechanics and should avoid implying real idle-window
  trace coverage unless the data are trace-backed.

## Critical-Flaw Questions And Experiment Branches

### P0: Does the scalable repair path preserve the main effect?

**Risk.** The main quality frontier uses exact query projections, while the
latency argument uses cheaper proxy scoring. A reviewer can accept exact
scoring as mechanistic evidence, but scalable-scorer language needs evidence
that the cheaper scorer keeps quality under the same controls.

**Current audit.**

- Existing proxy artifacts are useful but not main-ready: they include
  `A/B/B_match/IdleKV` only, not `Random-K`, `Oldest-K`, or `Gold-K`.
- 4Q proxy at K=96 passes the speed/quality gate and even exceeds exact quality.
  That is a positive signal, but it also means proxy should be described as a
  heuristic scorer, not as a strict approximation to exact query projections.
- 6Q proxy at K=96 keeps strong quality but narrowly misses the strict retained
  exact-gain gate (`0.833` versus the current `0.850` threshold).

**Target paper object.** One compact appendix or main-text quality-latency
frontier with exact IdleKV, proxy IdleKV, matched no-repair, content-agnostic
controls, and Gold-K headroom.

**Smoke.** 4Q and 6Q, `K={48,96,128}`, `n=4`, proxy scoring,
`A/B/B_match/Random-K/Oldest-K/IdleKV/Oracle-K`.

**Promotion gate.**

- Proxy IdleKV gain at K=96 is at least `0.10`.
- Proxy retains at least `0.80` of exact gain at K=96 for 6Q and at least
  `0.85` for 4Q.
- Proxy p50 total repair time is at least `3x` faster than exact.
- Random-K and Oldest-K do not close more than `0.10` of the IdleKV gain at
  the promoted operating point.
- Gold-K covers IdleKV or the violation is explained as a gold-span reference
  limitation.

**Locked run if smoke passes.** Repeat with `n=100` for 4Q/6Q at the smallest
K-grid that makes the graph honest. Prefer `K={48,64,80,96,128}` if time allows;
otherwise `K={48,96,128}`.

### P1: Is Refresh-K a boundary result or a stronger algorithm?

**Risk.** The specificity panel shows that full-budget Q2-time reselection
(`Refresh-K`) reaches the Gold-K reference at the current K=48 point. If the
paper frames IdleKV as the best repair algorithm, this is a serious weakness.
If the paper frames IdleKV as an incremental, low-mutation repair primitive,
Refresh-K becomes useful headroom evidence.

**Target paper object.** Either:

- a short main-text caveat explaining Refresh-K as an upper-bound/full-refresh
  comparator, or
- an appendix frontier showing how often Refresh-K dominates and what it costs.

**Smoke.** 4Q, exact scoring, `K={24,48,80,96}`, `n=2`,
`A/B/B_match/StaleQ-K/WrongQ-K/Refresh-K/IdleKV/Oracle-K`.

**Promotion gate.**

- If Refresh-K dominates IdleKV for most of the tested K values, do not hide it. Reframe the
  method section and results around incremental repair versus full reselection.
- If Refresh-K is only dominant in the current specificity point, keep the
  single-point figure but explain the boundary clearly.
- Promote a Refresh frontier only if it changes the paper claim; otherwise keep
  it as an audit artifact.

### P2: Is there enough cross-model evidence?

**Risk.** Llama evidence currently saturates at `1.0` across the locked K-grid.
That is useful portability evidence, not a broad model-family claim.

**Next action.** Do not run a larger Llama grid until a smoke finds a
non-saturated setting with at least two adjacent useful K values. Candidate
smoke: Llama-3.1-8B-Instruct, MQ-NIAH-6Q, lower base budget or lower K-grid,
`n=2`, `K={24,32,48,64}`.

**Promotion gate.** Full-cache score at least `0.90`, matched no-repair gap at
least `0.20`, IdleKV improves over matched and content-agnostic controls at two
K values, and Gold-K covers IdleKV.

**Locked run if smoke passes.** Repeat the calibrated non-saturated setting
with `n=24` and the same K-grid using
`run_llama_calibrated_locked.sh`. This is the minimum bar for moving Llama from
appendix portability to a main-paper cross-model claim.

### P3: Should policy breadth use an exact named prior policy?

**Risk.** Current retention-rule breadth is useful but not a faithful
reproduction of full H2O or StreamingLLM systems.

**Next action.** Do not start this until P0/P1 are resolved. If opened, use
Scissorhands first because it better matches the fixed-budget two-turn protocol.
Tests must cover the fixed-buffer update invariant before any GPU work.

**Smoke.**

1. CPU-only synthetic attention trace test: persistent high-attention rows
   survive, recency/sink rows obey their explicit budget, ties are
   deterministic, and evicted rows are written to the offloaded store.
2. GPU smoke only after the CPU invariant passes: 4Q, `B=16384`, `n=2`,
   `K={48,96}`, conditions
   `A/B/B_match/Random-K/Oldest-K/IdleKV/Oracle-K`.

**Promotion gate.** The branch can enter the main paper only if it is a faithful
named algorithm implementation and not merely an inspired retention rule. If
faithfulness is incomplete, keep the existing inspired-policy figure/prose and
state the limitation.

### P4: Are broad agent-system claims overextended?

**Risk.** The paper motivates dynamic agent workflows, but the primary task is
controlled split-query MQ-NIAH.

**Paper gate.** Until a more realistic task is added, the paper must call the
current benchmark a controlled relevance-shift diagnostic and avoid claiming
validated end-to-end agent performance.

**Smoke.** Design one realistic-ish diagnostic only after P0/P1/P2. The cheapest
candidate is a code/repo-style key-value retrieval task where Q1 asks about one
module and Q2 asks about a later module or revisits an earlier module after a
tool-output-like inserted segment. The smoke must prove the full-context model
can solve it and matched no-repair fails before any locked run.

**Promotion gate.** Main paper only if it adds a distinct claim beyond MQ-NIAH:
dynamic relevance after a tool/repo-like context update. Otherwise it belongs
in future work.

### P5: Does multi-turn repair need a harder or broader schedule?

**Risk.** Current multi-turn evidence is positive, but one displayed turn
saturates and stale-query repair recovers a nontrivial fraction of the gain.

**Current audit.** The locked hard-revisit result is main-ready as a controlled
diagnostic, but it is not broad agent validation.

**Smoke.** Use the challenge schedule already documented in Phase 13 only if
the paper needs another dynamic-workflow figure. Try `n=2`, `K={64,80,96}` and
require stale-query separation before scaling.

**Promotion gate.** Promote a second multi-turn figure only if it is cleaner
than the existing one or supports a different point. Otherwise do not add
figure clutter.

### P6: Can runtime evidence support the idle-window claim?

**Risk.** Current latency plots show repair capacity and exact/proxy costs, but
the idle-window axis is a representative envelope, not an empirical tool-call
trace distribution.

**Smoke.** No GPU needed. Build a trace-backed or literature-backed idle-window
summary only if a defensible source/dataset is available. Otherwise keep the
claim as "capacity under representative pause windows" and avoid claiming
observed real-agent fit.

**Promotion gate.** Main paper only if the figure shows either a real
distribution or a clearly labeled capacity envelope with no implied empirical
trace.

### P7: Are selector variants worth an algorithmic contribution?

**Risk.** Coverage-aware/MMR variants could be algorithmically interesting, but
weak ablations would clutter the paper.

**Smoke.** 4Q exact scoring, `n=1`, `K={24,48,96}`, conditions
`A/B/B_match/IdleKV/IdleKV-Coverage/IdleKV-MMR/Oracle-K`.

**Promotion gate.** Promote only if a variant improves current IdleKV by at
least `0.05` at mid-K without hurting high-K by more than `0.02`, and if the
Gold-K gap narrows for a clear reason.

**Current implementation gate.** The Phase 14 summarizer/evaluator must export
`IdleKV-Coverage` and `IdleKV-MMR` columns before this smoke can be trusted.
Without those columns, the run is invalid even if the GPU artifact exists.

**Locked run if smoke passes.** Repeat the same setting with `n=24` using
`run_selector_variant_locked.sh`. A selector result enters the main paper only
if it changes the algorithmic claim; otherwise it becomes appendix evidence
that simple diversity variants did or did not close the Gold-K gap.

### P8: Is quantized KV repair viable now?

**Risk.** The first quantization branch was negative: low-bit storage destroyed
accuracy before selective promotion could help, while 8-bit already preserved
accuracy.

**Next action.** Do not run another full GPU sweep until the baseline is
redesigned around page/channel-aware quantization and byte accounting has unit
tests. Keep the current result as future-work evidence, not a main claim.

### P9: Does the query-count frontier need more than 2Q/4Q/6Q/8Q?

**Risk.** Extra query counts could make the frontier look broader but may add
little information.

**Current audit.** 2Q/4Q/6Q/8Q already covers easy-to-stress progression. More
query counts should not run unless they answer a specific graph question.

**Promotion gate.** Do not run 3Q/5Q/7Q unless the main figure needs a
monotonic trend claim with finer resolution. Current paper should avoid that
claim.

### P10: Does the two-tier KV framing need a scaling experiment?

**Risk.** The discussion gestures at GPU active KV plus slower buffered KV. A
systems reviewer may ask how this scales when sessions reach millions of rows.

**Current audit.** Runtime capacity artifacts already cover up to multi-million
row synthetic selection/movement regimes. Do not rerun unless the paper needs a
specific missing row count or byte budget.

**Promotion gate.** Main text should include only the concise scaling insight:
moving selected rows is cheap relative to exact scoring; large offloaded stores
need their own lower-tier retention policy. Detailed capacity curves belong in
appendix.

### P11: Does the paper need another main result figure?

**Risk.** More figures can make the paper look more empirical but can also
weaken the argument if they are redundant with Figure 2 or rely on smoke data.

**Test.** For every candidate figure, write the one-sentence reviewer question
it answers and identify which existing figure or paragraph it replaces.

**Promotion gate.** A new main figure must satisfy at least one condition:

- it closes a critical flaw that Figure 2/3/4/runtime cannot close;
- it gives non-saturated cross-model evidence;
- it shows a scalable scorer preserving the main effect under controls;
- it introduces a faithful named baseline that changes the method comparison.

Otherwise it goes to the appendix or Phase 15+ notes.

### P12: Is a realistic workflow diagnostic ready?

**Risk.** Agent/tool framing is motivating, but a weak realistic task would be
less convincing than the controlled relevance-shift diagnostic.

**Smoke.** CPU/design first: define a repo/tool-style retrieval task with a
known answer, prove full context succeeds, and prove matched no-repair fails.
Only then run a GPU smoke.

**Promotion gate.** Main paper only if it demonstrates a qualitatively new
dynamic-workflow behavior rather than another MQ-NIAH-shaped retrieval shift.

## Execution Rules

- Smoke before any locked run.
- Unit-test any new code before GPU work.
- Long locked runs go in tmux with timestamped logs.
- Do not promote smoke data to the main paper.
- Do not run a GPU job unless it is either a falsifying smoke or a locked run
  for a pre-specified figure/table.
- While a GPU job is running, use CPU time for code tests, evaluator scripts,
  paper economy audits, and next-experiment design.
- Every figure must have a distinct reviewer question. If it does not, move it
  to the appendix or remove it.
- After paper edits, rebuild `paper/main.pdf` and check the LaTeX log.

## Immediate Execution Order

1. Run the Phase 14 readiness audit on current artifacts.
2. Run targeted unit tests for the new audit code and script syntax checks.
3. Launch the controlled proxy smoke if the audit still marks P0 as open.
4. Audit the smoke. Only then decide whether to run the locked proxy frontier.
5. Run the Refresh frontier smoke if P0 does not consume the GPU queue or if the
   proxy smoke fails quickly.
6. Run the calibrated Llama smoke if P0/P1 leave the main claim too
   single-model.
7. Open exact Scissorhands only after P0-P2 are settled and only with CPU
   invariant tests first.
8. Run selector-variant smoke only after its summary columns and evaluator pass
   unit tests.
9. Only after P0-P2/P7 decisions, edit the paper so main text contains the
   strongest compact set of promoted results and appendix contains useful but
   non-load-bearing diagnostics.
