# Phase 14 Status

Last updated: 2026-05-03 21:08 UTC.

## Implemented

- Added the Phase 14 plan with an explicit idea inventory, smoke design,
  promotion gates, and paper actions.
- Added a readiness audit:
  `phases/phase14_critical_flaw_closure/scripts/audit_phase14_readiness.py`.
- Added smoke wrappers for branches that can be tested with current code:
  proxy scorer controls, Refresh-K frontier, calibrated Llama, and selector
  variants.
- Added unit tests for the readiness audit.
- Expanded the Phase 14 plan into a uniform test loop: reviewer question,
  paper object, code readiness, minimal smoke, automated gate, locked run, and
  paper audit.
- Added automated Phase 14 smoke evaluators for Refresh-K frontiers, calibrated
  Llama smokes, and selector variants.
- Updated the readiness audit so that, once the locked controlled-proxy CSV
  exists, P0 uses that CSV for controlled quality and Random/Oldest/Gold checks
  while still using the existing fixed-K proxy reference for latency speedup.
- Tightened the P0 status logic so a completed controlled proxy artifact that
  fails retained-gain/quality/speed gates is reported as `needs_proxy_redesign`,
  not as another request for a controlled run.
- Added a log monitor:
  `phases/phase14_critical_flaw_closure/scripts/monitor_proxy_progress.py`,
  which reports live controlled-proxy progress and partial per-$K$ means from
  the tmux log without touching the GPU run.
- Extended the log monitor with a rough ETA based on completed example-split
  rows and the configured 4Q/6Q split counts.
- Added `postprocess_proxy_controlled_locked.sh` to make the finish sequence
  repeatable: evaluate the locked CSV, run the readiness audit, rerender paper
  figures, rebuild `paper/main.pdf`, and scan for blocking LaTeX warnings.
- Added pre-specified locked-run wrappers for calibrated Llama and selector
  variants, used only after their smoke gates pass.
- Patched the shared Phase 9 artifact summarizer to export
  `IdleKV-Coverage` and `IdleKV-MMR`, because selector-variant smokes were not
  auditable without those columns.
- Incorporated completed expert-agent critiques: keep main text compact,
  foreground tiered-KV promotion, treat Refresh-K as a boundary comparator, and
  require non-saturated cross-model evidence before any main-paper model
  generality claim.
- Tightened the paper latency paragraph and conclusion around the distinction
  between exact research scoring, proxy scoring, and scalable tiered-KV
  score/select/promote mechanics.
- Added a peer-reviewed/archival agent-efficiency latency citation to the
  introduction and latency discussion: published web-agent measurements report
  API/environment wait components, with environment interaction up to `53.7%`
  of total latency. The paper now frames Figure 6 as capacity evidence rather
  than an empirical tool-call duration distribution.
- Added a `paper_guide.md` rule for future runtime/idle-window edits: cite
  agent-efficiency measurements as web-agent API/environment-wait evidence,
  not as a universal tool-call distribution.
- Ran a focused ramble/hole pass against `paper/outline.md` and
  `paper_guide.md`: removed repeated workflow examples, shortened method-scope
  exclusions, condensed main split descriptions, trimmed redundant runtime
  caveats, and made the trace-scheduled systems-evaluation gap explicit.
- Added a dormant, tested paper-figure hook for the controlled proxy locked
  run. Once `proxy_controlled_locked_n100.csv` exists, `paper/scripts/
  render_paper_figures.py` can render `proxy_controlled_frontier.pdf`, and the
  appendix has an `\IfFileExists` block ready to include it.
- Strengthened the idle-window evidence with a second agent-systems citation:
  AgentCgroup reports OS-level execution including tool calls and
  container/agent initialization as `56-74%` of coding-agent task latency.
- Tightened an overlong appendix portability/runtime caption without changing
  the plotted evidence.
- Rebuilt `paper/main.pdf` with `latexmk -pdf -interaction=nonstopmode
  -halt-on-error main.tex`; the PDF rebuilt successfully.
- Updated `README.md` so the active closure phase points to Phase 14.
- Added `paper/.latexmkrc` and removed regenerated LaTeX intermediates from
  `paper/`; future rebuilds keep aux/log files in `paper/aux/`.
- Ran a second outline-guided paper economy pass: removed the paper-facing
  "locked" run label, shortened the method-scope exclusion list, compressed
  multi-turn result narration, demoted saturated Llama numbers out of main
  prose, and split the limitations paragraph so the trace-scheduled systems
  gap and tiered-KV scaling implication are separate.
- Added a concise setup sentence stating why split-query MQ-NIAH is used:
  control over annotated future-relevant spans, explicit second-turn relevance
  shift, and shared turn-`N` history across conditions.
- Split dense runtime/discussion prose into technical run-in labels for the
  capacity envelope, tiered-cache scaling implication, and future benchmark
  axes.
- Added the Figure 6 runtime decomposition
  `T_repair ~= T_scan + T_topK + T_promote`, with explicit wording that the
  candidate scan is linear in offloaded candidate rows for fixed query/model
  shape, while promotion is independent of candidate-store size but depends on
  restored rows and the reinsertion path.
- Fixed the readiness audit to match the pre-registered proxy retained-gain
  gates: `0.85` for 4Q and `0.80` for 6Q. The audit output now records which
  retention gate was applied, and the behavior has a regression test.
- Accepted the latest AdaptFM/KV-runtime reviewer audits: proxy wording now
  says preliminary fixed-`K` until the controlled run finishes, runtime
  evidence is framed as a capacity envelope consistent with multi-second
  tool/environment components rather than a measured idle-window trace,
  retention-rule evidence is scoped to protocol-matched heuristics, Llama is
  not used for broad model-family claims, and the overlap diagnostic is a short
  mechanism check.
- Added the outside-facing README framing and pushed it.
- Reframed the paper around test-time adaptation of active KV state across
  relevance shifts, with a direct AdaptFM scope sentence in the introduction.
- Replaced the stale saturated Llama 6Q appendix source with the Phase 11
  Llama-3.1-8B 4Q full K-grid (`n=24`) and updated the main text to treat it as
  a cautious same-protocol portability check rather than a model-family claim.
- Updated the figure renderer and readiness audit so non-saturated cross-model
  full grids are preferred over older saturated short-grid artifacts.

## Validation

- `bash -n phases/phase14_critical_flaw_closure/scripts/*.sh`
- `bash -n phases/phase14_critical_flaw_closure/scripts/postprocess_proxy_controlled_locked.sh`
- `.venv/bin/python -m py_compile phases/phase14_critical_flaw_closure/scripts/*.py phases/phase9_experiment_deepening/scripts/phase9_artifact_summary.py`
- `.venv/bin/python -m pytest phases/phase14_critical_flaw_closure/tests/test_audit_phase14_readiness.py -q`
  - `13 passed`
- `.venv/bin/python -m py_compile phases/phase14_critical_flaw_closure/scripts/monitor_proxy_progress.py`
- `.venv/bin/python -m pytest phases/phase14_critical_flaw_closure/tests/test_audit_phase14_readiness.py -q`
  - `13 passed` after adding monitor ETA coverage and controlled-proxy
    readiness coverage
- `.venv/bin/python -m pytest -q`
  - `234 passed, 16 warnings, 304 subtests passed`
- `.venv/bin/python -m pytest phases/phase6_repair/tests/test_runner.py phases/phase6_repair/tests/test_reporting.py -q`
  - `37 passed`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from
  `paper/`
  - rebuilt `paper/main.pdf` successfully after the citation/framing edits
  - log scan found no undefined citations or overfull boxes; remaining warnings
    are underfull vboxes from float/page layout
- Rebuilt `paper/main.pdf` again after the final concision trim; log scan
  again found no undefined citations or overfull boxes.
- `.venv/bin/python -m py_compile paper/scripts/render_paper_figures.py phases/phase9_experiment_deepening/tests/test_paper_figure_renderer.py`
- `.venv/bin/python -m pytest phases/phase9_experiment_deepening/tests/test_paper_figure_renderer.py -q`
  - `10 passed`
- Rebuilt `paper/main.pdf` after adding the dormant proxy-controlled appendix
  hook; the optional figure is absent until the locked CSV exists, so the PDF
  remains 12 pages.
- Rebuilt `paper/main.pdf` after adding the AgentCgroup citation; log scan
  found no undefined citations or overfull boxes.
- Rebuilt `paper/main.pdf` after the appendix caption trim; log scan found no
  undefined citations or overfull boxes.
- `.venv/bin/python -m pytest phases/phase9_experiment_deepening/tests/test_paper_figure_renderer.py phases/phase14_critical_flaw_closure/tests/test_audit_phase14_readiness.py phases/phase13_iteration_framework/tests/test_paper_language.py -q`
  - `27 passed`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from
  `paper/`
  - rebuilt `paper/main.pdf` successfully after the test-time adaptation and
    Llama portability edits
  - log scan found no undefined citations, undefined references, or overfull
    boxes
- Rebuilt `paper/main.pdf` after the latest outline-guided ramble/hole pass;
  log scan again found no undefined citations or overfull boxes.
- Incorporated two reviewer-agent critiques into `paper/main.tex`:
  repair scoring and burst packing are now specified in Method; matched active
  budget accounting now states which off-device bytes and latency costs are
  unmatched but reported separately; 2Q is defined in Setup; StaleQ/WrongQ and
  Refresh-buffered are defined before Results; the multi-turn diagnostic now
  defines its recurrent recompress/repair protocol; and the appendix now states
  greedy decoding, seed policy, bootstrap unit, and the core runtime stack.
- Rebuilt `paper/main.pdf` after each paper edit. References start on page 7,
  so the main body remains within the six-page limit; total PDF length is 13
  pages with appendix.
- Rebuilt `paper/main.pdf` after the second outline-guided paper economy pass;
  log scan again found no undefined citations or overfull boxes, and the PDF
  text scan found no visible internal run vocabulary beyond bibliography
  titles.
- Rebuilt `paper/main.pdf` after adding runtime/discussion labels; log scan
  again found no undefined citations or overfull boxes, and paper-language
  tests passed.
- Rebuilt `paper/main.pdf` after the Figure 6 complexity and reviewer-economy
  edits; log scan again found no undefined citations or overfull boxes, and
  paper-language tests passed.
- `.venv/bin/python -m pytest -q`
  - `234 passed, 16 warnings, 304 subtests passed` at 2026-05-03 20:49 UTC.
- `.venv/bin/python -m pytest phases/phase14_critical_flaw_closure/tests/test_audit_phase14_readiness.py -q`
  - `14 passed` after aligning the 6Q proxy retained-gain gate with the
    written Phase 14 plan.

## Paper Economy Audit

Most main-text detail is now earning its place because it answers a named
reviewer question: matched-budget effect, cue specificity, repeated relevance
shift, retention-rule sensitivity, robustness/mechanism, or latency capacity.
The remaining over-detail risks are:

- Repeated result narration: secondary diagnostics should stay in prose only
  when they block a likely misread. The multi-turn paragraph now keeps the
  stale-query audit because reviewers could otherwise attribute the result to
  reused old queries, but it avoids walking through every condition.
- Runtime prose: the latency paragraph should not read like a systems trace
  result. The paper now calls Figure 6 a capacity envelope rather than a
  measured tool-call trace and names the missing scheduler evidence.
- Method exclusions: implementation limits should be concise in Method and
  expanded only in Discussion. The latest edit shortened the distributed
  systems exclusion list without hiding the assumption.
- Saturated portability checks: the Llama result is now appendix-only and is
  explicitly not used for model-family claims.

The reviewer holes that remain material are:

- Controlled proxy scoring: the locked run must decide whether the scalable
  proxy path preserves quality under Random-K/Oldest-K/Gold-K controls.
- Trace-scheduled evaluation: the paper cites web/coding-agent wait evidence,
  but it still lacks a real wait-distribution scheduler experiment.
- Broader generality: current Llama and retention-rule evidence are appendix
  breadth checks, not enough for a broad model-family or named-policy claim.
- Algorithmic headroom: Refresh-buffered and Gold-K show that IdleKV is a
  useful promotion primitive, not the final selector.
- Off-device retention policy: the paper now states the two-level retention
  problem explicitly, but it does not solve how a long-running agent should
  choose which evicted rows remain searchable, compressed, summarized, or
  recomputable in warm/cold tiers.

Reviewer-agent triage:

- We accepted and fixed underdefined repair mechanics, active-budget
  accounting, 2Q setup, Refresh-buffered definition, multi-turn recurrence, and
  reproducibility details.
- We kept the retention-rule sensitivity figure in the main paper for now
  because it answers a distinct AdaptFM robustness question, but the text now
  explicitly says it is sensitivity to protocol-matched retention rules, not
  named H2O/StreamingLLM performance.
- We did not promote a new Refresh-buffered frontier because the existing
  Phase 14 smoke confirmed it is a method-boundary result rather than a clean
  new main figure.

## Initial Readiness Audit

Command:

```bash
.venv/bin/python phases/phase14_critical_flaw_closure/scripts/audit_phase14_readiness.py
```

Result:

- `4q_proxy`: needs controlled proxy smoke or locked run. Existing n=100 proxy
  has strong retained gain and speedup, but lacks Random-K, Oldest-K, and Gold-K
  controls and only has two K points.
- `6q_proxy`: needs controlled proxy smoke or locked run. Existing n=100 proxy
  has large speedup but narrowly misses the strict retained-gain gate and also
  lacks controls.
- `specificity`: boundary result. IdleKV beats stale/donor controls, but
  Refresh-K exceeds IdleKV by `+0.458`, so the paper must frame Refresh-K
  explicitly as full-budget reselection/headroom.
- `llama`: appendix-only portability. The current Llama run has `n=12`, only
  three K values, and saturates at `1.0`.
- `policy breadth`: protocol-matched breadth only, not a faithful named
  prior-policy reproduction. If needed, exact Scissorhands is the next branch.

## Completed Runs

### `phase14_proxy_smoke`

Completed: 2026-05-03 19:42 UTC.

Output:

- `phases/phase14_critical_flaw_closure/results/proxy_controlled_smoke_n4.csv`

Evaluator:

```bash
.venv/bin/python phases/phase14_critical_flaw_closure/scripts/evaluate_proxy_controlled_smoke.py \
  --summary-csv phases/phase14_critical_flaw_closure/results/proxy_controlled_smoke_n4.csv
```

Result:

- 4Q controlled proxy smoke passed. At K=96, proxy IdleKV gain is `+0.833`;
  max Random-K/Oldest-K control lift is `+0.042`.
- 6Q controlled proxy smoke passed. At K=96, proxy IdleKV gain is `+0.500`;
  max Random-K/Oldest-K control lift is `+0.042`.

Decision:

- The controlled smoke supports running a locked proxy frontier.
- Before launching the longer locked proxy run, test the Refresh-K boundary
  smoke because it is short and changes paper framing.

## Active Run

`phase14_proxy_locked` is running in tmux.

Command:

```bash
tmux new-session -d -s phase14_proxy_locked \
  'cd /home/ubuntu/IdleKV && bash phases/phase14_critical_flaw_closure/scripts/run_proxy_controlled_locked.sh'
```

Purpose:

- Produce controlled, locked proxy-scorer evidence for the scalable repair
  path.
- Settings: 4Q and 6Q, `K={48,64,80,96,128}`, `n=100`, proxy scorer,
  `A/B/B_match/Random-K/Oldest-K/IdleKV/Oracle-K`.
- Progress at 2026-05-03 19:57 UTC: 4Q is active, around example `19/100`
  across three splits; 6Q has not started.
- Progress at 2026-05-03 20:33 UTC: 4Q is active, around example `56/100`
  across three splits; 6Q has not started.
- Progress at 2026-05-03 20:44 UTC: 4Q has completed; 6Q is active, around
  example `10/100` across four splits. Early 6Q partial means still separate
  proxy IdleKV from matched, Random-K, and Oldest-K, but the final CSV and
  evaluator remain the decision point.
- Progress at 2026-05-03 20:50 UTC: 6Q is active, around example `15/100`
  across four splits; the rough monitor ETA is about 58 minutes.

Promotion decision after completion:

- If it passes the controlled proxy evaluator, use it as the primary
  scalable-scorer quality/latency bridge for this benchmark.
- If it fails because content-agnostic controls close the gap, demote proxy
  scoring and keep exact scoring as mechanistic evidence only.

### `phase14_refresh_smoke`

Completed: 2026-05-03 19:46 UTC.

Output:

- `phases/phase14_critical_flaw_closure/results/refresh_frontier_smoke_n2.csv`

Evaluator:

```bash
.venv/bin/python phases/phase14_critical_flaw_closure/scripts/evaluate_phase14_smokes.py \
  --kind refresh \
  --summary-csv phases/phase14_critical_flaw_closure/results/refresh_frontier_smoke_n2.csv
```

Result:

- Status: `refresh_boundary_confirmed`.
- IdleKV improves over matched at every tested K, but Refresh-K dominates
  IdleKV across the frontier.
- At high K, stale/donor controls get close or equal to IdleKV, so this smoke
  should not become a new main figure. It supports text framing: IdleKV is an
  incremental pre-resume promotion primitive, while Refresh-K is full-budget
  Q2-time reselection headroom.
