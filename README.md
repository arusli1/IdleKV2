# IdleKV

IdleKV is a research prototype for **test-time adaptation of long-context KV
cache state**. In multi-turn agent workflows, a new user turn, tool result, test
failure, or file change can shift which earlier context matters. IdleKV studies
whether a runtime can use the pause before decoding resumes to adapt the active
GPU KV cache to that new relevance signal.

The prototype keeps evicted KV rows in an off-device store, scores them after
the next-turn signal is known, and promotes a restore budget `K` back into the
active cache. All main comparisons use a matched resumed active-cache budget:
IdleKV is compared against no-repair and content-agnostic restore controls with
the same number of active context KV rows.

## Current Evidence

- Main controlled task family: split-query multi-query needle-in-a-haystack
  (MQ-NIAH), which gives explicit cross-turn relevance shifts and annotated
  future-relevant spans.
- Main model: Qwen2.5-7B-Instruct at 32K context.
- Main result set: MQ-NIAH-2Q/4Q/6Q/8Q restore-budget sweeps, specificity
  controls, a five-turn relevance-shift diagnostic, retention-rule sensitivity,
  and runtime-capacity probes.
- Current limitation: this is not yet an end-to-end agent benchmark. Llama and
  proxy-scorer checks are useful diagnostics, but broader model-family and
  deployment-latency claims require additional non-saturated and trace-backed
  evidence.

## Paper

- Source: `paper/main.tex`
- Rebuilt PDF: `paper/main.pdf`
- Figure renderer: `paper/scripts/render_paper_figures.py`
- Writing and venue guide: `paper_guide.md`

Rebuild from the repo root:

```bash
.venv/bin/python paper/scripts/render_paper_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

LaTeX intermediates are written to `paper/aux/` by `paper/.latexmkrc`. Underfull
box warnings from float placement are acceptable; undefined references,
undefined citations, overfull boxes, or figure overlap should be fixed before a
paper snapshot.

## Tests

Run focused paper and closure tests:

```bash
.venv/bin/python -m pytest \
  phases/phase14_critical_flaw_closure/tests/test_audit_phase14_readiness.py \
  phases/phase13_iteration_framework/tests/test_paper_language.py \
  phases/phase13_iteration_framework/tests/test_framework.py \
  phases/phase10_expansion/tests/test_multiturn.py \
  phases/phase10_expansion/tests/test_multiturn_runner.py -q
```

Run the broader CPU-side suite:

```bash
.venv/bin/python -m pytest -q
```

GPU experiments should always start with the smallest smoke test that can
falsify the design, then move to a locked run only after the smoke passes a
written gate.

## Repository Map

- `paper/`: ICML-style paper draft, figure assets, and rendering scripts.
- `paper_guide.md`: terminology, venue constraints, figure rules, and editing
  guardrails.
- `phases/phase6_repair/`: core matched-budget repair protocol, selectors,
  reporting, and unit tests.
- `phases/phase9_experiment_deepening/` through
  `phases/phase14_critical_flaw_closure/`: experiment expansions, smoke
  evaluators, locked-run wrappers, and paper-readiness audits.
- `docs/`: project status and result-retention notes.
- `saved_results/`: retained summaries from earlier runs.
- `models/`: local model weights; ignored by git.
- `ruler/`: vendored RULER checkout; treated as external benchmark code.

## Experiment Conventions

- `Matched`: no repair under the same resumed active-cache budget.
- `IdleKV`: restore conditioned on the current next-turn signal from the
  offloaded evicted-KV store.
- `Gold-K`: benchmark-metadata hindsight reference, not an implementable
  algorithm.
- `Random-K` and `Oldest-K`: content-agnostic restore controls.
- `Refresh-buffered`: reselects the full resumed active budget from active plus
  offloaded rows using the next-turn signal. It is a method-boundary reference,
  not a deployable full-prefix recompute baseline.
- `Proxy` scoring: cheaper scorer based on appended next-turn state. Treat it
  as scalable-scorer evidence only when controlled Random-K, Oldest-K, and
  Gold-K gates pass.

## Current Work Queue

The active closure plan is in
`phases/phase14_critical_flaw_closure/phase14_plan.md`. The highest-priority
open questions are:

- whether the controlled proxy scorer preserves the repair effect under
  Random-K, Oldest-K, and Gold-K controls;
- whether any non-saturated cross-model run is strong enough for a main-paper
  model-family statement;
- whether a real agentic or trace-scheduled benchmark can replace some
  synthetic evidence;
- whether additional selector or retention-policy variants add enough signal to
  replace, rather than merely append, existing figures.

## Git Hygiene

Generated phase outputs, local model weights, LaTeX intermediates, and rendered
plot binaries are ignored unless deliberately promoted. Keep paper-ready source
changes in tracked code, `.tex`, `.md`, and compact CSV artifacts that are
needed to regenerate figures.
