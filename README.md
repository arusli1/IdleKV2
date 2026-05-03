# IdleKV

IdleKV is a research prototype for **test-time adaptation of long-context KV
cache state**. In paused, multi-turn agent workflows, a new user turn, tool
result, test failure, or file change can shift which earlier context matters.
IdleKV studies whether the runtime can use the pause before decoding resumes to
adapt the active GPU KV cache to that new relevance signal.

The prototype keeps evicted context KV rows in a host-memory warm tier, scores
them after the next-turn signal is known, and promotes a restore budget `K` back
into the active cache. The main comparisons use a matched resumed active-cache
budget: IdleKV is compared against no-repair and content-agnostic restore
controls with the same number of active context KV rows.

## Research Question

Long-context KV compression is usually a one-shot decision made before the next
turn is known. IdleKV asks a narrower question:

> If a future turn reveals new evidence about which past tokens matter, can a
> runtime repair the active KV state during idle time without increasing the
> resumed active-cache budget?

This is a mechanism study, not a production serving stack. The paper treats KV
repair as an inference-time adaptation operator over active state: model weights
stay fixed, while the runtime revises which historical context the next decode
can directly attend to.

## Scope and Prerequisites

- The matched budget is the resumed active context KV budget. IdleKV also keeps
  an offloaded warm store, and the paper reports those storage, scoring, and
  transfer costs separately.
- The runtime figure is a single-node capacity envelope for score/select/promote
  mechanics, not a trace-backed distribution of real tool-call wait times.
- Paper-grade GPU runs assume local model weights under `models/`, the vendored
  `ruler/` checkout, CUDA/PyTorch, and enough GPU memory for the selected model
  and context length.
- CPU tests and figure-rendering checks can be run without launching new GPU
  experiments.

## Evidence in This Repository

- Controlled benchmark: split-query multi-query needle-in-a-haystack
  (MQ-NIAH), which provides explicit cross-turn relevance shifts and annotated
  future-relevant spans.
- Primary model: Qwen2.5-7B-Instruct at 32K context.
- Main experiments: MQ-NIAH-2Q/4Q/6Q/8Q restore-budget sweeps, next-turn
  specificity controls, a five-turn relevance-shift diagnostic, retention-rule
  sensitivity, and runtime-capacity probes.
- Breadth checks: a same-protocol Llama-3.1-8B 4Q portability probe and
  protocol-matched retention-rule variants.
- Open gap: the current paper is still controlled/synthetic. The repo includes
  a CPU-tested RepoDelta generator for future real-repository relevance-shift
  experiments, but it is not paper evidence until GPU ability and repair smokes
  pass the written gates.

## Paper

- Source: `paper/main.tex`
- Rebuilt PDF: `paper/main.pdf`
- Figure renderer: `paper/scripts/render_paper_figures.py`
- Writing and venue guide: `paper_guide.md`

Rebuild the paper from the repo root:

```bash
.venv/bin/python paper/scripts/render_paper_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

LaTeX intermediates are written to `paper/aux/` by `paper/.latexmkrc`.
Undefined references, undefined citations, overfull boxes, or figure overlap
should be fixed before a paper snapshot.

## Reproducing Checks

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

GPU experiments should start with the smallest smoke test that can falsify the
design, then move to a locked run only after the smoke passes a written gate.

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

## Experiment Vocabulary

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

## Active Questions

- Does the controlled proxy scorer preserve the repair effect under Random-K,
  Oldest-K, and Gold-K controls?
- Can the RepoDelta real-repository diagnostic become a credible non-synthetic
  relevance-shift result after full-context GPU smokes?
- Which selector or retention-policy variants add enough signal to replace an
  existing figure rather than simply append another one?
- What trace-scheduled repair experiment best connects the runtime-capacity
  envelope to real tool/environment wait distributions?

## Git Hygiene

Generated phase outputs, local model weights, LaTeX intermediates, and rendered
plot binaries are ignored unless deliberately promoted. Keep paper-ready source
changes in tracked code, `.tex`, `.md`, and compact CSV artifacts that are
needed to regenerate figures.
