# Phase 0: Baseline RULER

This directory contains the repo-local entrypoints, outputs, and notes for the
initial `Qwen/Qwen2.5-7B-Instruct` RULER baseline run.

This baseline is intentionally narrow and strict. The goal is not to run every
task in upstream RULER first. The goal is to establish a trustworthy long-
context baseline on a focused subset before later experiments change model
behavior, prompting, or serving settings.

## Purpose

This run answers one question first:

Can the local Qwen2.5 setup reproduce sane long-context behavior on a focused
RULER subset, with scores that degrade gradually rather than collapsing because
of a harness bug?

That is why this folder exists separately from the vendored benchmark code. It
keeps the runner, logs, artifacts, and results for this specific initial test
in one place.

## What Is Being Run

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Serving stack: local `vLLM`
- Context lengths: `4096`, `8192`, `16384`, `32768`
- Samples: `500` per task per length
- Tasks: `niah_single_1`, `niah_single_2`, `niah_single_3`, `vt_2hop`, `fwe`

This is a 5-task subset of RULER, not the full default upstream benchmark.

## Why These Tasks

These tasks are enough to expose the most important failure modes early.

| Task | What it measures | Local configuration | Why it matters |
| --- | --- | --- | --- |
| `niah_single_1` | Exact retrieval from a distractor-filled context | Noise haystack, word key, numeric value | Fast sanity check for straightforward retrieval |
| `niah_single_2` | Exact retrieval from a more natural haystack | Essay haystack, word key, numeric value | More realistic distractors than repeated noise |
| `niah_single_3` | Exact retrieval with harder string matching | Essay haystack, word key, UUID value | Tests brittle exact copying rather than simple numbers |
| `vt_2hop` | Multi-step tracing | One variable chain, two hops | Detects whether the model can follow state rather than only recall a single fact |
| `fwe` | Aggregation over long context | Top-3 frequent coded words, `alpha=2.0` | Harder than retrieval; shows whether performance drops on non-retrieval work |

The three `niah_single_*` tasks are grouped into one aggregate in the runner:
`S-NIAH`.

## How The Pipeline Works

The main entrypoint is `scripts/run_baseline.py`.

For each run, it does the following:

1. Verifies the local model exists in `models/Qwen2.5-7B-Instruct`.
2. Verifies the tokenizer and exact Qwen2.5 chat template.
3. Confirms the model config exposes a `32768` token context window.
4. Ensures the synthetic RULER assets are present.
5. Starts a local `vLLM` server.
6. For each context length:
   - Generates synthetic datasets with `scripts/data/prepare.py`
   - Runs inference with `scripts/pred/call_api.py`
   - Scores predictions with `scripts/eval/evaluate.py`
   - Writes a checkpointed results snapshot
7. Writes summary artifacts:
   - `results/baseline_ruler.json`
   - `results/degradation_curve.csv`
   - `results/degradation_curve.svg`

The runner is resumable. If a dataset or prediction file already has the
expected number of rows, the runner skips that step unless `--force` is used.

## Important Prompting Detail

Structured RULER tasks rely on an `answer_prefix` that is stored in each
generated example.

For this baseline, the prediction path must send:

- `prompt = input + answer_prefix`

and not only `input`.

This matters most for:

- `vt_2hop`
- `fwe`

If those tasks collapse at short context length while NIAH stays high, the
first thing to check is whether the `answer_prefix` made it into the inference
prompt.

## Files In This Folder

- `scripts/run_baseline.py`: repo-local runner for this benchmark pass
- `requirements.txt`: Python packages used by the runner and local RULER flow

The vendored upstream benchmark code stays in `ruler/`.

## Runtime Outputs

- `artifacts/`: generated datasets and raw prediction files
- `results/`: summarized benchmark output
- `results/degradation_curve.csv`: tabular degradation data for the tracked tasks
- `results/degradation_curve.svg`: lightweight plot for `S-NIAH`, `vt_2hop`, and `fwe`
- `logs/`: `vLLM` startup and serving logs

## Output Layout

For each context length, outputs are written under:

- `artifacts/benchmark_root/qwen2.5-7b-instruct/synthetic/<length>/data/`
- `artifacts/benchmark_root/qwen2.5-7b-instruct/synthetic/<length>/pred/`

The main files to inspect are:

- `data/<task>/validation.jsonl`
  - Generated benchmark inputs and reference outputs
- `pred/<task>.jsonl`
  - Raw model outputs, one line per example
- `pred/summary.csv`
  - Per-task scores and null counts for that context length
- `pred/submission.csv`
  - Flat summary export from upstream RULER evaluation

At the runner level:

- `results/baseline_ruler.json`
  - Tokenizer checks
  - Server configuration
  - Per-length scores
  - Paths to generated artifacts
  - Acceptance report
- `results/degradation_curve.csv`
  - Machine-readable curve data for quick plotting
- `results/degradation_curve.svg`
  - Repo-local visualization of degradation shape

## How To Read The Results

### 1. Tokenizer And Context Checks

`baseline_ruler.json` includes:

- `chat_template_matches_expected`
- `rendered_prompt_tokens`
- `tokenizer_model_max_length`
- `config_max_position_embeddings`
- `context_32k_accessible`

These are harness checks, not benchmark scores. If these fail, benchmark scores
cannot be trusted.

### 2. Per-Task Scores

Each `summary.csv` contains:

- `Tasks`
- `Score`
- `Nulls`

`Score` is the percentage score returned by the RULER evaluator for that task.

`Nulls` counts missing or empty outputs. Ideally this stays at `0/500`.

### 3. S-NIAH Aggregate

The runner computes:

- `S-NIAH = mean(niah_single_1, niah_single_2, niah_single_3)`

This is the most important retrieval sanity curve in the initial baseline.

### 4. Degradation Curve

The curve artifacts track:

- `S-NIAH`
- `vt_2hop`
- `fwe`

This gives a quick view of whether quality degrades smoothly with longer
contexts or fails abruptly.

## What Good Results Should Look Like

### Expected short-context behavior

At `4K`:

- `S-NIAH` should be near perfect
- `vt_2hop` should also be near perfect if prompting is correct
- `fwe` is harder, so it is expected to be lower than NIAH and VT

The runner currently encodes two explicit acceptance checks:

- `S-NIAH @ 4K >= 98`
- `S-NIAH @ 32K >= 85`

### Expected medium-context behavior

At `8K` and `16K`:

- `S-NIAH` should remain very high
- `vt_2hop` should stay strong, though small degradation is plausible
- `fwe` may fluctuate more, but it should not collapse

### Expected long-context behavior

At `32K`:

- retrieval should degrade somewhat relative to shorter lengths
- the degradation should be gradual, not catastrophic
- `S-NIAH` should still remain comfortably above a weak baseline

### Null counts

Across all tasks and lengths:

- `Nulls` should ideally stay `0/500`

High null counts usually indicate serving failures, decoding failures, or
context-window problems rather than real model behavior.

## What Suspicious Results Look Like

These patterns usually mean something is wrong with the harness or prompt path:

- `vt_2hop` near `0` at `4K` while NIAH is near `100`
- a sudden cliff between neighboring lengths with no gradual decline
- non-zero null counts at short contexts
- `context_32k_accessible = false` after a completed `32K` run
- missing `32768` in `baseline_ruler.json` after the run is supposed to be done

For this benchmark specifically, a short-context collapse on `vt_2hop` or `fwe`
is more likely to indicate prompt formatting trouble than a real model limit.

## What Not To Over-Interpret

Do not compare this run too literally against the single aggregate numbers in
the upstream RULER README.

Reasons:

- this harness runs a 5-task subset, not the full default benchmark
- this setup uses `Qwen2.5-7B-Instruct`, not `Qwen2.5-7B-Instruct-1M`
- the upstream table reports a broader aggregate than the per-task subset here

This run is still useful for:

- detecting harness bugs
- checking short-context sanity
- measuring degradation shape from `4K -> 32K`
- creating a local baseline file for future deltas

## Monitoring A Live Run

While the benchmark is running, the easiest live indicators are:

- the growing line counts in `pred/<task>.jsonl`
- `pred/summary.csv` once a full context length finishes
- `results/baseline_ruler.json`, which is refreshed after each completed length
- `logs/vllm_qwen25.log` for server-side issues

## Suggested Rule For Declaring The Baseline "Done"

Treat the baseline as complete only when all of the following are true:

- `baseline_ruler.json` includes `4096`, `8192`, `16384`, and `32768`
- `context_32k_accessible` is `true`
- `S-NIAH @ 4K` passes the short-context sanity threshold
- `S-NIAH @ 32K` passes the long-context threshold
- `Nulls` are low or zero across tasks
- the degradation curve looks gradual rather than broken
