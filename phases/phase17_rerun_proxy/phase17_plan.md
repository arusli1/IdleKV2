# Phase 17: Paper edit plan for scorer and runtime framing

## Goal

Make the paper's scorer and runtime story cleaner:

- keep exact-Q2 quality results as the main controlled evidence;
- discuss runtime through the dedicated GPU runtime experiment;
- move proxy out of the main-text story;
- remove defensive phrasing and repeated caveats;
- focus wording on what the experiments measure and enable.

Do not rerun quality experiments for this phase. Do not edit `paper/main.pdf`
directly; edit `paper/main.tex` and rebuild.

## Edit 1: Scorers paragraph

Target: `paper/main.tex`, the Results setup paragraph beginning:

```tex
\paragraph{Scorers.}
The turn-$N$ eviction signal is the exact generated $Q_1$ answer tail
instead of a generic last-$32$-token cache suffix.
The main curves use an exact Q$_2$ attention-query scorer that extracts
the model's $Q_2$ self-attention query vectors after RoPE and scores
them against active and offloaded keys.
The latency probe also measures a cheaper $Q_2$-appended-state proxy
scorer, reported separately from the exact quality curves.
```

Replace with:

```tex
\paragraph{Scorers.}
The turn-$N$ eviction signal is the exact generated $Q_1$ answer tail
instead of a generic last-$32$-token cache suffix.
For the main quality curves, we append $Q_2$ to the compressed active
cache, extract the model's post-RoPE $Q_2$ query projections, and score
evicted rows against the active and offloaded keys. This exact-Q$_2$
selector gives a controlled measurement of how well the newly revealed
turn signal identifies rows worth promoting.
```

Reason:

- Says what exact-Q2 actually does.
- Avoids down-facing caveats.
- Removes proxy from the main scorer paragraph.
- Distinguishes selector signal from answer metric without overexplaining.

## Edit 2: Latency accounting paragraph

Target: `paper/main.tex`, paragraph beginning:

```tex
\paragraph{Latency accounting.}
The exact Q$_2$ scorer is a mechanism diagnostic for the controlled quality
curves, not the proposed runtime implementation.
A $Q_2$-appended-state proxy provides benchmark evidence for a cheaper
Q$_2$-conditioned scoring path: on paired 4Q/6Q runs at $K=96$, it scores
$0.970/0.894$ versus matched $0.245/0.422$.
Appendix Figure~\ref{fig:app-proxy-controlled} gives the proxy frontier.
Production deployment would require GPU-side or \textcolor{red}{paged host-memory} selection
integrated with scheduler slack.
```

Replace with:

```tex
\paragraph{Runtime accounting.}
We run a separate GPU runtime probe for the repair operation. The probe
measures chunked scan, top-$K$ selection, and KV movement over pinned
host-memory BF16 tensors with Qwen-shaped dimensions. Figure~\ref{fig:runtime-envelope}
reports the resulting capacity envelope, giving a controlled estimate of
how repair time scales with the number of offloaded candidate rows and
the restore budget.
```

Reason:

- Runtime is presented as its own experiment.
- No discussion of quality-run latency.
- No proxy numbers or proxy pointer in the main flow.

## Edit 2a: Contribution bullet

Target: contribution bullet that currently says:

```tex
\item a repair prototype with \textcolor{red}{transfer/promotion} feasibility results
and an instrumented latency probe that separates state movement, proxy
scoring, and the diagnostic exact Q$_2$ attention-query scorer.
```

Replace with:

```tex
\item \textcolor{red}{a repair prototype with transfer/promotion feasibility
results and a controlled GPU runtime probe for chunked scan, top-$K$
selection, and KV movement.}
```

Reason:

- Removes main-text proxy mention.
- Aligns the contribution with Figure 6.

## Edit 2b: Method scoring definition

Target: method paragraph currently saying:

```tex
... Ties break
by the per-token score from the turn-$N$ first-stage compression, then
by ascending position. The proxy
scorer obtains $\{q^{(\ell,h)}_{2,t}\}$ by appending $Q_2$ to the
compressed cache and reusing the resulting cache rows; selection is
unchanged.
```

Replace the proxy sentence with no main-text proxy definition:

```tex
... Ties break
by the per-token score from the turn-$N$ first-stage compression, then
by ascending position.
```

Reason:

- Proxy details move to the appendix.
- Main method describes the exact-Q2 selector used for the quality claims.

## Edit 3: Capacity-envelope paragraph

Target: immediately after Edit 2, current paragraph beginning:

```tex
\paragraph{Capacity envelope.}
Figure~\ref{fig:runtime-envelope} reports a single-node p95 envelope for a
chunked GPU selector over pinned host-memory BF16 keys.
At $K=5000$, p95 grows from $50.0$\,ms at 32K candidates to $1.20$\,s at
1M and $4.64$\,s at 4M.
The plot should be read as a capacity envelope; trace-scheduled latency
distributions remain a systems follow-up.
```

Replace with:

```tex
\paragraph{Capacity envelope.}
At $K=5000$, the p95 repair envelope grows from $50.0$\,ms at 32K
offloaded candidates to $1.20$\,s at 1M and $4.64$\,s at 4M.
These measurements show the operating range for a single-node GPU
implementation; trace-scheduled serving latency is a natural next systems
step.
```

Reason:

- Avoids repeating the full Figure 6 setup from the previous paragraph.
- Positive framing: operating range and next systems step.
- Keeps the numeric claim.

## Edit 4: Figure 6 caption

Target: Figure `\ref{fig:runtime-envelope}` caption:

```tex
\caption{Runtime-capacity probe on the evaluation GPU. (a) Component-summed
p95 service time at query length 64. (b) Stage-wise p95 component time for
$K=5000$. Generation and the research exact answer scorer are excluded.}
```

Replace with:

```tex
\caption{Runtime-capacity probe on the evaluation GPU. (a) Component-summed
p95 service time for chunked scan, top-$K$ selection, and KV movement at
query length 64 over pinned host-memory BF16 KV-shaped tensors.
(b) Stage-wise p95 component time for $K=5000$. Generation and answer
evaluation are excluded.}
```

Reason:

- Says exactly what is measured.
- Removes "research exact answer scorer."
- Keeps exclusions short and concrete.

## Edit 5: Discussion limitation sentence

Target: Discussion sentence:

```tex
The exact scorer is a diagnostic rather than a production selector, the
proxy scorer is only a first approximation, and stronger $Q_2$-aware
reselectors or full-prefix recompute may dominate when enough
\textcolor{red}{idle-window slack} is available.
```

Replace with:

```tex
The exact-Q$_2$ selector provides a controlled view of the repair signal,
and stronger $Q_2$-aware reselectors, page-level policies, or full-prefix
recompute may further improve the quality-runtime frontier when enough
\textcolor{red}{idle-window slack} is available.
```

Reason:

- Up-facing: states what exact-Q2 contributes and what future systems can
  improve.
- Removes the repeated proxy caveat.
- Avoids "rather than" framing.

## Edit 6: Appendix scorer details

Target: appendix paragraph beginning:

```tex
The exact Q$_2$ attention-query scorer extracts the model's actual
self-attention query vectors for $Q_2$ prompt tokens after RoPE and
scores them against active and offloaded keys.
This is distinct from decode-time answer-token attention.
The $Q_2$-appended-state proxy instead appends $Q_2$ to the compressed
cache with an ordinary forward pass and reuses the appended $Q_2$ cache
rows as scoring rows.
Both scorers are benchmark instrumentation rather than fused production
serving kernels.
```

Replace with:

```tex
The exact-Q$_2$ attention-query selector appends $Q_2$ to the compressed
active cache, extracts the model's post-RoPE query projections for the
$Q_2$ prompt tokens, and scores evicted rows against active and offloaded
keys. This selector is distinct from decode-time answer-token attention
and from the final answer metric. The $Q_2$-appended-state proxy appends
$Q_2$ to the compressed cache and uses the appended cache rows as a
second Q$_2$-conditioned scoring signal.
```

Reason:

- Appendix can carry the proxy definition.
- No repeated production caveat.
- Exact/proxy distinction is technical and compact.

## Edit 7: Appendix runtime/proxy figure

Target: appendix Figure `\ref{fig:app-proxy-controlled}` block.

Preferred direction:

- Keep the proxy panel in the appendix. The review consensus is that it is
  useful as an additional selector diagnostic, but too distracting for the
  main text.
- Do not mention proxy in the main text.
- If proxy stays, retitle the panel as a diagnostic rather than a runtime
  result.

Replace caption text with:

```tex
\caption{Supplementary runtime and selector diagnostics. (a) Synthetic
move-and-inject capacity for Qwen-shaped BF16 KV on the evaluation GPU,
separate from the Figure~\ref{fig:runtime-envelope} scan/select envelope.
Measurements use pinned host-memory \textcolor{red}{promoted} blocks and report p95
latency as $K$ or active cache size grows. (b) Appendix proxy-selector
diagnostic on MQ-NIAH-4Q/6Q ($100$ examples per split). At $K=96$,
content-agnostic \textcolor{red}{promotions} stay within $0.010$/$0.004$ of matched
no-repair while proxy \repairkv{} remains well above matched.}
```

If proxy is cut entirely:

- remove panel (b);
- rename the figure label away from `app-proxy-controlled`;
- keep only the move-and-inject capacity panel if it still adds value beyond
  Figure 6.

Current recommendation: do not cut proxy entirely. Keep it appendix-only.

Reason:

- Keeps main paper simpler.
- Retains proxy only as appendix robustness evidence.
- Avoids making readers wonder why proxy is not the headline method.

## Figure 6 audit before edits

Run:

```bash
head -n 2 paper/figures/runtime_latency_envelope_select.csv
head -n 2 paper/figures/runtime_latency_envelope_move.csv
```

Required provenance:

- `device=cuda`
- `dtype=bfloat16`
- `pin_memory_effective=True`
- `host_pool_coverage=1.0` for select rows
- `trials=80`
- `chunk_tokens=16384`
- candidate rows include `32768`, `1048576`, and `4194304`
- `query_len=64` rows exist
- `K=5000` rows exist

Numeric check:

```text
p95_total = p95_select_total_ms(candidate_tokens, K=5000, query_len=64)
          + p95_move_total_ms(active_tokens=32768, K=5000)
```

Expected values:

- 32K candidates: about `50.0 ms`
- 1M candidates: about `1.20 s`
- 4M candidates: about `4.64 s`

Rerun Figure 6 only if provenance or numeric checks fail, or if the paper
starts claiming end-to-end serving latency rather than a controlled GPU
capacity envelope.

## Validation after paper edits

Run:

```bash
rg -n "full uncompressed|production selector|canonical deployable|answer scorer|not a claim|rather than a fused|proxy is" paper/main.tex
rg -n "exact-Q\\$_2\\$|exact Q\\$_2\\$|proxy-selector|runtime-envelope|capacity envelope" paper/main.tex
.venv/bin/python paper/scripts/render_paper_figures.py
cd paper && latexmk -pdf -interaction=nonstopmode main.tex
```

Then inspect:

```bash
rg -n "Warning|Error|undefined|multiply defined|Overfull" paper/main.log
```

Expected:

- no "full uncompressed cache" claim for exact scoring;
- no "not a claim about..." defensive phrasing;
- no repeated fused-kernel caveat;
- no "research exact answer scorer" phrase;
- proxy absent from main text; proxy appendix-only;
- Figure 6 framed as the runtime experiment.

## Open questions

- Should proxy remain as an appendix-only diagnostic, or should the appendix
  proxy panel be cut entirely?
- Confirm after final proofread that proxy has no main-text pointer.
- Does Figure 6 caption need an explicit exclusion sentence, or is the
  positive measurement description enough?
- Should the runtime paragraph say "GPU runtime probe" or "controlled runtime
  probe" for the most natural paper tone?
