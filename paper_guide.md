# IdleKV Paper Guide

Last verified: 2026-05-03.

Read this file before editing `paper/main.tex`. Treat it as the paper
style contract unless the venue instructions change.

## Source Of Truth

- AdaptFM workshop site: https://adaptfm.gitlab.io/call-for-papers.html
- ICML 2026 template/example paper:
  https://media.icml.cc/Conferences/ICML2026/Styles/example_paper.pdf
- Adjacent KV-cache/effective-inference style references:
  - SnapKV, NeurIPS 2024:
    https://papers.nips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf
  - H2O, NeurIPS 2023:
    https://papers.nips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf
  - Scissorhands, NeurIPS 2023:
    https://proceedings.neurips.cc/paper_files/paper/2023/file/a452a7c6c463e4ae8fbdc614c6e983e6-Paper-Conference.pdf
  - StreamingLLM, ICLR 2024:
    https://arxiv.org/abs/2309.17453
  - QUEST, ICML 2024:
    https://arxiv.org/abs/2406.10774
  - PyramidKV:
    https://arxiv.org/abs/2406.02069
  - KIVI, ICML 2024:
    https://proceedings.mlr.press/v235/liu24bz.html
  - Dynamic Memory Compression, ICML 2024:
    https://proceedings.mlr.press/v235/nawrot24a.html
- AdaptFM appears to be a new ICML 2026 workshop, so there are no prior
  AdaptFM accepted papers to imitate directly. Use the official call/template
  plus adjacent ICML/ICLR KV-cache and efficient-inference papers as style
  references.
- Current paper: `paper/main.tex`
- Idea-flow outline: `paper/outline.md`. Read it before major rewrites as a
  suggested narrative map, not as ground truth. Follow it when it still fits
  the evidence and paper constraints; update or override it when results,
  reviewer-risk, or concision require a different flow.
- Phase 9 experiment/figure plan:
  `phases/phase9_experiment_deepening/phase9_plan.md`

## Venue Constraints

- Venue: AdaptFM, Resource-Adaptive Foundation Model Inference, co-located
  with ICML 2026.
- Submission deadline: May 8, 2026 AoE.
- Maximum main length: 6 pages excluding references, using the official ICML
  2026 template.
- This is a ceiling, not a target. A shorter paper is preferable if every
  paragraph, figure, and table earns its space.
- Appendices are allowed, but reviewers are not required to read them.
- AdaptFM explicitly permits unlimited appendix pages in the same PDF. The
  appendix supports auditability; it cannot carry any claim required to
  understand or accept the main paper.
- ICML templates allow either a one-column or two-column appendix. For this
  paper, prefer the default two-column flow unless a specific artifact is
  unreadable at column width. Do not change margins, font size, caption style,
  or page numbering to force spacing.
- The ICML 2026 example places `\newpage` before `\appendix`, then uses
  `\onecolumn` for the appendix while noting that two-column appendices are
  allowed. For this paper, use the example's one-column appendix layout because
  the appendix is figure-heavy and the two-column version produced large
  flush-bottom gaps between paragraph blocks.
- Submissions are double-blind unless preparing a camera-ready version.
- For anonymous review, do not include visible author names,
  acknowledgments, grant numbers, funding statements, or non-anonymized
  repository links. The ICML style may keep hidden author metadata in
  `\icmlauthor`/`\icmlaffiliation`, but it must not render in the review
  PDF.
- For camera-ready, turn on the accepted style option and include author
  names/affiliations. The intended metadata is Andrew Rusli, Harvard
  College; Shreyan Paliwal, Harvard College and Opt32. Acknowledgments and
  funding can be added then if space and venue instructions permit.
- Contribution statements are not part of the ICML template by default.
  Use the existing equal-contribution footnote for equal first authors;
  do not add a visible contribution section unless AdaptFM/OpenReview
  explicitly asks for one.
- Use the official ICML style. Do not edit margins, font sizes, spacing, or
  template commands to gain space.
- The AdaptFM scope explicitly includes dynamic KV cache compression, runtime
  systems for flexible computation, benchmarking/profiling across resource
  budgets, and quality-resource tradeoff analysis.

## Overall Thesis And Experimental Thesis

The paper is not "better KV compression in general."

Overall thesis:

> Agent workflows are not always query-answer loops. A later turn can shift
> relevance through a user correction, tool result, test failure, file change,
> new topic, or explicit query. A KV cache for such workflows should be treated
> as mutable maintained state, not only as a one-shot compressed artifact.

Systems thesis:

> A compressed paused KV cache can be repaired between turns once the next
> turn supplies a new relevance signal. This uses otherwise idle time to revise
> active memory under the same resumed active-cache budget, supporting dynamic
> cache maintenance as a resource-adaptive inference primitive.

Experimental thesis:

> The paper tests the smallest controlled instance of that broader idea:
> split-query retrieval, where the next-turn relevance signal is an explicit
> `Q2` query. Success here is not the whole agent workflow claim; it is
> preliminary evidence that post-compression, pre-resume repair can matter and
> deserves broader benchmarks and stronger algorithms.

Keep broader claims framed as implications or research agenda:

- Good: "This suggests idle-window cache maintenance is a useful primitive for
  resource-adaptive agent inference."
- Good: "The benchmark instantiates next-turn relevance as a query, but the
  systems problem is broader: any new turn text or tool result can change what
  past context matters."
- Too strong: "IdleKV improves agent performance."
- Good: "Exact scoring is mechanistic evidence; proxy scoring shows a possible
  latency path."
- Too strong: "IdleKV is deployable at production latency."

## Abstract Rules

ICML guidance: one paragraph, self-contained, roughly 4-6 sentences.

Target abstract structure, usually 5 sentences:

1. Problem and motivation: why idle gaps and KV compression create a new issue.
2. Gap: existing compression/serving/query-aware methods do not repair an
   already-compressed paused cache after the next-turn relevance signal is
   known.
3. Method: IdleKV buffers evicted KV, scores it after the controlled `Q2`
   signal is known, restores a budget `K`, and resumes under a matched
   active-cache budget.
4. Main result: the strongest compact numbers from calibrated
   4Q/6Q/8Q panels.
5. Scope and implication: controlled diagnostics, not end-to-end agent proof;
   supports dynamic cache maintenance/resource-adaptive inference.

Avoid:

- More than 6 sentences.
- Undefined acronyms beyond common venue terms. Define key-value (KV) on first
  use.
- Internal phase names.
- Long lists of secondary runtime/proxy details unless they are central to the
  current abstract.

## Recommended Main Sections

For a 6-page AdaptFM workshop paper, prefer:

1. Introduction
   - Motivation.
   - Problem setting.
   - Contributions.
2. Related Work
   - Compact novelty boundary against serving systems, compression, query-aware
     KV selection/loading, and KV-lifecycle benchmarks.
3. Method
   - Two-turn protocol.
   - Repair unit and budget.
   - Matched no-repair baseline.
   - Metrics and hindsight reference.
4. Experimental Setup
   - Model, task, calibrated settings.
   - Conditions and scorer semantics.
   - Scope.
5. Results
   - Main matched-budget frontier.
   - Robustness/heterogeneity.
   - Mechanism and runtime caveats.
   - New heatmap/operating-regime figure if the final run is strong.
6. Discussion and Limitations
   - Resource-adaptive positioning.
   - Applicability criteria.
   - Limitations and next steps.
7. Conclusion
   - One concise paragraph. No new claims.

Appendix:

- Graph-first supplementary views for sweeps, coverage diagnostics, and
  quality/latency tradeoffs.
- Dense endpoint tables only when exact numbers are easier to audit than read
  from a plot.
- Per-partition tables when they support robustness/cherry-picking checks.
- Failed/null controls only when they prevent a plausible reviewer
  misinterpretation.
- Additional mechanism plots that answer a distinct reviewer question.

## Terminology

Use paper-facing terminology, not internal repo terminology.

- "key-value (KV) cache": define on first use.
- "active cache": KV rows resident and visible to the next decode.
- "host-memory evicted-KV store" or "offloaded evicted-KV store": KV rows not
  active but retained for possible restoration. Use "CPU" only when describing
  this prototype's implementation or measured CPU-GPU transfers; use "host
  memory" or "offloaded store" for framework-level prose.
- "restore budget `K`": number of evicted context tokens restored before `Q2`
  decoding. Clarify that it is not an anchor count or byte budget.
- "matched no-repair": baseline that gives `Q2` the same number of active
  evictable-context tokens as IdleKV but does not use idle-window repair.
- "matched resumed active-cache budget": acceptable but define it; do not use
  "matched footprint" unless explicitly explaining memory accounting.
- "score gain over matched no-repair": preferred prose for
  `Score(IdleKV) - Score(B_match)`.
- `Delta_repair`: compact metric symbol for score gain over matched no-repair.
- "proxy/exact gain ratio": preferred term for fixed-K proxy/exact comparisons.
  Define it as `(Proxy - matched no-repair)/(Exact - matched no-repair)`.
- "`Gold-K` benchmark-metadata hindsight reference": acceptable if defined.
  Do not call it "oracle" in prose.
- "H2O-inspired accumulated-attention retention": use this wording for the
  current accumulated-attention policy branch unless the implementation exactly
  reproduces canonical H2O accumulated decode attention. Do not write
  "H2O result" in the paper without defining the approximation.
- "Sink-plus-recent retention inspired by StreamingLLM": use this wording for
  the current sink-plus-recent policy branch. Do not write "StreamingLLM
  result" unless the implementation includes the original
  streaming/position-management protocol rather than only the structural
  retention rule.
- "Exact H2O" is only allowed if the run logs actual attention weights during
  generation and applies the published heavy-hitter plus recent-token eviction
  policy under explicit budget accounting.
- "Exact StreamingLLM" is only allowed for a rolling streaming setup that
  preserves sink tokens, maintains a recent window, and handles the original
  cache/position-management assumptions. For our two-turn protocol, prefer
  "sink-plus-recent retention" because it names the tested mechanism.
- "Exact Scissorhands" is a plausible future retention baseline because it is
  also an attention-history fixed-budget cache policy. It needs its own tests
  and should not be used as shorthand for the current accumulated-attention
  branch.
- "SnapKV-style" is acceptable only when describing this repo's
  protocol-matched implementation. In main prose, define the implemented
  policy once as "context-only SnapKV-style retention" or "SnapKV-style
  first-stage retention" rather than implying a byte-for-byte reproduction of
  the upstream system.
- Exact-method shorthand:
  - H2O: dynamic retention of recent tokens plus heavy-hitter tokens from
    actual accumulated attention history.
  - StreamingLLM: rolling streaming inference with attention sinks plus a
    recent window and the method's cache/position assumptions.
  - Scissorhands: fixed-budget cache retention based on persistence of token
    importance from attention history.
  If a run lacks those runtime signals or accounting rules, name the mechanism
  directly and use "inspired by" only as a literature pointer.
- "QUEST-style" should not be used for the current repair scorer unless the
  implementation operates at page level with query-vector page criticality.
  QUEST is more naturally related to future page-level repair than to the
  first-stage retention-rule sweep.
- "Refresh-buffered": bounded Q2-time full-budget reselection over active plus
  offloaded evicted-KV rows, without full-prefix recompute. Do not call it
  plain "Refresh-K" in paper-facing prose, and do not describe it as a learned
  or systems-fair full refresh baseline.
- "exact Q2 attention-query scorer": preferred term for the current exact
  scorer. Avoid unexplained "question-query scorer."
- "proxy scorer": only after defining how it appends `Q2` and reuses the
  resulting cache rows.
- "next-turn relevance signal": broad workflow term. In experiments this is
  the `Q2` query; in discussion it can also cover tool output, user correction,
  topic shift, file/test feedback, or other new turn text.

Prior-algorithm fidelity checklist:

- Use "exact" only when the implementation reproduces the algorithm's
  information source, update timing, and budget accounting, not just its
  high-level intuition.
- H2O is an exact baseline only with decode-time accumulated attention history
  and a heavy-hitter plus recent-token eviction rule. A frozen-cache
  dot-product approximation is H2O-inspired accumulated-attention retention.
- StreamingLLM is an exact baseline only in a rolling streaming setup with
  sink tokens, a recent window, and the method's cache/position assumptions.
  In this paper's two-turn protocol, the rigorous name is sink-plus-recent
  retention inspired by StreamingLLM.
- Scissorhands is the best next exact fixed-budget eviction baseline if a
  named-algorithm reproduction becomes worth the implementation cost: it
  directly tests attention-importance persistence under a bounded KV cache.
  A faithful branch needs real attention-history logging, the paper's
  fixed-buffer update rule, deterministic tests for ties/budgets, and a smoke
  showing that the intended pivotal-token rows survive before any MQ-NIAH run.
- FastGen is an exact baseline only after adding its attention-head profiling
  stage and head-specific retention patterns. It is useful related work but
  too different from the current global-position repair protocol for a quick
  policy-breadth row.
- PyramidKV/Ada-KV-like methods should be deferred unless the repair machinery
  supports layer-varying retained positions and matched layer-wise budget
  accounting.
- QUEST should be discussed as query-aware page loading or future page-level
  repair, not as a first-stage retention rule in the current global-token
  protocol.

Never include:

- "phase 7", "phase 8", "phase 9" in paper-facing text.
- "bridge", "extension", "clean suite", "full suite", or run nicknames.
- "perfect data", "best data", or hype language.

## Figure Strategy

AdaptFM short workshop papers and related KV-cache systems papers are
figure-forward. Our main paper should be graph-first, not table-first.

Adjacent paper structure audit:

- H2O spends the introduction on the deployment bottleneck, three empirical
  observations, and why they imply a cache policy. Results are organized by
  accuracy/memory, throughput/latency, and ablations; prose states the insight
  and leaves dense numeric grids to tables/figures.
- SnapKV spends early space on the observation that prompt-token importance can
  be inferred before generation, then uses a method schematic and broad
  benchmark plots. Its main text does not walk through every curve; it argues why
  the observation supports the compressor.
- QUEST follows a strong systems-paper pattern: cost bottleneck, observation
  that token criticality is query dependent, page-level algorithm, accuracy
  frontiers, and kernel/runtime breakdown. This is the closest model for
  IdleKV's "next-turn relevance signal" framing.
- Modern tiered-KV systems such as Dynamo/KVBM, LMCache, and TTKV should be
  treated as substrates for storage, routing, transfer, and reuse. IdleKV should
  spend words on the missing policy question: which buffered KV units should be
  promoted after the next-turn signal arrives?
- Use production documentation and engineering blogs, such as NVIDIA Dynamo docs,
  for internal audit and claim-boundary checks. Do not cite them in the main
  bibliography unless a specific deployed-system fact is essential and no formal
  paper is available. Prefer peer-reviewed or archival papers in the reference
  list so the bibliography looks like adjacent ICML/NeurIPS systems papers.

Reference patterns from adjacent papers:

- SnapKV leads with a method schematic, then uses small multi-panel line plots
  to support a specific mechanistic claim about prompt-dependent attention
  features. Its figures usually answer one question per caption.
- H2O uses dense small multiples when the claim is broad model/task robustness,
  with a full-cache reference line and one or two strong baselines. It moves
  many task-specific details to tables/appendices.
- QUEST motivates query-aware cache access with a workflow figure plus compact
  quality/latency comparisons; use this as the comparator for our
  "next-turn relevance signal" language.
- PyramidKV and StreamingLLM both make the systems setting concrete before
  presenting broad benchmark results. They define the cache policy and the
  baseline memory/latency setting before reporting headline numbers.

Target main visual package:

1. Method/pipeline schematic.
2. Main matched-budget frontier. Use a one-column raw-score overlay over
   restore budget when the visual remains readable: IdleKV curves for
   2Q/4Q/6Q/8Q, faint matched no-repair traces, neutral query-count labels,
   and right-side endpoint or K=96 difference labels. Do not include Gold-K in
   this main figure unless it answers a new question; it usually makes the
   frontier too busy.
3. Specificity contrast if the locked Phase 10 run passes: use score gain
   over matched no-repair with uncertainty plus paired win/tie/loss rates.
   Label the bounded comparator as Refresh-buffered in caption/prose; a
   shortened axis label such as Refresh is acceptable only if the caption
   defines it.
4. Operating-regime heatmap in the main text only while it replaces weaker
   prose and answers "where does repair help?" Use normalized recovery,
   `IdleKV gain / Gold-K gain`, so the plot separates repair effectiveness
   from raw benchmark headroom. Demote it to appendix if a stronger
   specificity or multi-turn figure needs the main-text slot.
5. Multi-turn relevance-shift trajectory if the locked run passes the
   numerical gate and paired uncertainty is available. The main plot should
   show IdleKV, StaleQ-K, Gold-K, and a Random-K/Oldest-K band, with revisit
   turns marked. Keep CurrentQOnly-K/StaleQOnly-K as audit or caption/prose
   evidence unless adding them remains clearly readable.
6. Compact algorithm box or pseudocode only if it can replace prose and add
   reproducible detail beyond the method schematic.
7. Optional mechanism or latency plot only if it replaces text/table space and
   adds a distinct claim.

Tables should mostly move to appendix unless they summarize a small endpoint
that cannot be read cleanly from a plot.

Main-text word budget learned from H2O/SnapKV/QUEST:

- H2O spends main-section words on deployment bottleneck, one defining
  observation, method policy, accuracy, throughput/latency, and ablations. It
  does not narrate run provenance in the main text.
- SnapKV spends substantial main-text space on the observation that important
  prompt features are input/instruction dependent before presenting the
  compressor. For IdleKV, the analogous observation is next-turn relevance:
  what should remain active can change after a paused workflow reveals the next
  query/tool/user signal.
- QUEST frames query-aware sparsity by first defining why long-context decode is
  costly, then showing that critical tokens depend on the current query, then
  presenting an algorithm and quality/latency evidence. IdleKV should mirror
  that progression: cost/lifecycle, stale resumed cache problem, repair
  operation, matched-budget evidence.
- Therefore spend main-paper words on the dynamic pause/resume cache lifecycle,
  the matched resumed active-cache-budget problem, the repair mechanism, and the
  strongest controlled evidence. Move exact split lists, long artifact
  provenance, null smokes, broad phase planning, and exhaustive tables to the
  appendix or internal notes.

Main-paper diagnostic gate:

- Do not keep coverage-only or smoke-only diagnostics in the main Results
  section. They can motivate future work in the appendix, but the main
  figures/tables should report answer quality, specificity, runtime, or a
  clearly defined mechanism tied to the headline claim.
- A streaming spill table or coverage-only figure belongs in notes, not the
  current appendix, unless it includes Q2-conditioned repair quality, matched
  content-agnostic restore controls, and enough examples that the trend is not
  a low-sample artifact.
- If promoted, the operating-regime heatmap must be described as a
  within-task regime diagnostic. It should answer budget calibration and
  cherry-picking concerns, not be used as cross-task absolute effect-size
  evidence.

## Figure Formatting

ICML figure rules to preserve:

- Center figures.
- Keep artwork legible.
- Use dark lines at least 0.5 pt.
- Label axes and distinct components.
- Use captions instead of titles inside the graphic.
- Put figure captions below figures.
- Legends must not cover data; prefer below or outside the plot.
- Prefer vector formats for plots.
- Keep color semantics stable across the paper. Current convention: IdleKV is
  Okabe-Ito blue; Gold-K/reference headroom is orange/gold; matched and
  content-agnostic controls are black/gray; stale/refresh/query-control
  diagnostics are purple; accumulated-attention policy variants are green; and
  sink-plus-recent/policy variants that need a third policy color use
  vermillion. Do not reuse purple for a policy curve while it denotes stale
  query controls elsewhere.
- Two-column figures are allowed but should be at top or bottom; use sparingly
  in a 6-page workshop paper.
- Main-paper captions should be concise, usually about 35-70 words for a
  one-column result figure and rarely above about 90 words unless the figure is
  an overview diagram or dense multi-panel result. A caption should state the
  plotted quantity, the critical setup (`n`, budget, model/task when needed),
  visual encodings, and one caveat only if it prevents misreading. Move
  interpretation, secondary numbers, and limitations to the surrounding prose
  or appendix.

IdleKV-specific figure rules:

- Prefer one-column figures when possible.
- Use small multiples, heatmaps, or frontiers rather than crowded tables.
- Avoid decorative charts, radar charts, and low-information line plots.
- For heatmaps: use a sparse grid, a zero-anchored sequential palette, one
  shared colorbar, concise tick labels, and numeric cell labels only when the
  grid is at most about `3 x 4` per panel. Avoid star/corner-marker overlays
  unless the marker encodes the central claim; encode secondary diagnostics in
  caption/prose or an appendix table.
- For contrast plots: use direct y-axis labels and keep legends outside the
  data region. Show uncertainty intervals for promoted runs; no main-paper
  `n=1` smoke plots.
- For frontiers: include only baselines that establish the claim. Random/oldest
  are useful anti-generic-reinsertion controls; avoid adding every condition to
  the main plot if the legend becomes the visual center.
- A single raw-score overlay is acceptable when query-count breadth is the main
  story: use solid IdleKV curves, faint matched no-repair traces, direct
  query-count labels, and move Gold-K plus Random/Oldest details to prose or
  appendix tables. Do not normalize the main frontier just to merge panels.
- Include 2Q/4Q/6Q/8Q in the main frontier only from full K-grid evidence.
  It is acceptable for 2Q to saturate, but keep panel labels neutral and put
  interpretation in prose. Endpoint-only query-count evidence belongs in
  appendix breadth.
- For multi-turn diagnostics: promote only locked runs, not smokes. Prefer raw
  exact score when a Gold-K/reference marker is shown, so the reference remains
  visually interpretable as the upper benchmark-metadata score. If using gain
  over matched no-repair, omit or clearly remap references. The main diagnostic
  must include StaleQ-K if the claim is dynamic next-turn adaptation; otherwise
  reviewers can attribute the effect to stale query reuse. If CurrentQOnly-K
  and StaleQOnly-K diagnostics are present, require current-query-only repair
  to separate from stale-query-only repair before promoting the result.
- For cross-model evidence: require a full-cache/matched ability gate and a
  cache round-trip check before any repair comparison. Failed small-model
  ability checks belong in notes, not in the main paper.
- Each figure must answer one reviewer question:
  - Does repair beat matched no-repair?
  - When does repair help or disappear?
  - Does repair move active cache toward future-query spans?
  - Is exact scoring the latency bottleneck, and is there a faster path?
- Do not plot every available condition. Plot only conditions needed for the
  claim; move the rest to appendix.
- More high-signal objects are better than long prose. More low-signal objects
  are worse. Promote a new figure/algorithm only if it removes text, replaces a
  table, or answers a distinct reviewer question.

Current Phase 10 gate:

- `K=48` is the locked specificity operating point because the `n=1` smoke
  separates IdleKV from stale and donor wrong-query controls while leaving
  Gold-K headroom.
- `K=96` should not anchor next-turn specificity because the smoke showed
  stale-query catching up.
- If the locked run keeps `IdleKV > StaleQ-K` and `IdleKV > WrongQ-K` but
  `Refresh-buffered > IdleKV`, the paper claim is still useful but narrower:
  IdleKV is an incremental buffered-repair primitive, not the best possible
  Q2-time full-budget reselection.
- If `StaleQ-K` matches IdleKV in the locked run, or if `CurrentQOnly-K`
  does not beat `StaleQOnly-K` in a redesigned run, do not claim broad
  multi-turn adaptation in the abstract or introduction.

## Algorithm Boxes

Top systems/KV-cache papers often use compact method diagrams, pipeline
figures, and occasional algorithm boxes. Use an algorithm box only when it
clarifies the protocol better than prose.

Reference pattern:

- QUEST uses Algorithm 1 only after Section 3 has introduced the page-selection
  problem and Figure 5 has shown the workflow. The algorithm is implementation
  facing: it lists inputs, metadata updates, and the self-attention-time
  selection path.
- ShadowKV uses Algorithm 1/2 inside the method section after observations and
  insights establish why the algorithm exists. The algorithms are paired with
  short "Pre-filling" and "Decoding" paragraphs.
- KVzip puts detailed pseudocode in the appendix because it is implementation
  detail rather than the central main-text argument.

IdleKV rule:

- Do not keep a main-text Algorithm 1 unless it states the actual selector with
  enough detail to be reproducible: Q2-score computation, turn-N tie-break,
  burst width, overlap handling, and budget filling.
- If the algorithm duplicates Figure 1 without adding operational clarity,
  remove it and spend the space on a stronger figure or tighter result prose.
  Current main-paper decision: no Algorithm 1; keep precise prose plus Figure 1.

Good algorithm-box candidate:

- Inputs: compressed active cache, offloaded evicted-KV store, next-turn relevance text
  or `Q2`, restore budget `K`.
- Steps: score buffer, select non-overlapping local bursts, transfer selected
  KV, inject before resumption.
- Outputs: repaired active cache and unchanged offloaded-store accounting caveat.

Avoid:

- Long pseudocode for implementation details.
- Algorithm boxes that duplicate the pipeline figure exactly.
- New notation not used elsewhere.

## Table Strategy

Use tables for:

- Exact endpoint values.
- Per-partition robustness.
- Runtime accounting only when a compact plot would hide the relevant
  measurement.
- Proxy/exact paired comparisons only when the exact paired values matter more
  than the quality/latency tradeoff shape.
- Appendix reproducibility details.

Runtime table rules:

- Do not mix fundamentally different latency paths as if they are one protocol.
  If exact answer scoring, proxy answer scoring, KV movement, and
  offloaded-store scans all appear, separate them by role: quality-linked
  scorer diagnostics versus systems-capacity measurements.
- A "latency envelope" plot should put offloaded candidate-store size or
  context/KV rows on the x-axis and p95 repair service time on the y-axis.
  Horizontal idle-window thresholds are acceptable only when the caption states
  the budget fraction, the active-cache size, query length, restore budget, and
  whether generation is excluded.
- Idle-window gaps are continuous, not categorical. In main text, prefer a
  continuous latency-envelope figure plus a few prose anchor numbers. If
  discrete thresholds appear, call them representative idle-budget checkpoints
  and state the slack convention. Do not use an "idle" table column in the main
  paper unless it is explicitly tied to a measured idle-window distribution.
- For the main runtime claim, prefer one measured envelope: fixed hardware,
  fixed model-shaped KV layout, fixed active-cache size, fixed query length,
  explicit restored-row budget `K`, and p50/p95/p99 over enough trials.
- Report whether source data covered the whole measured candidate store,
  whether host memory was pinned, and whether the row includes scoring,
  transfer, reinsertion, and/or generation.
- Idle-window claims need an idle-window distribution. Without an empirical
  trace, write threshold statements such as "fits within a 1s idle window with
  10% slack" rather than "covers X% of tool calls."

Avoid main-text tables for:

- Full `K` sweeps already shown by a line plot.
- Split-by-split details.
- Repetitive condition grids.
- Anything whose message is primarily shape, trend, or tradeoff.
- Coverage sweeps, latency/quality tradeoffs, and operating-regime grids; these
  should usually be heatmaps, line plots, or compact small multiples instead.

ICML table rules:

- Caption/title above table.
- Center tables.
- Make rows and columns legible. A narrow table may sit at its natural width;
  do not stretch a small table to the column edge merely for appearance.
- Use booktabs style where possible.
- Put two-column tables at the top or bottom only if unavoidable.

## Appendix Strategy

ICML-style appendices are part of the same PDF, but reviewers are not required
to read them. The main paper must stand alone; the appendix should make the
evidence auditable and show high-signal secondary views without becoming a data
dump.

Reference pattern:

- Related KV-cache papers use appendices for additional ablations, robustness
  views, benchmark statistics, implementation details, and expanded
  throughput/latency evidence.
- Figures are preferred when the claim is a shape, threshold, trend, or
  tradeoff. Tables are preferred when the claim is exact reproducibility,
  paired audit values, or per-partition verification.

IdleKV appendix rules:

- Keep the appendix in the normal ICML two-column flow. Do not switch to
  `\onecolumn` unless a venue instruction or unavoidable artifact requires it.
- Use ordinary float placement (`[tbp]` or `[t]`) rather than pinned `[H]`
  floats; `[H]` often creates large white gaps in a two-column paper.
- Include one-column appendix plots at `\columnwidth`. Use `figure*` or
  `table*` only when an object truly needs both columns, or when several
  tail-end appendix diagnostics can be combined into one dense full-width panel
  that avoids an empty-column final page.
- Avoid ending on a mostly blank page caused by a full-width float when a
  one-column figure, shorter caption, or an additional high-signal appendix
  plot can make the page denser without spacing hacks.
- Do not include alternate figure variants that answer the same question as a
  main figure. Keep only the best view plus, if needed, one audit table.
- Prefer heatmaps for strict-cap/operating-regime grids and compact scatter or
  dumbbell plots for latency-quality tradeoffs.
- Treat same-family checkpoint changes as size-transfer or portability checks,
  not model-family diversity. A model-diversity claim needs a different model
  family such as Mistral/Llama/Phi/Gemma, plus a full-cache ability gate before
  any repair comparison.
- A lower-powered Llama result, even if positive, should be called a
  "locked cross-family portability check" or "preliminary Llama replication."
  Do not write that the method is robust across model families unless the run
  matches the main suite's sample size, K grid, and controls across multiple
  model families.
- Keep exact milestone and per-partition tables while they support Figure 2
  auditability. Remove or move them later if a stronger appendix plot replaces
  their audit role.

## Experimental Evidence Standard

Main text should prioritize:

- The locked 2Q/4Q/6Q/8Q exact frontier when full K-grid evidence is
  available and the low-query-count panel is described neutrally.
- A clean operating-regime heatmap if the final run confirms the smoke.
- Mechanism/runtime evidence only if it is compact and materially strengthens
  the thesis.

Do not add a main experiment just because it exists.
Do not add text just to use available pages. Cut setup bookkeeping and repeated
caveats when an appendix sentence or table can carry the reproducibility detail.
Prefer the accumulated-attention retention check over the older streaming spill
coverage diagnostic when choosing appendix robustness material. The
accumulated-attention result answers a reviewer generality question; the spill
heatmap only shows pre-repair accessible-token coverage.

Run hierarchy:

1. Unit tests for code changes.
2. Minimal smoke run to verify config and signal.
3. Full run in tmux only if the smoke supports the paper claim.
4. Export/plot validation.
5. Paper integration.
6. Rebuild `paper/main.pdf`.
7. Inspect PDF for spacing, legends, cut-off labels, and page count.

## Writing Style

Reference-paper style notes:

- SnapKV abstract pattern: long-context/KV-cache problem, gap in prior
  compression, one sentence naming the method, one sentence with concrete
  speed/memory/accuracy numbers, one sentence on practical implication.
- QUEST abstract/intro pattern: demand for long context, bottleneck, observation
  that criticality depends on the current query, method, then numbers. It uses
  "However", "To this end", "We show", and "In summary" sparingly as transition
  phrases.
- ShadowKV pattern: state limitations of prior systems as a short list, then
  present the system and its two core mechanisms. Method subsections use
  compact labels such as "Observation.", "Insights.", "Pre-filling.", and
  "High-throughput Decoding.".
- SCBench pattern: begin with a lifecycle diagram, define the under-specified
  benchmark setting, then enumerate categories/tasks. It is explicit about what
  existing benchmarks miss.
- KVzip pattern: use a front-page overview figure for the multi-query setting,
  then reserve long pseudocode and large benchmark matrices for the appendix.
  Its heatmap-style figures use compact square cells, shared color scales,
  short axis labels, and concise captions; avoid hand-drawn heatmap substitutes
  when a generated image asset can match this visual grammar.
- TinyServe workshop pattern: compact abstract, bold inline paragraph labels in
  the introduction, contribution bullets, and system components listed only
  after the motivation is clear.

Figure and diagram rules:

- Use vector PDFs for plots whenever possible. Match the paper typography:
  Times-compatible serif text, approximately 6.5-8 pt labels at final
  included size, thin axes, and embedded fonts. The ICML example explicitly
  uses 10 pt Times, captions below figures, and no large titles inside the
  graphic file; keep titles in captions or as small panel labels.
- Maintain one paper color system: IdleKV blue, hindsight/reference orange,
  matched/no-repair dark gray, random gray, oldest/static green. Use marker
  shape and line style in addition to color so grayscale and color-vision
  deficient readers can still read the plot.
- Avoid rainbow/red-green colormaps. For heatmaps, use a perceptually ordered
  sequential map with monotone lightness and a real colorbar; reserve diverging
  maps for quantities with a meaningful sign around zero.
- Experimental figures should be information-dense enough to earn space. A
  graph with only a few points belongs in prose or a compact table unless it
  communicates a key shape, threshold, or tradeoff. Prefer small multiples that
  answer distinct reviewer questions over isolated low-density panels.
- Method/system diagrams should usually be external vector assets, not raw
  plotting code. Use Figma, Illustrator, Keynote/PowerPoint, or diagrams.net;
  export PDF/SVG; include in LaTeX. The style target is a clean paper figure
  like PiKV Figure 1: consistent stroke widths, rounded modules, restrained
  accent colors, arrows that reveal data/control flow, and no decorative
  clutter.
- For IdleKV, a system diagram should show: turn-N compressed active cache,
  offloaded evicted-KV store in host memory, idle-window scoring using newly
  available turn text, restore of selected KV rows, and matched resumed
  active-cache budget. Avoid
  overclaiming a distributed serving system unless the paper actually evaluates
  one.
- Heatmaps should be generated as vector assets with a real colorbar,
  square-ish cells, clear axis ticks, and no fake-data warnings inside the
  plotting area. If a panel is preliminary, say so in the caption only and do
  not use it for claims. Use stars or hatching sparingly for secondary
  information such as remaining hindsight-reference headroom.
- Dense experimental figures are acceptable when each panel answers a distinct
  reviewer question. If a figure only repeats a table, move it to appendix or
  drop it.
- Figure 2 should remain the focused raw-score restore-budget frontier unless
  later data changes the story. Do not keep several near-duplicate frontier
  variants in the main paper; appendix alternatives must answer a distinct
  robustness question.
- Appendix robustness plots should reuse the main frontier grammar when
  possible: restore budget on the x-axis, exact score on the y-axis, IdleKV
  as the primary blue curve, `Gold-K` as a dashed orange reference, and
  matched / Random-$K$ / Oldest-$K$ summarized as a gray control band. Avoid
  lollipop marker piles for two-point robustness checks.
- If a robustness check has only two or three K values, do not render it as a
  fake frontier. Use a compact endpoint/interval dot plot instead: y-axis rows
  are K or budget-K settings, x-axis is exact score, gray interval is
  Random/Oldest controls, dark marker is matched no-repair, blue marker is
  IdleKV, orange hollow marker is Gold-K, and right labels show IdleKV gain
  over matched. Switch back to frontier grammar only after a full K-grid run.
- Accumulated-attention retention checks should stay appendix unless the
  full-grid run passes the main-candidate gate. The caption must explicitly
  distinguish the H2O-inspired accumulated-attention variant from a canonical
  H2O reproduction.
- A main first-stage-policy robustness figure needs at least two clean
  non-SnapKV policies or one non-SnapKV policy plus a strong reason it changes
  the interpretation of the main frontier. Do not add a policy graph just to
  increase figure count. Use a one-column gain-over-matched plot: rows/curves
  are SnapKV, accumulated-attention retention, and sink-plus-recent retention;
  blue is IdleKV/SnapKV, orange dashed is Gold-K gain, and a gray band is the
  Random-K/Oldest-K control-gain range.
- Do not promote a first-stage-policy figure on a single good endpoint.
  Sink-plus-recent retention needs a clean multi-point frontier with controls
  pinned near matched no-repair and Gold-K covering IdleKV. If it is only
  endpoint-positive or noisy, leave prior-policy variants as appendix
  robustness and future-benchmark evidence.
- Before any result enters the main text, run the Phase 13 result-rigor gate:
  full run rather than smoke, paired/shared examples, enough K-grid points,
  audited matched active-cache budget, strong full-cache reference, clean
  Random/Oldest controls, non-saturated regime, clear effect size, and an
  explicit confound check. Primary main-text claims should also report paired
  uncertainty or an equivalent paired audit. A positive smoke is only
  permission to run the locked experiment.
- For a main multi-turn claim, paired uncertainty must be decision-relevant,
  not only present in a CSV. Require positive lower bootstrap bounds for
  IdleKV over matched no-repair, IdleKV over Random-K/Oldest-K, and
  CurrentQOnly-K over StaleQOnly-K on non-initial turns. If current-query-only
  cannot be separated from stale-query-only, demote the result to appendix or
  future benchmark design.
- Before any graph enters the main text, run the Phase 13 figure-quality gate:
  real data only, graph type matched to the claim, one-column fit when
  possible, no legend/data collision, readable labels at column width, visible
  controls, scoped caption, no redundant panel, and visual style consistent
  with top KV-cache/ICML papers.
- Llama transfer checks are portability evidence, not broad model-family
  robustness. If the curve saturates, use an endpoint/interval plot or move to
  prose; do not spend main-paper space on flat saturated curves.
- Proxy latency figures should be quality-latency scatter plots or compact
  ablations with no connecting lines unless the line represents a real
  continuous algorithmic path. The caption should state that proxy scoring is a
  fixed-K probe and that exact scoring remains the quality reference.
- Do not turn the proxy result into a main table unless there is a full
  restore-budget proxy frontier or multiple proxy algorithms. Current evidence
  is strongest as a main-text latency sentence plus an appendix quality-latency
  scatter: it is `n=100` at fixed `K=96`, positive on 4Q/6Q, and explicitly
  not a proof of production-ready scoring.

Paragraphing and sentence rules:

- Abstract: one paragraph, 4-6 sentences, self-contained. Most sentences should
  be 18-35 words; allow one longer result sentence if it carries the main
  numbers.
- Introduction: use 4-5 compact paragraphs or bold inline labels. Each paragraph
  should have one job: motivation, novelty boundary, setting/gap, approach,
  contributions/scope.
- Method/Protocol: use technical noun labels ("Two-turn protocol.",
  "Matched-budget evaluation.", "Repair operation.") instead of informal
  question-like labels.
- Results: paragraph labels should name the evidence object or claim
  ("Matched-budget frontier.", "Partition robustness.", "Latency accounting.").
  Start with what the figure/table shows, then give one or two numbers.
- Discussion: use labels such as "Applicability.", "Limitations.", and
  "Pause/resume serving." rather than conversational titles.
- Reference papers use inline labels when they are technical handles, not as
  mini-sentences. ShadowKV uses labels like "Observation.", "Insights.",
  "Analysis.", "Setup.", and "Baselines."; TinyServe uses labels like "Hardware
  Execution Model." and "System Implication."; KVzip uses "Evaluation." and
  "Baseline Methods." The IdleKV paper should follow this pattern. Do not use
  labels such as "When repair should help."; use "Applicability." or
  "Applicability criteria." instead.
- Top KV-cache papers usually keep section-specific details at the level needed
  to understand the claim. Main text emphasizes the problem, novelty boundary,
  method object, strongest figure, and one or two diagnostic numbers. Exact
  split lists, exhaustive hyperparameter rationale, and failed/null controls
  belong in appendix unless a reviewer needs them to understand the main figure.
- Avoid paragraphs longer than roughly half a column unless they contain a
  necessary definition.

Vocabulary and claim rules:

- Define terms before using them as shorthand.
- Prefer precise claims over broad claims.
- Use concrete baselines and budgets in claims.
- State what is not matched when using "matched" language.
- State synthetic-task and fixed-transcript limitations where relevant.
- Avoid long method details in the abstract.
- Avoid low-signal filler and broad literature summaries.
- Every paragraph should either define the setting, support the claim, delimit
  the claim, or explain why the result matters to AdaptFM.
- Common transition phrases are acceptable but should not carry the argument:
  "To address this gap", "We instantiate", "This isolates", "We evaluate",
  "These results suggest". Do not overuse them.

## Pre-Edit Checklist

Before editing `paper/main.tex`:

1. Read this guide.
2. Check `phases/phase9_experiment_deepening/phase9_plan.md` for current
   figure/data decisions.
3. Search for internal vocabulary:
   `rg -n "phase|lift|oracle|bridge|extension|clean suite|matched footprint" paper/main.tex`
4. Decide whether the edit affects main-paper claims, appendix support, or only
   formatting.
5. After any `paper/main.tex` edit, rebuild:
   `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
6. Inspect the log for undefined references, overfull boxes, and figure/table
   placement problems.

## Optional System Diagram Backlog

- A future manually designed system diagram could show the tiered-KV lifecycle:
  strict GPU-resident retention, looser searchable off-device retention,
  colder compressed/summarized/recomputable state, and idle-window promotion
  back to the active cache. Do not add this as a generated filler figure in the
  main paper unless it is polished enough to replace or materially extend
  Figure 1.
- Keep the current claim boundary: IdleKV motivates scoring/promotion/demotion
  interfaces for tiered KV systems, but it does not make chip-level area,
  energy, or bandwidth claims.
