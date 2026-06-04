# Phase 10 High-Signal Experiment Map

Last updated: 2026-05-03 07:18:59 UTC.

This is the compact Phase 10 control sheet. Do not discard a branch just
because it is not immediately ready for the main paper. Keep each branch
live until its smoke or locked run fails a clear gate.

## Main-Paper Candidates

1. **Matched-budget repair frontier**
   - Question: does idle-window repair recover future-turn answers under
     the same resumed active KV budget?
   - Current evidence: robust 4Q and 6Q locked runs are strong, the full
     8Q `n=24` frontier passed the promotion gate, and the full 2Q
     `n=100` K-grid run is included neutrally as the low-query-count
     edge case.
   - Figure: one-column raw-score overlay over restore budget. Solid
     lines show RepairKV for 2Q/4Q/6Q/8Q; faint dotted lines show matched
     no-repair. Direct right-side labels report query count and the
     `K=96` score difference. Gold-K and Random-K/Oldest-K stay out of
     the main overlay and are summarized in prose/appendix views.
   - Current decision: keep this as the main experimental figure. Do not
     split it back into four small panels unless the overlay becomes
     unreadable after new data.

2. **Specificity and novelty-boundary contrast**
   - Question: does the gain require newly revealed next-turn information,
     or is any buffer reinsertion enough?
   - Current evidence: locked `n=24`, `K=48` specificity run is positive.
   - Figure: score gain over matched no-repair plus paired win/tie/loss
     rates for StaleQ-K, donor-query repair, RepairKV, Refresh-buffered,
     and Gold-K.
   - Promotion gate: RepairKV must beat stale/donor controls; Refresh-buffered
     must be described as a bounded Q2-time buffered reselection comparator,
     not a systems-fair full recompute baseline.

3. **Multi-turn relevance-shift trajectory**
   - Question: can repair repeatedly adapt cache state as relevance shifts
     and returns, closer to agent workflows than a single Q2 handoff?
   - Current evidence: the harder `T=5` locked run is positive versus
     matched, Random-K, and Oldest-K but not clean enough for the main
     paper because StaleQ-K closes part of the gap. At `K=96`, RepairKV
     scores `0.992` versus `0.517` matched no-repair and
     `0.525/0.542` Random-K/Oldest-K, while StaleQ-K reaches `0.767`.
   - Figure: one-column turn trajectory over turns `1..T`, with score gain
     over matched no-repair on the y-axis. Plot RepairKV, StaleQ-K,
     Random-K/Oldest-K control band, and Gold-K at the selected K. If both
     `K=48` and `K=96` are useful, use two tiny panels rather than one
     overplotted axis. Put cache-state heatmaps in appendix unless they
     replace substantial prose.
   - Placement: appendix-only unless a follow-up schedule separates
     current-query repair from stale-query reuse more cleanly. Do not call
     this agent performance; call it a controlled multi-turn
     relevance-shift diagnostic.

4. **Operating-regime map**
   - Question: is the effect confined to one cherry-picked budget?
   - Current evidence: 6Q final operating-regime heatmap is useful but
     visually and semantically secondary to specificity and multi-turn.
   - Figure: one-column heatmap or small multiple showing fraction of
     Gold-K headroom recovered by RepairKV over base budget and restore budget.
   - Placement gate: main only if there is room after frontier, specificity,
     and a positive multi-turn figure; otherwise appendix.

## Appendix-Or-Main Robustness Candidates

5. **Compressor-policy breadth**
   - Question: is repair a SnapKV artifact?
   - Current evidence: StreamingLLM smoke is weak but interpretable. The
     H2O-inspired accumulated-attention branch has passed a locked `n=12`
     check at `B=16384`, with RepairKV scoring `0.514` at `K=48` and
     `0.917` at `K=96` versus `0.208` matched no-repair.
   - Figure: compact appendix robustness plot only. The implementation
     is H2O-inspired, not a canonical H2O reproduction, so the paper should
     use it to answer "not only SnapKV?" without claiming broad
     compressor generality.

6. **Model transfer**
   - Question: is the effect tied to one checkpoint?
   - Current evidence: Qwen2.5-0.5B remains unusable because full-cache
     accuracy is zero. Qwen2.5-3B-Instruct passed the ability gate,
     repair smoke, and locked `n=12` follow-up.
   - Locked result: at `B=8192`, RepairKV reaches `1.000` at `K=96` versus
     `0.278` matched no-repair and `0.292/0.264` Random-K/Oldest-K. At
     `B=16384`, RepairKV reaches `1.000` at `K=96` versus `0.611` matched
     no-repair and `0.625/0.611` Random-K/Oldest-K. Full-cache score is
     `1.000` and Gold-K covers RepairKV at both budgets.
   - Decision: integrate as appendix portability evidence only. Keep the
     main claim at "one primary model"; this is a positive cross-model
     size-transfer check within the Qwen family, not a broad transfer
     study.
   - True diversity follow-up: `meta-llama/Llama-3.1-8B-Instruct` passed
     the ability gate, repair smoke, and locked `n=12` follow-up. At
     `B=8192`, RepairKV reaches `1.000` versus `0.028` matched no-repair
     and near-zero content-agnostic controls. At `B=16384`, RepairKV reaches
     `1.000` versus `0.500` matched no-repair, with Random-K and Oldest-K
     tracking matched. Decision: include as an appendix cross-family
     portability check, not broad model-family robustness.

7. **Repair selector algorithms**
   - Question: is the current selector leaving obvious Gold-K headroom?
   - Implemented variants: coverage-aware burst packing and MMR-style
     diverse-anchor repair. Both are runner conditions for smoke testing,
     not replacements for the main method unless they clearly win.
   - CPU status: synthetic tests cover overlapping bursts, deterministic
     backfill, diverse anchors, and fixed-budget behavior.
   - Smoke: 4Q, `B=16384`, `K={24,48,96}`, `n=1` first via
     `run_selector_variant_smoke.sh`; scale to `n=4` only if one variant
     beats current RepairKV without adding confusing clutter.
   - Gate: a new selector beats current RepairKV by at least `0.05` at mid-K
     without hurting high-K by more than `0.02`; otherwise keep the method
     simple and do not add ablation clutter.
   - Current priority: not a final-phase blocker. Run only after the
     Llama locked portability check completes, and only if the paper
     still needs an algorithmic-headroom ablation; otherwise keep the
     current method simple and avoid ablation clutter.

## Exploratory But Still Live

8. **Dynamic precision / quantized-cache repair**
   - Question: can idle-window repair revise a quantized or mixed-precision
     KV state after relevance changes?
   - Current result: the first HQQ row-store precision-promotion sweep is
     negative. At 2-bit and 4-bit, low-bit storage is too destructive for
     selective row promotion; at 8-bit, there is no repair gap.
   - Why still live: prior work covers low-bit and mixed-precision KV
     heavily, but post-compression, next-turn, idle-window precision repair
     is a narrower systems question.
   - Better design: page- or channel-aware precision promotion, byte-matched
     against MiKV/MixKVQ-style static/query-aware mixed precision, with a
     real `QuantizedCache` baseline where possible.
   - Smoke gate: low-bit baseline must degrade, Gold-precision must show
     selective recoverability, and RepairKV-Precision must beat static/random/
     oldest precision controls at the same active byte budget.
   - Placement: future-work or appendix unless the redesigned smoke is very
     clean and claims are explicitly quality/byte, not latency.

9. **Benchmark difficulty axes**
   - Question: what benchmark family makes dynamic-cache repair useful
     without collapsing into either an easy saturation case or an impossible
     over-compressed case?
   - Current evidence: 2Q is useful as the easy boundary because it saturates
     quickly; 4Q/6Q carry the clean main effect; 8Q is valuable because it
     shows how easily eviction errors accumulate as future-turn constraints
     increase.
   - Future benchmark design: vary query count, turn timing, constraint
     conflict, offloaded-store size, and resume budget independently. Report
     full K-grid frontiers instead of single operating points so reviewers can
     see whether repair improves the transition region or only the endpoints.
   - Placement: future-work paragraph now; full benchmark suite later.

10. **Runtime/proxy path**
   - Question: can the exact scorer be approximated cheaply enough for an
     idle-time system?
   - Current status: useful only if framed as a heuristic path, not as proof
     of production latency.
   - Gate: proxy/exact gain ratio should track exact-Q across the same
     operating points; otherwise keep runtime/proxy in appendix.

## Preferred Execution Order

1. Keep the 2Q/4Q/6Q/8Q full K-grid raw-score frontier as the default
   main plot. The current main-paper design uses one overlay axis:
   solid RepairKV curves, faint matched no-repair traces, and direct
   query-count labels with `K=96` repair differences.
2. Keep Random-K, Oldest-K, and Gold-K out of the main overlay to avoid
   spaghetti; report them in the Results prose and appendix milestone
   table unless a reviewer specifically asks for inline controls.
3. Prefer the Llama-3.1-8B-Instruct locked result in the appendix as the
   cautious cross-family portability check; keep the Qwen2.5-3B locked
   result as fallback same-family evidence only.
4. Keep the H2O-inspired compressor check in the appendix unless a reviewer
   explicitly asks for broader compressor coverage in the main text.
5. Keep multi-turn as appendix-quality unless a cleaner follow-up
   separates RepairKV from StaleQ-K.
6. Run selector variant smoke only if the higher-priority portability
   branch leaves GPU room.
7. Revisit dynamic precision as Phase 10b with a redesigned page/channel
   formulation, not the failed row-only HQQ sweep.
8. Treat improved dynamic-cache benchmarks as a first-class future-work item:
   2Q/8Q are not just extra curves, they define easy and high-constraint
   edges that future suites should cover deliberately.
