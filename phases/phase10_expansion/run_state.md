# Phase 10 Run State

Last updated: 2026-05-03 22:10 UTC.

This file is historical Phase 10/13 operational context, not the current
project queue and not paper-facing prose. The canonical live closure status is
`phases/phase14_critical_flaw_closure/phase14_status.md`.

Current validation snapshots are tracked in
`phases/phase14_critical_flaw_closure/phase14_status.md`.

The old `phase13_multiturn_locked` tmux job completed and was integrated into
the paper. Do not use this file to infer active GPU state.

## Active Queue

1. `phase9_heatmap_6q`
   - Status: completed; final 6Q CSV is present at
     `phases/phase9_experiment_deepening/results/phase9_phase_diagram_6q_final_n24.csv`.
   - Integration: promoted from appendix to the main Results section as
     an operating-regime figure after expert review. The figure now plots
     normalized recovery relative to benchmark-metadata Gold-K gain.

2. `phase10_specificity_smoke`
   - Status: completed; output is
     `phases/phase10_expansion/results/specificity_smoke_n1.csv`.
   - Purpose: smoke-test the highest-priority Phase 10 novelty-boundary
     panel.
   - Conditions: Matched, StaleQ-K, WrongQ-K donor, Refresh-buffered,
     IdleKV, Gold-K.
   - Setting: MQ-NIAH-4Q balanced split-query suite, `B=16384`, `K=48,96`,
     `n=1`, exact Q2 scorer, gold-span hindsight reference.
   - Output expected:
     `phases/phase10_expansion/results/specificity_smoke_n1.csv`.
   - Gate result: `K=48` separates IdleKV from stale and wrong-query
     controls but Refresh-buffered and Gold-K still have headroom; `K=96`
     is demoted because stale-query catches up.

3. `phase10_specificity_locked`
   - Status: completed.
   - Purpose: paper-grade locked specificity panel at the smoke-selected
     operating point.
   - Setting: MQ-NIAH-4Q balanced split-query suite, `B=16384`, `K=48`,
     `n=24`, exact Q2 scorer, gold-span hindsight reference.
   - Output:
     `phases/phase10_expansion/results/specificity_locked_n24_k48.csv`.
   - Gate result: promoted to main Results as a novelty-boundary figure.
     IdleKV beats matched no-repair by `+0.326` score with gain CI lower
     bound `+0.243`, beats stale and donor-query controls by `+0.299`,
     and has paired win rate `0.56`. Refresh-buffered and Gold-K both
     reach `1.000`, so the paper frames IdleKV as an incremental
     buffered-repair primitive rather than the best possible Q2-time
     full-budget reselection policy.

4. `phase10_model_transfer_smoke`
   - Status: completed.
   - Purpose: local cross-model feasibility smoke using the cached
     Qwen2.5-0.5B-Instruct model.
   - Setting: MQ-NIAH-4Q balanced split-query suite, `n=1`, budgets `{8192,16384}`,
     `K={48,96}`, exact Q2 scorer, gold-span hindsight reference,
     content-agnostic restore controls.
   - Output:
     `phases/phase10_expansion/results/model_transfer_qwen05b_smoke_n1.csv`.
   - Gate result: do not use as paper evidence. Full-cache score `A`,
     matched no-repair, IdleKV, controls, and Gold-K are all `0.000` at
     both tested budgets and both K values, so the repair comparison is
     uninterpretable for the cached Qwen2.5-0.5B-Instruct model.

5. `phase10_query_count_smokes`
   - Status: completed.
   - Purpose: preliminary breadth check for 2Q/3Q/8Q.
   - Output:
     `phases/phase10_expansion/results/query_count_smoke_n2.csv`.
   - Script now summarizes only artifacts produced by the current run, so
     stale partial smoke files cannot leak into the breadth CSV.
   - Gate result: keep as smoke evidence only. The `n=2` results show
     useful breadth signal: 2Q and 3Q recover cleanly at larger restore
     budgets, while 8Q needs a higher K before recovery is reliable.
     Main-paper query-count curves require full K-grid follow-ups.

6. `phase10_sink_recent_smoke`
   - Status: completed.
   - Purpose: non-SnapKV structural-retention smoke.
   - Conditions: A, B, matched no-repair, Random-K, Oldest-K, IdleKV, and
     Gold-K under a sink-plus-recent first-stage retention rule inspired by
     StreamingLLM.
   - Output expected:
     `phases/phase10_expansion/results/streamingllm_smoke_n1.csv` plus a
     timestamped log in `phases/phase10_expansion/results/logs/`.
   - Gate result: do not promote. The smoke is interpretable but weak:
     at `B=8192`, IdleKV does not improve over matched no-repair at
     `K=48` or `K=96`; at `B=12288` and `B=16384`, it improves only
     from `0.333` to `0.500` at `K=96`, while Gold-K reaches `1.000`.
     Random-K and Oldest-K remain at matched, so this is a small positive
     signal for structural-retention compatibility, not paper-grade
     retention-rule breadth.

7. `phase10_query_count_locked`
   - Status: completed.
   - Purpose: appendix-first locked breadth follow-up for 2Q/3Q/8Q after
     the positive smoke.
   - Setting: exact Q2 scorer, Gold-K reference, Random-K/Oldest-K
     controls, `K={48,96}`, `n=12`.
   - Tasks/budgets: 2Q at `B=8192`, 3Q at `B=14336`, 8Q at `B=18432`.
   - Output expected:
     `phases/phase10_expansion/results/query_count_locked_n12.csv`.
   - Gate: keep appendix-only unless 3Q and 8Q both show robust IdleKV
     gain over matched no-repair and content-agnostic controls stay near
     matched.
   - Result: endpoint evidence alone is not main-paper material. 3Q and
     8Q are usable appendix breadth candidates at `K=96`, but controls
     do not add a distinct main-paper story. The later full 2Q K-grid
     run enters the main raw-score frontier.

8. `phase10_quant_promo_sweep`
   - Status: completed.
   - Purpose: budget sweep for the HQQ-backed low-bit row-store
     precision-promotion branch after the tiny smoke showed that
     `K<=192` does not recover.
   - Setting: MQ-NIAH-2Q clean split, `B=4096`, `n=1`,
     `nbits={2,4,8}`, `K={96,192,512,1024,2048,4096}`,
     HQQ packed row store, exact Q2 scorer.
   - Output expected:
     `phases/phase10_expansion/results/precision_promotion_budget_sweep_4k_hqq.csv`.
   - Gate result: do not promote to the main paper. At 2-bit and 4-bit,
     low-bit storage degrades answer accuracy to zero and selective
     promotion does not recover until the all-row limit, where static,
     random, oldest, IdleKV, and Gold all match. At 8-bit, low-bit storage
     already preserves the answer, so there is no repair problem. This is
     a useful negative appendix/future-work note, not a main result.

9. `phase10_even_query_locked_n24`
   - Status: completed.
   - Purpose: endpoint breadth run to decide whether the main matched
     budget frontier should include 2Q and 8Q in addition to the existing
     Phase 7 4Q/6Q curves.
   - Setting: exact Q2 scorer, Gold-K reference, Random-K and Oldest-K
     content-agnostic controls, `K={48,96}`, `n=24`.
   - Tasks/budgets: 2Q at `B=8192`; 8Q at `B=18432`.
   - Output:
     `phases/phase10_expansion/results/query_count_even_locked_n24.csv`.
   - Gate result: endpoint-only 2Q is saturated and not sufficient by
     itself for the main figure. 8Q is strong at `K=96` and useful at
     `K=48`, so it motivates the full 8Q frontier now running as item 10.

10. `phase10_8q_full_frontier_n24`
   - Status: completed and promoted into the main paper frontier.
   - Purpose: full 8Q restore-budget frontier so the main frontier can
     use the same `K={8,16,24,32,48,64,80,96,128}` axis for 4Q, 6Q, and
     8Q rather than showing 8Q as an endpoint-only follow-up.
   - Setting: MQ-NIAH-8Q, `B=18432`, `n=24`, exact-Q scorer, Gold-K
     reference, Random-K/Oldest-K controls.
   - Output:
     `phases/phase10_expansion/results/mq_niah_8q_frontier_n24.csv`.
   - Integration target: one-column raw-score overlay with separate
     2Q/4Q/6Q/8Q IdleKV curves, faint matched no-repair traces, direct
     query-count labels, and Gold-K / Random-K / Oldest-K moved to the
     appendix milestone table.
   - Integration helper:
     `phases/phase10_expansion/scripts/finalize_8q_frontier_for_paper.sh`
     exported the finished 8Q artifact, rerendered figures, rebuilt
     `paper/main.pdf`, and removed LaTeX byproducts.
   - Promotion helper:
     `phases/phase10_expansion/scripts/recommend_frontier_promotion.py`
     is tested. It verifies enough K-grid points, meaningful IdleKV gain,
     low Random-K/Oldest-K control lift, broad curve shape, and Gold-K
     consistency before a frontier enters the main paper.
   - Paper edits after clean 2Q/8Q results: abstract/contributions,
     benchmark setup, Figure 2 caption, Results frontier paragraph, and
     appendix partition/milestone tables are updated to 2Q/4Q/6Q/8Q.

11. `phase10_2q_full_frontier_n100`
   - Status: completed and integrated into the main raw-score frontier.
   - Purpose: optional 2Q curve for the main frontier.
     2Q saturation is expected and can be useful if shown on the same
     full K grid, but endpoint-only evidence should stay out of Figure 2.
   - Setting: MQ-NIAH-2Q, `B=8192`, `n=100`, exact-Q scorer, Gold-K
     reference, Random-K/Oldest-K controls,
     `K={8,16,24,32,48,64,80,96,128}`.
   - Output:
     `phases/phase10_expansion/results/mq_niah_2q_frontier_n100.csv`.
   - Integration helper:
     `phases/phase10_expansion/scripts/finalize_2q_frontier_for_paper.sh`
     is syntax-checked. It rerenders figures, rebuilds `paper/main.pdf`,
     and removes LaTeX byproducts. The renderer now reads the completed
     2Q result CSV directly for the main raw-score frontier; set
     `IDLEKV_EXPORT_2Q_PAPER_CSVS=1` only if separate paper-facing 2Q
     CSVs are needed.
   - Gate result: include after the user requested the full
     restore-budget curve in the main Figure 2. The full curve is
     mechanically clean and has large gain,
     but the promotion helper flags it for manual review: 2Q saturates by
     `K=80`, has no content-agnostic control lift, and mostly documents
     the low-query-count edge of the task family. Figure 2 is now a
     raw-score plot rather than a gain-over-matched plot, which keeps
     that role legible without inflating a delta axis.

12. `phase10_multiturn_smoke`
   - Status: completed.
   - Purpose: first GPU smoke for rolling multi-turn relevance shifts.
   - Setting: 8Q shift-revisit schedule, `T=4`, `n=1`, exact-Q scorer,
     `K={96}` initially, conditions Full, Matched, IdleKV, Random-K,
     Oldest-K, StaleQ-K, and Gold-K.
   - Output:
     `phases/phase10_expansion/results/multiturn_smoke_summary_n1.csv`.
   - Gate result: useful but not final. IdleKV improves non-initial and
     revisit turns over matched, Random-K, and Oldest-K, but StaleQ-K
     also ties on this easy schedule. This should not be promoted until
     a harder schedule separates stale-query reuse.
   - Follow-up: `run_multiturn_hard_smoke.sh` is implemented and tested.
   - Evaluator status: the multi-turn recommendation gate now explicitly
     rejects runs where StaleQ-K closes the IdleKV non-initial gain gap.

13. `phase10_multiturn_hard_smoke`
   - Status: completed.
   - Purpose: harder rolling-revisit smoke that separates the final
     current query from the previous stale query.
   - Schedule: `(6,7) -> (0,1) -> (4,5) -> (2,3) -> (0,1)`.
   - Setting: 8Q, `T=5`, `n=1`, exact-Q scorer, `K={48,96}`,
     conditions Full, Matched, IdleKV, Random-K, Oldest-K, StaleQ-K, and
     Gold-K.
   - CPU tests: `test_multiturn.py` and `test_multiturn_runner.py` pass.
     The runner now logs target active context count, base active count,
     evicted-buffer count/bytes, and active-budget gap per row so locked
     runs can be audited for matched resumed-cache budgets.
   - Gate result: K=48 is rejected because the final revisit gain is
     zero. K=96 passes the smoke recommendation as
     `main_candidate_if_artifact_checks_pass`: IdleKV has non-initial
     gain `0.625`, revisit gain `1.000`, and win rate `0.600`; StaleQ-K
     remains close but below IdleKV on non-initial gain (`0.500`), so the
     locked run is needed before any paper claim.
   - Artifact audit: all rows have matched target active-context count
     and zero active-budget gap.
   - Locked wrapper:
     `phases/phase10_expansion/scripts/run_multiturn_hard_locked_n12.sh`
     launched in tmux as `phase10_multiturn_hard_locked_n12`.

14. `phase10_multiturn_hard_locked_n12`
   - Status: finished 2026-05-03 05:03 UTC.
   - Purpose: locked follow-up for the positive K=96 hard multi-turn
     smoke.
   - Setting: 8Q, `T=5`, `n=12`, exact-Q scorer, `K={48,96}`,
     conditions Full, Matched, IdleKV, Random-K, Oldest-K, StaleQ-K, and
     Gold-K.
   - Output:
     `phases/phase10_expansion/results/multiturn_hard_locked_summary_n12.csv`.
   - Result: positive but nuanced. At `K=96`, IdleKV scores 0.992 versus
     0.517 matched no-repair and 0.525/0.542 Random-K/Oldest-K, but
     StaleQ-K reaches 0.767. Treat as appendix-quality evidence for
     dynamic repair unless a follow-up cleanly separates stale-query reuse.

15. `phase10_accumulated_attention_smoke`
   - Status: completed; locked follow-up finished 2026-05-03 05:16 UTC.
   - Purpose: content-aware non-SnapKV retention-rule breadth check using
     the new accumulated-attention first-stage retention rule inspired by H2O.
     Paper-facing text should call this an accumulated-attention retention
     variant inspired by H2O unless the implementation is upgraded to
     canonical decoding-time H2O attention accumulation.
   - Setting: 4Q `clean_suite`, `B={14336,16384,18432}`, `n=2`,
     exact-Q scorer, Gold-K reference, Random-K/Oldest-K controls,
     `K={48,96,128}`.
   - Smoke output:
     `phases/phase10_expansion/results/h2o_compressor_smoke_n2.csv`.
   - Smoke result: directionally strong but slightly noisy. Across
     `B={14336,16384,18432}`, best IdleKV gains are 0.750, 0.833, and
     0.583; controls are mostly near matched, but the automatic gate
     rejected because one control-lift value is 0.083 against a 0.080
     cutoff.
   - Locked follow-up:
     `phases/phase10_expansion/scripts/run_h2o_compressor_locked_n12.sh`
     ran `B=16384`, `n=12`, `K={48,96}`.
   - Locked output:
     `phases/phase10_expansion/results/h2o_compressor_locked_n12.csv`.
   - Result: positive retention-rule breadth evidence. At `K=48`, IdleKV
     scores `0.514` versus `0.208` matched no-repair, `0.222` Random-K,
     and `0.208` Oldest-K. At `K=96`, IdleKV scores `0.917` versus
     `0.208` for matched no-repair and both content-agnostic controls.
     Gold-K is `1.000` at both budgets.
   - Integration: use as a compact appendix robustness figure, not a
     main-paper generality claim. The implementation is H2O-inspired because
     it approximates accumulated attention from recent post-Q1 cache rows;
     it is not a canonical decoding-time H2O reproduction.

16. `phase10_selector_variant_smoke`
   - Status: staged, not running until after the higher-priority
     multi-turn and compressor smokes unless the GPU queue opens with a
     short gap.
   - Purpose: test whether the current Q2-score burst selector leaves
     obvious recoverable headroom.
   - Implemented variants: `IdleKV-Coverage` greedily selects burst
     windows by marginal score coverage; `IdleKV-MMR` adds a diversity
     bonus between restore anchors.
   - CPU tests: selector helpers, runner condition validation, reporting,
     and syntax checks pass.
   - Setting: 4Q `clean_suite`, `B=16384`, `n=1`, exact-Q scorer,
     Gold-K reference, `K={24,48,96}` via
     `phases/phase10_expansion/scripts/run_selector_variant_smoke.sh`.
   - Gate: scale only if a variant beats current IdleKV by at least
     `0.05` at mid-K without hurting high-K by more than `0.02`.

17. `phase10_download_qwen25_3b`
   - Status: completed; tmux session closed after download.
   - Purpose: stage a viable 3B-class model-transfer candidate while the
     GPU is busy. This is not a promoted experiment; no GPU time should be
     spent until the model is local and passes a full-cache ability smoke.
   - Target: `Qwen/Qwen2.5-3B-Instruct` into
     `models/Qwen2.5-3B-Instruct`.
   - Log:
     `phases/phase10_expansion/results/logs/download_qwen25_3b_20260503T014930Z.log`.
   - Verification: local config/tokenizer load succeeds (`qwen2`, 36
     layers, 16 attention heads, 2 KV heads).
   - Next gate: run a small full-cache/matched ability smoke via
     `phases/phase10_expansion/scripts/run_model_transfer_ability_smoke.sh`
     before any matched-budget repair comparison. The ability smoke now
     uses `n=4` and conditions `A/B/B_match`.
   - Wrapper status: the full repair smoke wrapper now requires an
     ability-gate CSV by default and uses a model-derived output label, so
     a later Qwen2.5-3B smoke will not overwrite or mislabel the failed
     Qwen2.5-0.5B result.
   - Qwen2.5-3B repair smoke wrapper:
     `phases/phase10_expansion/scripts/run_model_transfer_qwen25_3b_smoke.sh`
     is staged but must not run until the 3B full-cache ability smoke
     shows the task is answerable.

18. `phase10_model_3b_ability`
   - Status: completed 2026-05-03 05:19 UTC.
   - Purpose: full-cache/matched ability gate for
     `models/Qwen2.5-3B-Instruct`, before any model-transfer repair
     comparison.
   - Wrapper:
     `phases/phase10_expansion/scripts/run_model_transfer_ability_smoke.sh`.
   - Setting: 4Q `clean_suite`, `B=16384`, `K=48`, `n=4`,
     conditions `A/B/B_match`, proxy scorer only for the ability gate.
   - Gate: queue the model-transfer repair smoke only if full-cache and
     matched rows show that the task is answerable for this model.
   - Result: gate passed. Full-cache score is `1.000`; base compressed
     and matched no-repair are `0.542`, leaving compression headroom.

19. `phase10_model_3b_repair_smoke`
   - Status: completed 2026-05-03 05:22 UTC.
   - Purpose: first repair smoke for Qwen2.5-3B after the ability gate
     passed.
   - Wrapper:
     `phases/phase10_expansion/scripts/run_model_transfer_qwen25_3b_smoke.sh`.
   - Setting: 4Q `clean_suite`, budgets `{8192,16384}`, `n=2`,
     `K={48,96}`, exact-Q scorer, Gold-K reference, Random-K/Oldest-K
     controls.
   - Gate: run a locked cross-model result only if full-cache stays high,
     IdleKV beats matched no-repair by a meaningful margin, and
     content-agnostic controls do not explain the gain.
   - Result: gate passed. Full-cache score is `1.000` for both budgets.
     At `B=8192`, matched no-repair is `0.333` and IdleKV reaches
     `0.750` at `K=48` and `1.000` at `K=96`, while Random-K/Oldest-K
     are `0.417/0.333`. At `B=16384`, matched no-repair is `0.583` and
     IdleKV reaches `0.917` at `K=48` and `1.000` at `K=96`, while
     Random-K/Oldest-K remain `0.583`.

20. `phase10_model_3b_locked_n12`
   - Status: completed 2026-05-03 05:37 UTC.
   - Purpose: locked cross-model follow-up for Qwen2.5-3B after the
     positive ability and repair smokes.
   - Wrapper:
     `phases/phase10_expansion/scripts/run_model_transfer_qwen25_3b_locked_n12.sh`.
   - Setting: 4Q `clean_suite`, budgets `{8192,16384}`, `n=12`,
     `K={48,96}`, exact-Q scorer, Gold-K reference, Random-K/Oldest-K
     controls.
   - Result: positive same-family size-transfer evidence, not true
     model-family diversity. Full-cache score is `1.000` at both
     budgets. At `B=8192`, IdleKV reaches `1.000` at `K=96` versus
     `0.278` matched no-repair and `0.292/0.264` Random-K/Oldest-K. At
     `B=16384`, IdleKV reaches `1.000` at `K=96` versus `0.611` matched
     no-repair and `0.625/0.611` Random-K/Oldest-K.
   - Integration: appendix-only, framed as cautious portability evidence
     within the Qwen family. Do not call this a model-diversity result.

21. `phase10_download_mistral7b`
   - Status: completed 2026-05-03 05:45 UTC.
   - Purpose: true model-family diversity candidate after the Qwen2.5-3B
     run was identified as same-family size transfer.
   - Target: `mistralai/Mistral-7B-Instruct-v0.3` into
     `models/Mistral-7B-Instruct-v0.3`.
   - Rationale: Mistral is a different model family from Qwen, has a
     7B-class instruct checkpoint, and its documented 32k context window
     matches the current benchmark scale.
   - Result: local config, tokenizer, and three safetensor shards are
     present in `models/Mistral-7B-Instruct-v0.3`; no incomplete download
     files remain.

22. `phase10_mistral7b_ability`
   - Status: aborted 2026-05-03 05:47 UTC after user preference changed
     to Llama.
   - Purpose: full-cache/matched ability gate for the first true
     model-family diversity candidate.
   - Wrapper:
     `phases/phase10_expansion/scripts/run_model_transfer_ability_smoke.sh`.
   - Setting: 4Q `clean_suite`, `B=16384`, `K=48`, `n=4`,
     conditions `A/B/B_match`, proxy scorer only for the ability gate.
   - Result: stopped before completion; do not use as evidence.

23. `phase10_download_llama31_8b`
   - Status: completed 2026-05-03 05:50 UTC.
   - Purpose: true model-family diversity candidate preferred by the
     user over Mistral.
   - Target: `meta-llama/Llama-3.1-8B-Instruct` into
     `models/Llama-3.1-8B-Instruct`.
   - Rationale: Llama is a different model family and 8B is the closest
     size match to the primary Qwen2.5-7B setup.
   - Result: local config, tokenizer, index, and four safetensor shards
     are present in `models/Llama-3.1-8B-Instruct`; no incomplete
     download files remain.

24. `phase10_llama31_8b_ability`
   - Status: completed 2026-05-03 05:54 UTC.
   - Purpose: full-cache/matched ability gate before treating Llama as
     cross-family portability evidence.
   - Wrapper:
     `phases/phase10_expansion/scripts/run_model_transfer_ability_smoke.sh`.
   - Setting: 4Q `clean_suite`, `B=16384`, `K=48`, `n=4`,
     conditions `A/B/B_match`, proxy scorer only for the ability gate.
   - Result: passed. Full-cache `A=1.000`; matched no-repair
     `B_match=0.500`, leaving clear repair headroom.
   - Output:
     `phases/phase10_expansion/results/model_transfer_ability_llama_3_1_8b_instruct__n4.csv`.

25. `phase10_llama31_8b_repair_smoke`
   - Status: completed 2026-05-03 06:02 UTC.
   - Purpose: exact-Q controlled repair smoke for Llama cross-family
     portability.
   - Wrapper:
     `phases/phase10_expansion/scripts/run_model_transfer_smoke.sh`.
   - Setting: 4Q `clean_suite`, budgets `{8192,16384}`, `K={48,96}`,
     `n=4`, conditions `A/B/B_match/Random-K/Oldest-K/IdleKV/Oracle-K`,
     exact-Q scorer, gold-span hindsight reference.
   - Output:
     `phases/phase10_expansion/results/model_transfer_llama_3_1_8b_instruct__smoke_n4.csv`.
   - Result: passed appendix-candidate gate at both budgets. At `B=8192`,
     Full and Gold-K are `1.000`, matched no-repair is `0.000`, IdleKV is
     `1.000`, and Random-K/Oldest-K remain at or near zero. At `B=16384`,
     Full and Gold-K are `1.000`, matched no-repair is `0.500`, IdleKV is
     `1.000`, and Random-K/Oldest-K stay at `0.500/0.417`. The gate script
     reports `all_budgets_appendix_candidate=True`.

26. `phase10_llama31_8b_locked_n12`
   - Status: completed 2026-05-03 06:26 UTC and integrated into the
     appendix robustness figure.
   - Purpose: locked cross-family portability check after the Llama
     ability gate and repair smoke both passed.
   - Wrapper:
     `phases/phase10_expansion/scripts/run_model_transfer_locked_n12.sh`.
   - Setting: 4Q `clean_suite`, budgets `{8192,16384}`, `K={48,96}`,
     `n=12`, conditions `A/B/B_match/Random-K/Oldest-K/IdleKV/Oracle-K`,
     exact-Q scorer, gold-span hindsight reference.
   - Output:
     `phases/phase10_expansion/results/model_transfer_llama_3_1_8b_instruct__locked_n12.csv`.
   - Gate result: passed. At `B=8192`, IdleKV and Gold-K both reach
     `1.000` for `K=48` and `K=96`, while matched no-repair is `0.028`,
     Random-K is at most `0.042`, and Oldest-K is `0.014`. At `B=16384`,
     IdleKV and Gold-K remain `1.000`, while matched no-repair,
     Random-K, and Oldest-K are `0.500`, `0.500`, and `0.472`. The
     recommendation gate reports `all_budgets_appendix_candidate=True`.
   - Paper integration: `paper/scripts/render_paper_figures.py` now uses
     this Llama CSV before the Qwen2.5-3B fallback; the appendix caption
     and limitation text frame it as same-protocol cross-family
     portability evidence, not broad model-family robustness.

27. `phase13_multiturn_locked`
   - Status: completed 2026-05-03 09:38 UTC and integrated into the main
     paper after paired uncertainty audit.
   - Purpose: test repeated repair across controlled relevance shifts and
     revisits, closer to dynamic agent-style workflows than a single Q2
     handoff.
   - Wrapper:
     `phases/phase13_iteration_framework/scripts/run_multiturn_hard_locked.sh`
     via tmux session `phase13_multiturn_locked`.
   - Setting: MQ-NIAH-8Q hard revisit schedule, five turns, `K=80`,
     `B_base=18432`, `n=24`, exact-Q scoring, conditions
     `Full/Matched/IdleKV/CurrentQOnly-K/Random-K/Oldest-K/StaleQ-K/
     StaleQOnly-K/Gold-K`.
   - Output:
     `phases/phase13_iteration_framework/results/multiturn_hard_locked_rows_n24_k80.csv`,
     `..._summary_n24_k80.csv`, `..._raw.json`, and
     `..._uncertainty_n24_k80.csv`.
   - Result: passed the main gate. IdleKV non-initial gain is `0.542`
     with paired interval `[0.458,0.620]`; revisit-turn gain is `0.938`
     with interval `[0.875,1.000]`. Random-K and Oldest-K non-initial
     gains are `0.010` and `0.021`. CurrentQOnly-K beats StaleQOnly-K on
     non-initial turns by `0.307` with interval `[0.240,0.370]`, although
     stale-query-only still gains `0.234`, so the paper frames the result
     as a controlled dynamic-workflow diagnostic rather than end-to-end
     agent validation.

## Current Phase 10 Priority

Use `phase10_high_signal_map.md` as the compact source of truth for live
high-signal branches. Do not discard multi-turn, retention-rule breadth,
model transfer, selector variants, or dynamic precision repair merely
because they are not immediately main-paper ready; each branch should
fail or pass its explicit smoke gate.

The adversarial critiques agreed that specificity/novelty evidence is
more important than extra query-count breadth. The current main-paper
candidate package is therefore:

1. Method schematic.
2. Matched-budget frontier: 2Q/4Q/6Q/8Q are locked in the main raw-score
   restore-budget plot. The completed full 2Q K-grid run is included
   with neutral panel labeling.
3. Specificity contrast, now promoted after the locked run. It plots
   score gain over matched no-repair with confidence intervals plus
   paired win/tie/loss rates. The bounded comparator is named
   Refresh-buffered in prose and shortened to Refresh in the figure.
4. Multi-turn revisit figure, promoted after the locked `n=24`, `K=80`
   run passed paired uncertainty and stale-query separation gates.
5. First-stage retention-rule breadth figure, promoted after the
   sink-plus-recent full grid passed and terminology was scoped to
   mechanism-level retention variants.

The operating-regime heatmap is now appendix calibration context. Query-count
breadth also remains appendix unless a later final run changes the main
frontier story.

## Final-Phase Closure Checklist

1. Phase 13 locked multi-turn follow-up: completed, passed, and integrated.
2. Llama locked portability check: completed, passed the appendix gate,
   and integrated.
3. Figure rendering and PDF rebuild: completed after Llama integration and
   after the latest terminology edit.
   The LaTeX log has no undefined references or overfull boxes; it has
   the known empty-anchor warning and underfull vboxes only.
4. Rerun focused tests for any new edited code plus the repo-wide pytest
   suite if code changes again after the current clean test run.
5. Remove LaTeX and Python byproducts before final handoff or commit.
6. Do not launch additional full GPU suites unless the written Phase 13 gate
   identifies a paper-critical hole.

The strict-cap streaming spill coverage table has been demoted to the
appendix. It remains useful evidence that CPU spill can retain future
relevant tokens, but the current `n=4` coverage-only diagnostic is not
strong enough for the main Results section without an answer-quality
repair run and matched restore controls.

## Work Completed While Specificity Runs

- Added and tested a multi-turn score-trajectory summarizer for paired
  gain, non-initial-turn gain, revisit-turn gain, and matched win rate.
  This will be the gate for any future rolling/revisit smoke figure.
- Extended the specificity recommendation printout with the IdleKV gain
  CI lower bound and paired win rate so the locked run can be judged
  immediately when its CSV lands.
- Integrated the locked specificity result into the main paper and
  rebuilt `paper/main.pdf`.
- Redesigned the specificity figure from a single dot plot into a
  two-panel gain plus paired-outcome plot so the figure itself shows both
  effect size and consistency.

## Quantization Branch State

- Added and tested low-bit-rowstore precision-promotion utilities:
  integer row codes, per-row scales, materialization back to model dtype,
  high-precision row promotion, and byte accounting.
- Upgraded the precision-promotion smoke to support an HQQ-backed packed
  row store. The default smoke path now uses HQQ quantization metadata,
  not the older symmetric row quantizer.
- Added `run_precision_promotion_smoke.py` for the quality/byte
  diagnostic. This is not a low-bit attention-kernel claim.
- Added `recommend_precision_promotion.py` and gate tests. Promotion
  requires low-bit degradation, IdleKV-Precision beating static/random/
  oldest precision controls, Gold-Precision consistency, and explicit
  byte accounting.
- Installed `optimum-quanto==0.2.7`, `hqq==0.2.8.post1`, and the
  matching CUDA 12.8 nvcc wheel dependency with `uv`.
- `smoke_quantized_cache_generation.py` passes with Transformers
  `QuantizedCache(backend="hqq")` on Qwen2.5-7B-Instruct. The `quanto`
  path gets past missing `ninja` but still needs a full CUDA toolkit
  layout for its extension builder, so use HQQ as the real-cache smoke
  backend for now.
- Tiny HQQ row-store promotion smokes show that `LowBit-all` degrades
  to 0.0 and `K={96,192}` does not recover, while promoting all 4096
  context rows returns to 1.0. This validates materialization and turns
  the open question into a budget-transition search.
- Local source inspection suggests Transformers `QuantizedCache` is good
  for a real low-bit baseline but does not obviously expose arbitrary
  row-level high-precision promotion. If the tiny smoke passes, the next
  decision is whether to build a custom cache class or keep precision
  promotion as a row-store diagnostic.
