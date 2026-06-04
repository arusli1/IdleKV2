# Phase 15 Status

Last updated: 2026-05-04 22:39 UTC.

## Current State

Phase 15 is complete for this paper pass. No Phase 15 tmux job is active.
The bounded `phase15_filegated_full` follow-up finished on the frozen v13
manifest with two locality-aware diagnostic selectors,
`FileGatedRepairKV-K` and `LexicalAnchor-K`, at `K={96,192}`. It did not build a
new manifest or change the task distribution.

The first strict RepoDelta-Edge ability smoke,
`phase15_ability_v10_quick_n10_k96`, the improved redacted-unique ability smoke
`phase15_ability_v11_redacted_unique_n24_k96`, and the fresh v12 pilot
`phase15_ability_v12_fresh_pilot_n36_k192` all failed the locked promotion
gate, so none is paper evidence.
After expert audit, the original RepoDelta line/path lookup is no longer the
paper-facing plan. It is now a smoke/fallback scaffold.

The converged paper-facing candidate is **RepoDelta-Edge**: a controlled
real-repository relevance-shift diagnostic over pinned Python repos, using a
single static edge such as `anchor function + callsite -> leaf callee
identifier`. The main repair signal must be **event-only**: repair sees the
pre-turn-2 tool/callsite cue, restores K rows from the warm evicted store, and
then all conditions decode the same final `event + Q2` prompt.

The v13 ability artifact was promising but not paper evidence by itself:
`A=0.875`, `B=0.167`, `B_match=0.188`, and `A-B_match=0.688`, but the strict
gate fails because `CueOnly=1/48` and some rows retain answer tokens in
`B/B_match`. Strict repair-eligible rows exist (`16` rows from `9` repos with
`Q1=1`; `27` rows from `11` repos without the Q1 filter), but selected subsets
remain secondary/appendix only.

The v13 whole-manifest repair artifact is strong against deployable and
content-agnostic controls but fails the strict main-promotion gate because the
label-assisted locality reference is stronger:

- `K=96`: RepairKV `0.5625`, matched `0.1875`, random `0.2083`, oldest/stale/
  wrong-event `0.1667`, ToolFile `0.2083`, AnchorWindow `0.8958`.
- `K=192`: RepairKV `0.7292`, matched `0.1875`, random/oldest/stale/wrong-event
  `0.1667`, ToolFile `0.1875`, AnchorWindow `0.8958`.
- 2000-draw paired repo-bootstrap lower bounds are positive against matched,
  random, oldest, stale, wrong-event, and ToolFile at `K=192`.
- Sensitivity checks remain positive after excluding cue-only hits and
  answer-retention rows (`+0.553` and `+0.629` gain over matched at `K=192`).
- All `96` result rows have `phase15_manifest_audit.passed=true`, and each K
  slice has zero duplicate example rows.

The bounded locality-aware follow-up passed the explicit paper-integration
gate:

- `K=96`: FileGatedRepairKV `0.7292`, RepairKV `0.5625`, ToolFile `0.2083`,
  AnchorWindow `0.8958`; LexicalAnchor `0.2708`.
- `K=192`: FileGatedRepairKV `0.8333`, RepairKV `0.7292`, ToolFile `0.1875`,
  AnchorWindow `0.8958`; LexicalAnchor `0.3958`.
- FileGated improves over RepairKV by `+0.1042` at `K=192`, beats ToolFile by
  `+0.6458`, and remains below the label-assisted AnchorWindow reference by
  `-0.0625`.
- The event cue contains the target file path for all FileGated rows, all
  FileGated rows are budget-matched, and sensitivity checks remain
  nonnegative after excluding cue-only hits or answer-retention rows.
- LexicalAnchor has zero answer-leak rows but is answer-sanitized: it may use
  the hidden gold identifier only to remove answer-like cue terms before
  selection, so it is appendix diagnostic evidence rather than a deployable
  selector.

Decision: **preliminary main-text evidence plus appendix details**, not a
headline validation figure or broad real-code benchmark claim. The paper gives
a short Results paragraph, adds one cautious FileGated sentence, and reports
the full diagnostic in Appendix Figure `fig:app-real-repo-diagnostic`.

## Expert Audit Summary

Critical flaws found and resolved into the plan:

- Plain EventLoc/path-line lookup is too close to passkey retrieval over real
  code. It is useful for ability smoke, not the headline main result.
- Current Phase 6 scoring uses the full Q2 prompt. A main Phase 15 tool-event
  claim requires a separate `repair_signal_mode=event_only` path.
- Q2 must not restate path or line. The scaffold now asks about the reported
  failure location rather than repeating `path + line`.
- Frozen manifests are required before GPU. On-the-fly repo sampling is not
  publishable evidence.
- Strict identifier scoring is required. Existing substring/case-insensitive
  scoring is not valid for code identifiers.
- Token-level audits, not raw character audits, must reject tail leakage,
  duplicate answers, overlength prompts, and Q2 spans already preserved by the
  matched base cache.
- Stale/wrong cue controls are mandatory for causal specificity.
- Related-work framing must stay narrow: this is not a repository coding
  benchmark, SWE-bench slice, BFCL task, or scheduler/JCT evaluation.
- A broad creative-eval sweep did not find a stronger primary than
  RepoDelta-Edge. The best challenger is TestLog->Source, but it is riskier;
  the best non-repo fallback is DocDelta-Anchor/KubeDelta-Flag; the best
  complementary systems lane is TraceSched-Repair.

## Prior Branch Audit

- Main paper evidence is currently finalized for the MQ-NIAH frontier,
  specificity, multi-turn, retention-rule sensitivity, and runtime-capacity
  figures.
- Appendix/prose evidence is finalized for controlled proxy scoring, Llama
  portability, SpanRef-K diagnostics, and supplementary mechanism/endpoint
  diagnostics.
- Positive but not promoted: Coverage selector. It shows algorithmic selection
  headroom but lacks a dedicated full K-grid paper figure.
- Boundary evidence: Refresh-buffered shows stronger full-budget reselection can
  outperform incremental RepairKV. It is a method-boundary comparator, not a
  deployable baseline or theoretical optimum.
- Negative/deferred: quantized row-store promotion, Qwen2.5-0.5B transfer,
  exact named eviction reproductions, trace-scheduled idle-window benchmark,
  off-device retention policy, and candidate-restricted best-recovery
  references.

## Completed In This Pass

- Added Phase 15 plan/status files.
- Updated README to show Phase 15 as the active open question.
- Fixed RepoDelta file cards to include visible line numbers.
- Changed Q2 wording so it no longer restates the exact path and line after the
  tool event.
- Added/updated RepoDelta unit tests for line visibility and Q2 wording.
- Added the initial Phase 15 CPU-gate package:
  - `src/edge.py`: Python AST extraction for one-hop callsite-to-callee Edge
    candidates.
  - `src/scorer.py`: strict first-line identifier scoring.
  - `src/manifest.py`: RepoSource records, audited manifest rows, event-only
    cue IDs, token/span/leakage audits, and stable manifest hashes.
  - `src/protocol.py`: frozen protocol record and protocol hashing.
  - `src/bootstrap.py`: paired repo-cluster bootstrap utility.
  - `src/runner.py`: event-only repair signal object, wrong-event metadata
    helper, and ToolFile-K position helper.
  - `scripts/build_phase15_manifest.py`: manifest-builder CLI that rejects
    unpinned or self-repo sources by default.
- Added `phase15_protocol.json` and `repo_registry.example.json`.
- Added a non-invasive Phase 6 hook so `_run_one_split` can accept separate
  `repair_question_ids` and `stale_question_ids`; existing MQ-NIAH behavior
  still defaults to full-Q2 scoring.
- Staged 12 pinned repositories drawn from the SWE-bench Verified repository
  pool outside this repo and recorded commit SHA, license, and archive SHA256
  in a local dev registry.
- Built multiple SWE-bench-pool RepoDelta-Edge dev manifests. Early versions
  were useful for debugging but invalid for paper evidence because many rows
  were below the base active-cache budget and therefore had little or no
  evicted context to repair.
- Added a manifest-consuming Phase 15 GPU wrapper that loads the model once,
  rebuilds frozen Q1/Q2 views from JSONL, passes event-only repair IDs into the
  Phase 6 path, and overwrites substring scores with strict identifier scores.
- Dry-ran the wrapper over 24 rows with tokenizer-only preparation.
- Ran focused Phase 15 tests successfully after each code change.
- Ran one GPU ability smoke on the earlier v5 manifest: `n=5`, `K=96`,
  conditions `A/B/B_match`. It failed strict full-cache ability
  (`A=0.20`) and is discarded as non-evidence.
- Added stricter CPU gates after audit: source checkout verification, tokenizer
  replay fingerprints, minimum context length above the repair budget,
  project-local target filtering, and budget-matched ToolFile backfill.
- Built strict manifest `repodelta_edge_swebench_manifest_probe_v10_quick`:
  12 rows from 12 SWE-bench-pool repos, 25.8K-32.5K rendered tokens, no warning
  flags, project-local target filtering, and manifest hash
  `1457c27b043f7e8c8bad0795ff9e2f9cab4f9433138e69544e80a8f96aeba59d`.
- Dry-ran the exact v10 quick manifest through the Phase 15 wrapper.
- Ran v10 ability smoke at `K=96`. It had real eviction and cue-only leakage was
  clean, but failed scientifically: `A=0.50`, `B=0.50`, `B_match=0.50`,
  `A-B_match=0.00`, `CueOnly=0.00`. It did not create a recoverable gap.
- Added answer-token retention diagnostics to Phase 15 artifacts:
  `b_answer_token_overlap_fraction` and
  `b_match_answer_token_overlap_fraction`. These are now separate from broad
  line-span overlap because a row is not a valid repair target if the active
  cache still contains the answer tokens.
- Added a deterministic discovery selector that can derive a follow-up manifest
  only from rows satisfying the predeclared ability/gap gate:
  `A=1`, `B_match=0`, `CueOnly=0`, real eviction, and zero B/B_match
  answer-token retention. This is a diagnostic filter, not final locked
  evidence.
- Built and dry-ran
  `repodelta_edge_swebench_manifest_probe_v11_redacted_unique`: 24 rows from
  12 SWE-bench-pool repos, unique answer occurrences, answer-redacted event
  cues, 24.8K-32.5K rendered tokens, and manifest hash
  `ef7bc35d16806edbe11f9c98da19cec6649152bbd34461ef056af0310807b227`.
- Ran v11 ability smoke at `K=96`. It improved the recoverable-gap signal:
  `A=0.708`, `B=0.083`, `B_match=0.125`, `A-B_match=0.583`,
  `CueOnly=0.000`. It still failed the full-cache locked-run gate and had
  answer-token retention in some rows.
- Deterministically selected discovery rows from v11:
  - 11 rows satisfied `A=1`, `B_match=0`, `CueOnly=0`, real eviction, and zero
    B/B_match answer-token retention.
  - 6 of those also had `Q1=1`; this stricter Q1-clean manifest has hash
    `aaf3149ca812dd9fc6bc677f7daa96f925fa76d9780fa87e552ce902abd74cba`.
- Started a repair smoke on the Q1-clean discovery subset:
  `phase15_repair_v11_selected_gap_q1clean_n6_k48_96`, with all required
  controls. This is a design diagnostic, not final evidence.
- Q1-clean repair smoke finished with a small positive signal:
  - `K=48`: RepairKV `1/6`, all controls `0/6`.
  - `K=96`: RepairKV `2/6`, all controls `0/6`.
  ToolFile-K recovered `0/6`, so the positive cases are not explained by a
  simple file-local restore heuristic. The run is still not paper evidence
  because it is outcome-selected and has only four repos.
- Expert audit tightened the confirmatory rule: outcome-selected rows are
  discovery only. If the repair smoke is positive, the next valid move is a
  fresh frozen manifest with pre-outcome eligibility rules and repo balance, not
  promotion of the selected subset.
- The diagnostic selector now supports deterministic total-row and per-repo
  caps and records `repo_balance_truncated` reasons. This is for smoke hygiene;
  it still does not make selected rows confirmatory.
- Started a broader discovery repair smoke on all 11 v11 ability-gap rows
  without requiring `Q1=1`: `phase15_repair_v11_selected_gap_n11_k48_96`.
  This tests whether the mechanism signal persists across more examples/repos.
- The 11-row discovery repair smoke finished:
  - `K=48`: RepairKV `1/11`, Random `1/11`, all other controls `0/11`.
  - `K=96`: RepairKV `2/11`, all controls `0/11`.
  The result is directionally useful but weak; it supports more diagnostic
  exploration, not a locked main-paper claim.
- High-K diagnostic on the same 11 discovery rows finished:
  - `K=128`: RepairKV `3/11`, all controls `0/11`.
  - `K=192`: RepairKV `5/11`, Random `1/11`, all other controls including
    ToolFile `0/11`.
  This suggests the Edge signal is budget-limited and justifies one fresh
  preregistered pilot with `K=192`; it still does not justify promoting
  discovery-selected rows.
- Fresh v12 pilot finished on 36 rows from 12 SWE-bench-pool repositories:
  - `A=0.722`, `B_match=0.194`, `A-B_match=0.528`,
    `CueOnly=0.083`, `Q1=0.583`.
  - The run failed strict gates: full-cache ability below `0.80`, three
    cue-only hits, and nonzero answer-token retention in some B/B_match rows.
  - Strict repair eligibility yielded only 9 selected rows after per-repo
    truncation, from 6 repos, so repair is not justified from v12.
  - Row audit showed `callsite_leaf_callee` is the only promising Edge family
    (`A=0.833`, gap `0.625`, cue-only `0.042`), while exception rows are noisy
    (`A=0.500`, gap `0.333`, cue-only `0.167`).
- Implemented the bounded v13 generator changes:
  - Q1 now uses an answer-redacted declaration cue rather than a brittle line
    lookup.
  - Q2 candidate priority now tries callsite edges before exception reserves.
  - Source discovery now excludes tests, docs, examples, externals, and vendored
    trees for both Q1 and Q2 candidates.
  - The default Phase 15 protocol now matches the bounded real-repo diagnostic:
    `K={96,192}` with `primary_k=192`.
  - Added unit tests for callsite priority, Q1 redaction, and the repair-gate
    audit helper. Focused tests passed; the current focused suite is
    `56 passed`.
- Started v13 manifest build in tmux:
  `phase15_build_v13`, output
  `repodelta_edge_swebench_manifest_probe_v13_callsite_q1redacted.jsonl`.
  It is callsite-only, unique-answer, redacted-cue, fresh-seed, 4 examples per
  repo, and uses `K*=192` as the real-repo diagnostic budget with `K=96` as the
  adjacent sanity check.
- v13 build finished:
  - 48 rows, 12 repos, exactly 4 rows per repo.
  - All rows are `callsite_leaf_callee`.
  - Zero manifest flags and zero warnings.
  - Unique answer boundary occurrence for all rows.
  - Rendered context length range: 24.9K-32.6K tokens.
  - Q2 depth fraction range: 0.508-0.842.
  - Manifest hash:
    `754b2c16a32bde830b25ee2171ba4b3c9723c3097d3ccb36cfc1fdf11983fbaa`.
  - Protocol hash:
    `aebd2f865762c3ac8d07225bb03a3e26e080873b12d99b3af47288bfcd32ddd5`.
- The exact v13 manifest passed tokenizer dry-run with matching hashes.
- Started and completed the ability run in tmux:
  `phase15_ability_v13`, output
  `phase15_ability_v13_callsite_q1redacted_n48_k192.json`, conditions
  `A/B/B_match` at `K=192`.
- Audited the v13 ability artifact:
  - `A=0.875`, `B=0.167`, `B_match=0.188`, `A-B_match=0.688`.
  - `CueOnly=0.0208` (`1/48` hit), `Q1=0.667`.
  - Minimum/maximum evicted context tokens: `8516/16260`.
  - The full-cache and matched-gap gates pass, but cue-only and answer-token
    retention gates fail (`max B/B_match answer overlap = 1.0`).
  - Strict eligible rows with `Q1=1`: `16` rows from `9` repos. Without the
    Q1 filter: `27` rows from `11` repos.
- Added the mandatory `AnchorWindow-K` locality control to the Phase 6 runner,
  Phase 15 manifest wrapper, repair audit, and default Phase 15 protocol.
- Fixed `WrongEvent-K` donor selection so wrong-event donors must come from a
  different repository with a different event and different answer. This avoids
  same-repo/duplicate-event controls on v13.
- Added a frozen v13 diagnostic repair protocol:
  `phase15_protocol_v13_callsite_q1redacted_repair_anchor.json`. It uses
  `K={96,192}` and includes all controls, including `ToolFile-K` and
  `AnchorWindow-K`.
- Focused Phase 15/Phase 6 tests after the repair-control and donor-metadata
  changes: `110 passed, 16 warnings`.
- Ran the v13 repair dry-runs:
  - `limit=5`, `K=192`, all controls prepared successfully.
  - Full manifest, `K={96,192}`, all controls prepared successfully.
- Ran the tiny GPU repair smoke:
  `phase15_repair_v13_smoke_limit5_k192_anchor.json`.
  It completed all 5 rows with all controls and all expected fields. ToolFile
  selected real file rows on every smoke row, and ToolFile/AnchorWindow were
  budget matched. Smoke scores are not evidence, but they exposed that
  `AnchorWindow-K` is a label-assisted locality reference (`5/5`), not a
  deployable baseline.
- Added repair-artifact sensitivity slices:
  `exclude_cue_only_hits`, `exclude_answer_retention`,
  `exclude_cue_and_answer_retention`, and `strict_repair_eligible`.
- Updated the repair audit so `AnchorWindow-K` is interpreted as a
  label-assisted locality reference rather than a deployable runtime baseline.
  The strict main-paper gate records whether RepairKV is competitive with this
  reference; if AnchorWindow dominates, Phase 15 should stay appendix/future
  work even if deployable/content-agnostic controls are positive.
- Added repo-level repair-audit summaries for each comparison: positive and
  nonnegative repo counts plus min/median/max per-repo mean lift. This is the
  dominance check for whether a result is carried by one or two repositories.
- Updated README, project status, paper guide, and outline terminology so Phase
  15 is described as a diagnostic until gated, and stale Gold-K wording is
  replaced with SpanRef-K.
- Ran the one allowed whole-manifest diagnostic repair run:
  `phase15_repair_v13_whole_k96_192_anchor.json`, conditions
  `A/B/B_match/RepairKV-EventOnly-K/Random-K/Oldest-K/StaleCue-K/WrongEvent-K/
  ToolFile-K/AnchorWindow-K`, `K={96,192}`.
- Tightened the repair audit after static review:
  - static manifest audit must pass for every used result row;
  - duplicate `example_id` rows within a K slice are rejected by the gate;
  - WrongEvent donor metadata is required when `WrongEvent-K` is present;
  - contamination-filtered sensitivity slices are explicit gate inputs;
  - ToolFile selection requires a minimum file-row fraction, and the paper
    names it as file-name-assisted with oldest-row backfill;
  - the legacy artifact was CPU-backfilled to
    `phase15_repair_v13_whole_k96_192_anchor_with_donors.json` without
    changing scores.
- Added appendix figure generation and a cautious main Results paragraph for
  the real-repository diagnostic, then rebuilt `paper/main.pdf`.

## Immediate Next Tasks

1. Keep Phase 15 as preliminary main-text evidence plus appendix details.
2. If editing the Results paragraph, appendix figure, or prose, preserve the caveats: not
   SWE-bench performance, not end-to-end coding validation, AnchorWindow is
   label-assisted, ToolFile is file-name-assisted with oldest-row backfill, and
   LexicalAnchor is answer-sanitized diagnostic evidence.
3. Maintain the main evidence stack as MQ-NIAH frontier, specificity,
   multi-turn, retention-rule sensitivity, and runtime capacity.
4. Future work can build a cleaner real-repository benchmark with a deployable
   locality-aware baseline and serialized wrong-event provenance from the start.

## Current Stop Criteria

- Do not put any Phase 15 selected-subset result in the main paper.
- Do not build another outcome-selected repair subset from the same v11 outputs.
- Do not run repair from v12; it failed the full ability and repo-balance gates.
- Treat v13 as the last bounded RepoDelta-Edge attempt. If v13 ability fails
  `A>=0.80`, clean cue-only, answer-retention, or at least 8 strict eligible
  rows from at least 8 repos, stop iterating the same design. Because v13
  passes ability/gap/yield but fails cue-only and retention, only one
  whole-manifest diagnostic repair run is allowed before stopping Edge.
- If v13 fails, pivot to a simpler controlled real-content diagnostic or leave
  real-repo evidence as appendix/future work.
- A paper-facing Phase 15 result requires a fresh frozen manifest, a whole-frame
  ability gate, repo-balanced repair eligibility, and deployable controls that
  are not explained by ToolFile-K. AnchorWindow-K is an explicitly
  label-assisted locality reference; it constrains the claim and must be
  reported if Phase 15 is promoted, but it is not a deployable-control gate.
- Strict-eligible selected rows are secondary/appendix only on v13. A selected
  subset cannot be the primary paper-facing result because eligibility is
  defined from the same ability artifact.

## GPU Efficiency Notes

- Do not start GPU runs until CPU manifest gates and tokenizer dry-runs pass.
- Load the model once per smoke/pilot process; do not launch per-row jobs.
- Keep ability smoke narrow before repair scoring, because exact query scoring
  is the expensive path.
- Use one K for ability smoke, two K values for repair smoke, and the full K
  grid only after both gates pass.
- All GPU jobs must run in tmux with persistent logs under
  `results/swebench_dev/`.

## V13 Execution Commands

After `phase15_build_v13` finishes, inspect static yield before GPU:

```bash
.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path
rows=[json.loads(line) for line in Path("phases/phase15_real_repo_relevance_shift/results/swebench_dev/repodelta_edge_swebench_manifest_probe_v13_callsite_q1redacted.jsonl").read_text().splitlines() if line.strip()]
print("rows", len(rows), "repos", len({r["repo"]["repo_id"] for r in rows}))
print("per_repo", Counter(r["repo"]["repo_id"] for r in rows))
print("edge_types", Counter(r["q2"]["edge_type"] for r in rows))
print("warnings", sum(bool(r["audit"]["warnings"]) for r in rows), "flags", sum(bool(r["audit"]["flags"]) for r in rows))
print("tokens", min(r["audit"]["rendered_context_tokens"] for r in rows), max(r["audit"]["rendered_context_tokens"] for r in rows))
PY
```

Then dry-run the exact frozen manifest:

```bash
.venv/bin/python -m phases.phase15_real_repo_relevance_shift.scripts.run_phase15_manifest \
  --manifest phases/phase15_real_repo_relevance_shift/results/swebench_dev/repodelta_edge_swebench_manifest_probe_v13_callsite_q1redacted.jsonl \
  --protocol phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_protocol_v13_callsite_q1redacted.json \
  --output phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_runner_dryrun_v13_callsite_q1redacted_k192.json \
  --dry-run --k 192 --conditions A B B_match
```

Only if both checks are clean, run ability in tmux:

```bash
tmux new-session -d -s phase15_ability_v13 \
  "cd /home/ubuntu/RepairKV && PYTHONUNBUFFERED=1 .venv/bin/python -m phases.phase15_real_repo_relevance_shift.scripts.run_phase15_manifest \
    --manifest phases/phase15_real_repo_relevance_shift/results/swebench_dev/repodelta_edge_swebench_manifest_probe_v13_callsite_q1redacted.jsonl \
    --protocol phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_protocol_v13_callsite_q1redacted.json \
    --output phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_ability_v13_callsite_q1redacted_n48_k192.json \
    --stage ability_v13_callsite_q1redacted --k 192 --conditions A B B_match \
    2>&1 | tee phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_ability_v13_callsite_q1redacted_n48_k192.log"
```

Audit strictly before any repair:

```bash
.venv/bin/python -m phases.phase15_real_repo_relevance_shift.scripts.audit_phase15_artifact \
  phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_ability_v13_callsite_q1redacted_n48_k192.json \
  --primary-k 192 --min-full-cache 0.80 --max-cue-only 0.0 \
  --min-gap 0.40 --max-answer-overlap 0.0 --max-cue-only-hits 0
```

Completed diagnostic repair run command:

```bash
tmux new-session -d -s phase15_repair_v13_full \
  "cd /home/ubuntu/RepairKV && stdbuf -oL -eL .venv/bin/python -m phases.phase15_real_repo_relevance_shift.scripts.run_phase15_manifest \
    --manifest phases/phase15_real_repo_relevance_shift/results/swebench_dev/repodelta_edge_swebench_manifest_probe_v13_callsite_q1redacted.jsonl \
    --protocol phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_protocol_v13_callsite_q1redacted_repair_anchor.json \
    --output phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_repair_v13_whole_k96_192_anchor.json \
    --stage repair_v13_whole_anchor --k 96 192 \
    --conditions A B B_match RepairKV-EventOnly-K Random-K Oldest-K StaleCue-K WrongEvent-K ToolFile-K AnchorWindow-K \
    2>&1 | tee phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_repair_v13_whole_k96_192_anchor.log"
```

Audit the provenance-complete repair artifact:

```bash
.venv/bin/python -m phases.phase15_real_repo_relevance_shift.scripts.audit_phase15_repair_artifact \
  phases/phase15_real_repo_relevance_shift/results/swebench_dev/phase15_repair_v13_whole_k96_192_anchor_with_donors.json \
  --bootstrap-draws 2000 --gate --primary-k 192 --adjacent-k 96 \
  --min-primary-lift 0.10
```

## Promotion Rule

Phase 15 gets a headline main-paper claim only if a locked RepoDelta-Edge run is
clean under the written gate in `phase15_plan.md`. The v13 run does not clear
that standard because AnchorWindow-K is stronger, but it is strong enough
against deployable controls to support a cautious preliminary Results paragraph.
A failed or messy real-repository result would stay out of the paper.
