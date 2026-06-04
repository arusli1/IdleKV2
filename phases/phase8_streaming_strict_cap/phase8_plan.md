# Phase 8: Strict-Cap Streaming + Bounded CPU Spill

Generated: 2026-05-01

## Thesis

Phase 6 showed that, after a single 32K compression, future-query-informed repair can beat a
matched no-repair baseline at the same final footprint.

Phase 8 asks a harder question:

- if the logical context is `10 x 32K = 320K`, and we enforce a **strict 32K active KV limit at
  all times**, can a small bounded CPU spill buffer still preserve enough signal for useful repair?

This is still a repair experiment, but now under **repeated streaming eviction**, not a one-shot
32K prefill.

## 1. Core Design

Keep the same high-level turn structure:

- long shared context `C`
- turn 1 question `Q1`
- turn 1 answer `A1`
- repair before turn 2
- turn 2 question `Q2`

The change is how `C` is loaded:

- total logical context length: `L_total = 327680`
- strict active GPU KV cap: `L_gpu_max = 32768`
- never allow the live tracked cache to exceed `32768` positions

## 2. Streaming Prefill Protocol

Context is streamed in fixed chunks.

Default chunk size:

- `chunk_size = 2048`

Rule:

1. start with empty active cache
2. append context chunks one by one
3. before appending a chunk that would push the active cache over `32768`, run SnapKV on the
   current active cache and evict `90%` of the currently active **context** positions
4. append the next chunk
5. continue until all `320K` logical context tokens have been processed

Active-cache invariant:

- after every append, eviction, question encode, and answer decode step:
  - `len(active_cache) <= 32768`

Implementation note:

- if turn-token headroom is needed to avoid transient overflow during `Q1` / `A1` / `Q2`, reserve a
  small fixed headroom `H_turn` and use an effective streaming context cap of
  `32768 - H_turn`
- default:
  - `H_turn = 512`

## 3. Repeated Eviction Policy

Every time the active cache is compressed during streaming:

- eviction policy: `SnapKV`
- eviction fraction: `90%`
- keep fraction on GPU: `10%`

Notation for one eviction event `t`:

- active context before eviction: `A_t`
- evicted positions this event: `E_t`
- kept positions this event: `K_t`

By design:

- `|K_t| ≈ 0.10 * |A_t|`
- `|E_t| ≈ 0.90 * |A_t|`

## 4. CPU Spill Policy

We do **not** keep all evicted tokens on CPU.

For each eviction event `t`:

1. compute a per-token query-norm score for every token in `E_t`
2. keep only the top `10%` of `E_t` on CPU
3. discard the remaining `90%` of `E_t` permanently

Per-token score:

- for token position `p`, let
  - `s_qnorm(p) = pooled L2 norm of the token's query vectors at eviction time`
- default pooling:
  - mean over layers
  - mean over heads

CPU spill retained from event `t`:

- `S_t = top_10_percent(E_t, by = s_qnorm)`

Final CPU spill buffer:

- `S_cpu = union_t S_t`

Expected scale:

- because each event stores `0.10 * 0.90 = 0.09` of the just-evicted mass, the final CPU spill
  should stay on the order of **one extra 32K window**, even though the model processed `320K`
  logical tokens

Optional hard cap:

- if the union of per-event spills grows above one active-window equivalent, apply one more global
  top-by-`s_qnorm` trim on CPU

## 5. Repair Step

Repair happens as in Phase 6, but now the candidate pool is only `S_cpu`.

At repair time:

1. the future query `Q2` is known
2. encode `Q2` against the final active cache
3. score CPU-spilled positions in `S_cpu` against `Q2`
4. select the top `b` CPU-spilled tokens
5. swap those `b` tokens into the active cache while swapping `b` currently active tokens out
6. keep the final active cache size exactly `32768`

- repair is now a **swap**, not an additive top-up
- the final GPU footprint is fixed at `32768`

Default swap rule:

- restore top `b` CPU-spilled positions by `Q2` score
- evict bottom `b` currently active context positions by the same `Q2` score

## 6. Conditions

Main conditions:

- `A_if_available`
  - only if a meaningful ceiling can be run for this setup
  - likely not feasible as a full 320K active-cache run on the current stack
- `B_stream`
  - repeated SnapKV streaming compression, no CPU spill used at repair
- `RepairKV_stream(b)`
  - repeated SnapKV streaming compression
  - CPU spill keeps top `10%` of each evicted batch by query-norm
  - repair swaps in the top `b` CPU-spilled tokens by `Q2` score
- `RandomRepair_stream(b)`
  - same CPU spill as above
  - swap in `b` random CPU-spilled tokens
- `OracleSpill_stream(b)`
  - same CPU spill as above
  - swap in the best `b` CPU-spilled tokens using hindsight over gold relevant positions

Recommended additional diagnostic:

- `RandomSpill_stream(b)`
  - during streaming, store a random `10%` of each evicted batch on CPU instead of top-by-query-norm
  - at repair time, use the normal `Q2`-aware selector

This isolates whether the **eviction-time spill triage** is useful.

Small-sample ceiling diagnostic:

- `OracleAll_stream(b)` on `n <= 4`
  - same repeated streaming protocol
  - but retain all evicted tokens for analysis only
  - this measures how much headroom is lost by the `10%` CPU spill policy itself

## 7. Benchmark

For now, stay on the same task family:

- `mq_niah_4q`

Use the same 3 turn splits as Phase 6:

- turn 1 asks `1 and 4`, turn 2 asks `2 and 3`
- turn 1 asks `2 and 4`, turn 2 asks `1 and 3`
- turn 1 asks `3 and 4`, turn 2 asks `1 and 2`

Reason:

- this keeps the Phase 8 change focused on **streaming repeated eviction**, not on a simultaneous
  task-family change

## 8. Metrics

Main metric:

- turn-2 score after repeated streaming compression

Report:

- `B_stream`
- `RepairKV_stream(b)`
- `RandomRepair_stream(b)`
- `OracleSpill_stream(b)`
- `RandomSpill_stream(b)` if run

Also report:

- `% examples where RepairKV_stream(b) > B_stream`
- spill coverage:
  - fraction of gold relevant positions that ever enter `S_cpu`
- repair coverage:
  - fraction of gold relevant positions restored at repair time
- number of streaming eviction events
- final CPU spill size
- mean repair latency
- mean eviction-time spill selection latency

## 9. Suggested `b` Grid

Calibration grid:

- `b in {512, 1024, 2048, 4096, 8192}`

Reason:

- much smaller values are likely too tiny relative to the repeated-eviction setting
- much larger values start replacing a large fraction of the final 32K active cache

Downselect after calibration if needed.

## 10. Preflight Risks

### 10.1 Logical Position Risk

Even though the active cache never exceeds `32K`, the logical positions still advance to `320K`.

So the first preflight check is:

- can the current Qwen2.5-7B-Instruct runtime handle logical positions near `320K` without runtime
  errors or obvious collapse?

If not:

- reduce `L_total` to the largest stable long-position regime (`128K` or `256K`) and keep the
  protocol otherwise identical

### 10.2 CPU Spill Size

Verify that:

- the union of per-eviction CPU spills stays near one extra-window scale
- if it does not, add the global hard cap described above

### 10.3 Strict Cap Integrity

Add an assertion in the runner:

- `assert len(active_cache) <= 32768`

after every state mutation.

## 11. Run Stages

### Stage 1: Geometry Smoke

Goal:

- verify strict-cap streaming works at all
- verify logical positions >32K are stable
- verify repeated evictions and CPU spill accumulation behave as expected

Run:

- one split only
- `n = 2`
- one `b` value: `2048`
- conditions:
  - `B_stream`
  - `RepairKV_stream`
  - `OracleSpill_stream`

### Stage 2: Calibration

Goal:

- choose the `b` grid and confirm the spill policy produces nontrivial headroom

Run:

- all 3 splits
- `n = 8`
- `b in {512, 1024, 2048, 4096, 8192}`
- conditions:
  - `B_stream`
  - `RepairKV_stream`
  - `RandomRepair_stream`
  - `OracleSpill_stream`
  - optional `RandomSpill_stream`

Success pattern:

- `RepairKV_stream(b) > B_stream`
- `RandomRepair_stream(b)` near `B_stream`
- `OracleSpill_stream(b) > RepairKV_stream(b)`

### Stage 3: Full Run

After calibration:

- all 3 splits
- `n = 100`
- chosen `b` grid
- main conditions:
  - `B_stream`
  - `RepairKV_stream`
  - `RandomRepair_stream`
  - `OracleSpill_stream`
  - optional `RandomSpill_stream`

## 12. Paper Value If This Works

If this works, the paper story becomes stronger:

- Phase 6: one-shot 32K matched-footprint repair works
- Phase 8: the same idea survives **repeated forgetting** over a `10x` longer logical context while
  keeping only:
  - a strict `32K` active GPU cache
  - a small bounded CPU spill buffer

That would move the story from:

- “repair helps after one compression”

to:

- “repair can serve as a bounded external memory mechanism under repeated long-context eviction”

## 13. Execution Notes: 2026-05-02

Implemented and smoke-tested the strict-cap streaming runner.

Code added:

- `src/streaming.py`
  - strict-cap streaming geometry
  - repeated SnapKV eviction
  - top-query-norm CPU spill
  - random-spill diagnostic
  - swap-based repair that preserves active footprint
- `src/runner.py`
  - Phase 8 CLI-backed experiment runner
  - exact Q2 repair scoring
  - `B_stream`, `RepairKV_stream`, `RandomRepair_stream`, `OracleSpill_stream`,
    `RandomSpill_stream`
- `scripts/run_phase8.py`
- `tests/test_streaming.py`

Validation:

- `python -m unittest phases.phase8_streaming_strict_cap.tests.test_streaming -v`
  passed.
- Phase 6 support tests plus Phase 8 tests passed:
  - 30 tests total.

### Smoke Results

Small live implementation smoke:

- task: `mq_niah_4q_split_14_to_23`
- logical context: `65,536`
- `n=1`, `b=512`
- result:
  - active context after streaming: `9,319`
  - qnorm spill size: `5,622`
  - qnorm spill coverage of Q2 gold tokens: `0.0`
  - `B_stream = 0.0`
  - `RepairKV_stream = 0.0`
  - `OracleSpill_stream = 0.0`

Stage 1 320K smoke:

- task: `mq_niah_4q_split_14_to_23`
- logical context: `327,680`
- `n=2`, `b=2048`
- elapsed: `233.49s`
- mean streaming prefill time: `63.08s` per base example
- geometry:
  - eviction events: `11`
  - peak active context tokens: `31,858`
  - final active context tokens: `13,426`
  - qnorm spill size: `31,433`
- aggregate:
  - `B_stream = 0.0`
  - `RepairKV_stream = 0.0`
  - `OracleSpill_stream = 0.0`
  - qnorm spill coverage: `0.0`
  - random-spill coverage diagnostic: `0.119565`

Broader split smoke:

- task: `clean_suite`
- logical context: `327,680`
- `n=1`, all three clean splits, `b=2048`
- elapsed: `174.37s`
- mean streaming prefill time: `63.29s`
- aggregate:
  - `B_stream = 0.0`
  - `RepairKV_stream = 0.0`
  - `RandomRepair_stream = 0.0`
  - `OracleSpill_stream = 0.0`
  - `RandomSpill_stream = 0.0`
  - qnorm spill coverage: `0.0`
  - random-spill coverage: `0.028985`

Per-split qnorm spill coverage:

- `1,4 -> 2,3`: `0.0`
- `2,4 -> 1,3`: `0.0`
- `3,4 -> 1,2`: `0.0`

### Decision

Do not run the Stage 2 calibration or Stage 3 full run with the current Phase 8 defaults.

Reason:

- The core high-L2 CPU-spill policy fails the smoke gate before repair:
  - the Q2 gold tokens never enter the qnorm spill buffer in the observed 320K smokes.
- Therefore even `OracleSpill_stream` cannot improve, because it is constrained to the same failed
  qnorm spill buffer.
- A full run would be a larger confirmation of a known zero-coverage failure, not useful paper
  evidence.

### Diagnosis

The failure is not mainly the repair selector.

The blocker is the eviction-time CPU-spill triage:

- qnorm spill stores about one extra window of tokens, as planned.
- But the selected high-query-norm tokens are not the relevant needle spans.
- Random spill occasionally captures a small number of Q2-relevant tokens, which confirms the gold
  spans are passing through evicted regions; the top-L2 rule is just not selecting them.

There is also an experimental-design issue:

- With the exact rule “evict 90% whenever the next chunk would overflow,” active cache size
  sawtooths.
- At exactly `10 x 32K` logical context and `chunk_size=2048`, the final active context is only
  `13,426` tokens, not near the `32K` cap.
- This is still a valid strict-cap run, but it underuses the allowed active budget at the final
  query point and weakens the paper framing.

### Recommended Next Step

Treat Phase 8 as a negative pilot for query-norm-only spill triage.

If continuing this direction, run a redesigned spill-policy diagnostic before any full suite:

- `OracleGoldSpill_stream`
  - store only gold span tokens when they are evicted
  - verifies whether 320K logical positions plus swap repair can recover answers if the right CPU
    tokens are available
- spill-policy ablation at `n <= 2`
  - top qnorm
  - random spill
  - SnapKV-score spill
  - query/key-norm hybrid
  - global hard-cap variants
- final-active geometry sweep
  - choose a chunk schedule or final-query placement that ends near the 32K active cap instead of
    `13,426`

Only run a full Phase 8 suite after:

- qnorm or replacement spill coverage is nonzero on all three clean splits
- `OracleSpill_stream > B_stream` in smoke
- expected full runtime is under the available compute budget

## 14. Follow-up: 64K Top-X QNorm Spill Binary Diagnostic

Question:

- If total logical context is only `2 x 32K = 64K`, how much of the evicted cache must be stored
  on CPU by top query-norm before Q2 needles appear in the spill buffer?

Implementation:

- Added `scripts/binary_search_qnorm_spill_threshold.py`.
- Uses a cheaper fill-cap streaming schedule:
  - stream `65,536` tokens
  - strict cap `32,768`
  - turn headroom `512`
  - effective context cap `32,256`
  - evict only enough tokens to fit each next chunk
- Streams each sample once, ranks every evicted token by qnorm within its eviction event, then
  computes exact top-X thresholds from those ranks.

Run:

- task: `clean_suite`
- `n=2`
- target coverages: any token, 25%, 50%, 100%
- artifact:
  - `results/coverage_binary_search/clean_suite_l65536_cap32768_h512_chunk2048_n2_targets0.01-0.25-0.5-1.json`

Geometry:

- final active context tokens: `32,256`
- total evicted tokens per sample: `33,280`
- stream time: about `30.5s` per sample while another Phase 6 GPU job was active

Main result:

- top `10%` qnorm spill captured `0.0` of evicted Q2 needle tokens
- top `25%` qnorm spill captured about `0.0074` of evicted Q2 needle tokens
- top `50%` qnorm spill captured about `0.0960` of evicted Q2 needle tokens

Minimum top-X needed:

- to capture any evicted Q2 needle token:
  - mean: `30.99%`
  - worst split/example: `49.02%`
- to capture 25% of evicted Q2 needle tokens:
  - mean: `71.85%`
  - worst: `75.39%`
- to capture 50% of evicted Q2 needle tokens:
  - mean: `82.76%`
  - worst: `87.35%`
- to capture all evicted Q2 needle tokens:
  - mean: `99.58%`
  - worst: `99.66%`

Decision:

- top-L2/query-norm CPU spill is not a viable triage rule, even in the easier 64K setting.
- Storing only top `10%` is far too aggressive for NIAH-style evidence.
- If Phase 8 continues, the next candidate should be a different spill selector:
  - SnapKV-score spill
  - lexical/key-span aware diagnostic
  - random spill baseline with much larger X
  - oracle-gold spill to confirm repair itself still works when the CPU buffer contains the right
    tokens

## 15. Follow-up: Two-Tier SnapKV Spill Smoke

Question:

- Instead of using query-norm to choose CPU spill tokens, can SnapKV itself do both tiers?

Protocol:

- total logical context: `65,536`
- strict active cap: `32,768`
- effective context cap with turn headroom: `32,256`
- task: `clean_suite`
- `n=1`
- fractions tested: `10%`, `25%`, `50%`, `100%` of tokens that would otherwise be removed from
  the GPU cache

At each overflow:

1. run SnapKV on the active cache to form a candidate pool
2. permanently discard tokens outside that candidate pool
3. run SnapKV again on the candidate pool
4. keep the top partition on GPU and spill the second partition to CPU

Implementation:

- Added `stream_context_two_tier_snapkv_spill(...)` in `src/streaming.py`.
- Added `scripts/sweep_two_tier_snapkv_spill.py`.
- Unit tests passed:
  - `python -m unittest phases.phase8_streaming_strict_cap.tests.test_streaming -v`

Artifact:

- `results/two_tier_snapkv/clean_suite_l65536_cap32768_h512_chunk2048_n1_x0.1-0.25-0.5-1.json`

Aggregate coverage of Q2-relevant tokens:

- `10%` CPU spill:
  - active coverage: `0.0290`
  - CPU coverage: `0.0725`
  - accessible coverage: `0.1014`
  - CPU spill size: `3,332`
- `25%` CPU spill:
  - active coverage: `0.0290`
  - CPU coverage: `0.1594`
  - accessible coverage: `0.1884`
  - CPU spill size: `8,320`
- `50%` CPU spill:
  - active coverage: `0.0290`
  - CPU coverage: `0.2899`
  - accessible coverage: `0.3188`
  - CPU spill size: `16,640`
- `100%` CPU spill:
  - active coverage: `0.0290`
  - CPU coverage: `0.9710`
  - accessible coverage: `1.0000`
  - CPU spill size: `33,280`

Interpretation:

- Two-tier SnapKV is clearly better than top-query-norm spill at small CPU budgets:
  - top `10%` qnorm captured `0.0` evicted Q2-token coverage in the 64K binary diagnostic
  - `10%` two-tier SnapKV captured `0.0741` of evicted Q2-relevant tokens
- However, `10%` and `25%` CPU storage are still too lossy for a main paper experiment.
- The clean next smoke is a two-tier SnapKV `X` sweep between `50%` and `100%`, followed by actual
  repair scoring only after accessible coverage is high enough to make repair meaningful.

## 16. Larger Phase-6/7-Style Two-Tier Test Design

This is the next real Phase 8 experiment if we want evidence comparable in structure to Phase 6/7.

### Hypothesis

Under a strict `32K` active-GPU KV cap, a two-tier SnapKV memory can preserve useful offloaded
tokens and later improve turn-2 accuracy by swapping Q2-relevant CPU tokens back onto GPU.

This differs from Phase 6/7 in one important way:

- Phase 6/7 repair drew from a one-shot evicted buffer after compressing a `32K` context.
- Phase 8 repair draws from a bounded CPU tier produced during streaming of a larger logical
  context.

### Main Test

Use a `64K` logical context first.

Reason:

- it is the smallest setting that genuinely exceeds the `32K` active cap
- it forces repeated strict-cap eviction while ending near the full active budget
- it is much cheaper and cleaner than the failed `320K` qnorm pilot
- the two-tier coverage smoke already showed nonzero recoverable Q2 signal at this length

Fixed settings:

- model: `Qwen2.5-7B-Instruct`
- logical context length: `65,536`
- active GPU cap: `32,768`
- turn headroom: `512`
- effective context cap: `32,256`
- chunk size: `2,048`
- compressor: `SnapKV`
- sink size: `4`
- recency window: `128`
- observation window: `128`
- score mode for repair: exact Q2 query rows

Benchmark:

- primary: `mq_niah_4q` clean suite
  - `1,4 -> 2,3`
  - `2,4 -> 1,3`
  - `3,4 -> 1,2`
- optional broader panel only after the `4q` suite is locked:
  - `mq_niah_6q_clean_suite`

### Two-Tier Streaming Policy

At each overflow event:

1. compute the GPU-only keep budget needed to append the next chunk without exceeding the cap
2. run SnapKV to keep a candidate pool of size:
   - `gpu_keep_budget + X * (pre_evict_tokens - gpu_keep_budget)`
3. permanently discard tokens outside the candidate pool
4. run SnapKV again on the candidate pool
5. keep `gpu_keep_budget` tokens on GPU
6. spill the remaining candidate tokens to CPU

`X` is the CPU-store fraction of tokens that a GPU-only SnapKV compression would have evicted.

Important:

- `X < 1.0` means some tokens are permanently forgotten
- `X = 1.0` means no permanent drop at `64K`; it is a diagnostic upper bound, not the preferred
  main policy

### Conditions

Use the same logic as Phase 6/7, adapted to swap repair:

- `B_stream(X)`
  - two-tier streaming cache after Q1
  - no CPU repair
  - matched active-GPU footprint baseline
- `RepairKV_stream(X, K)`
  - rank CPU-spilled tokens with exact Q2 query rows
  - swap top `K` CPU tokens into GPU
  - drop the bottom `K` active context tokens by the same Q2 score
- `Random-K_stream(X, K)`
  - same CPU spill
  - swap random CPU tokens
- `Oldest-K_stream(X, K)`
  - same CPU spill
  - swap oldest CPU-spilled tokens
- `Oracle-K_stream(X, K)`
  - gold-span hindsight ceiling over the CPU spill
  - still constrained to the same CPU tier
- `OracleAll_stream(K)`
  - diagnostic only
  - CPU tier contains every token that left GPU
  - measures whether failures come from spill triage or repair selection

Do not include the old qnorm-spill `RepairKV_stream` in the main result.

### Calibration Stages

#### Stage A: CPU-Tier Coverage Calibration

Goal:

- choose one or two `X` values where enough Q2 evidence is accessible before spending generation
  compute

Run:

- task: `clean_suite`
- context: `65,536`
- `n = 4`
- `X in {0.50, 0.625, 0.75, 0.875, 1.0}`
- no answer generation required

Report:

- active Q2-token coverage
- CPU Q2-token coverage
- accessible Q2-token coverage
- CPU spill size
- permanently evicted token count
- per-split coverage

Gate:

- choose the smallest `X < 1.0` with pooled accessible coverage at least `0.50`
- require every split to have nonzero CPU coverage
- keep `X = 1.0` as an upper-bound diagnostic regardless

Current `n=1` coverage anchor:

| X | CPU tokens | permanent drops | accessible coverage |
|---:|---:|---:|---:|
| `0.10` | `3,332` | `29,948` | `0.101` |
| `0.25` | `8,320` | `24,960` | `0.188` |
| `0.50` | `16,640` | `16,640` | `0.319` |
| `1.00` | `33,280` | `0` | `1.000` |

Expected decision:

- likely test `X = 0.75` or `0.875` as the main permanently-dropping policy
- also keep `X = 1.0` as the spill-triage ceiling

#### Stage B: Repair Frontier Smoke

Goal:

- verify that Q2-conditioned swap repair improves actual answer score, not just coverage

Run:

- task: `clean_suite`
- context: `65,536`
- `n = 8`
- `X`: selected Stage-A policy plus `1.0`
- `K in {64, 128, 256, 512, 1024, 2048, 4096}`
- conditions:
  - `B_stream`
  - `RepairKV_stream`
  - `Random-K_stream`
  - `Oldest-K_stream`
  - `Oracle-K_stream`

Gate:

- pooled `B_stream` is nonzero but not saturated
- `RepairKV_stream(K) > B_stream` for some middle `K`
- `Random-K_stream` and `Oldest-K_stream` stay near `B_stream`
- `Oracle-K_stream > RepairKV_stream` before the largest `K`
- largest `K` does not simply destroy performance by swapping out too much active context

#### Stage C: Full 4q Panel

Run only after Stage B passes.

Run:

- task: `clean_suite`
- context: `65,536`
- `n = 100`
- one selected `X < 1.0`
- optional diagnostic `X = 1.0` at lower `n` if full cost is too high
- `K`: downselected from Stage B, likely `K in {128, 256, 512, 1024, 2048}`
- conditions:
  - `B_stream`
  - `RepairKV_stream`
  - `Random-K_stream`
  - `Oldest-K_stream`
  - `Oracle-K_stream`

Main figure:

- score vs `K`, pooled across the three clean splits

Main table:

- endpoint score
- lift over `B_stream`
- CPU spill size
- permanent drop count
- repair coverage
- mean stream-prefill time
- mean exact repair time

Appendix:

- per-split curves
- accessible coverage vs `X`
- `X = 1.0` diagnostic ceiling

### Optional Harder Panel

Only run after the `4q` Phase 8 panel is clean.

Run:

- task: `mq_niah_6q_clean_suite`
- context: `65,536`
- `n = 32` smoke first, then `n = 100` only if useful
- use the same selected `X` if coverage is adequate
- otherwise recalibrate `X` with the same Stage-A rule

Reason:

- Phase 7 showed `6q` is the right harder same-family extension
- but Phase 8 already changes the memory regime, so `6q` should not be mixed into the main claim
  until the `4q` streaming result is locked

### Acceptance Rule

The full Phase 8 panel is paper-useful if all of the following hold:

- pooled `B_stream` is above `0.05` and below `0.80`
- pooled `RepairKV_stream - B_stream >= 0.10` at one middle `K`
- pooled `RepairKV_stream` exceeds both random and oldest controls at the same `K`
- pooled `Oracle-K_stream - RepairKV_stream >= 0.05` at one middle `K`
- every clean split has nonzero CPU coverage at the selected `X`
- the selected `X < 1.0`, so the main condition includes permanent forgetting
- the active GPU cache never exceeds `32,768`

If the selected `X < 1.0` fails but `X = 1.0` works:

- report Phase 8 as bounded CPU offload under a strict GPU cap
- do not claim successful lossy CPU triage

If both fail:

- Phase 8 remains a negative result for streaming spill triage
- do not run a full `6q` panel

### Implementation Gap Before Running

The current qnorm Phase 8 runner should not be reused for this test.

Required code changes:

- have `stream_context_two_tier_snapkv_spill(...)` return the CPU spill cache, not only spill
  positions
- add a two-tier repair runner with the conditions above
- add `Oldest-K_stream`
- add `OracleAll_stream` as a small diagnostic mode
- add task alias support for `mq_niah_6q_clean_suite` if the optional hard panel is run
- add artifact audit checks:
  - strict active cap
  - restored count equals `K` when enough CPU tokens exist
  - repaired active cache size equals no-repair active cache size
  - selected restore positions come only from CPU spill
  - dropped positions come only from active context
