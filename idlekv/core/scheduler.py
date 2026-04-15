"""
Idle-time scheduler for cache refinement.

Supports two modes:

- Legacy phase gating (`phases=1/2/1+2`, `policy=None`)
- Policy-driven anytime repair (`policy=shadow_only|sampled_spans|full_refresh`)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from idlekv.core.anytime_repair import repair_layer_pool, sample_cold_spans
from idlekv.core.phase1_rescore import RepairPassResult, phase1_rescore
from idlekv.core.phase2_refresh import FullKVStore, phase2_refresh
from idlekv.core.query_buffer import QueryBuffer
from idlekv.core.shadow_buffer import ShadowBuffer
from idlekv.utils.kv_cache import get_layer_kv, set_layer_kv


@dataclass
class RefinementResult:
    """Result of an idle-time refinement pass."""
    past_key_values: tuple
    phase1_ran: bool
    phase1_time_ms: float
    phase2_ran: bool
    phase2_layers_refreshed: int
    phase2_time_ms: float
    total_time_ms: float
    was_interrupted: bool
    policy: str = "legacy"
    sampled_rounds: int = 0
    sampled_layers: int = 0
    sampled_tokens: int = 0
    cpu_bytes_loaded: int = 0
    sampled_time_ms: float = 0.0
    retained_positions: Optional[list[torch.Tensor]] = None


class IdleScheduler:
    """Orchestrates idle-time refinement during tool-call pauses."""

    def __init__(
        self,
        shadow_buffer: ShadowBuffer,
        query_buffer: QueryBuffer,
        full_kv_store: FullKVStore,
        model,
        budget_per_layer: int,
        num_layers: int,
        retained_positions: Optional[list[torch.Tensor]] = None,
        prefill_importance_scores: Optional[list[torch.Tensor]] = None,
        anytime_min_idle_ms: float = 20.0,
        anytime_shadow_only_max_ms: float = 80.0,
        sample_span_size: int = 16,
        sample_spans_per_layer: int = 2,
        sample_sampler_seed: int = 0,
    ):
        self.shadow_buffer = shadow_buffer
        self.query_buffer = query_buffer
        self.full_kv_store = full_kv_store
        self.model = model
        self.budget_per_layer = budget_per_layer
        self.num_layers = num_layers
        self.retained_positions = retained_positions or []
        self.prefill_importance_scores = prefill_importance_scores or []
        self.anytime_min_idle_ms = anytime_min_idle_ms
        self.anytime_shadow_only_max_ms = anytime_shadow_only_max_ms
        self.sample_span_size = sample_span_size
        self.sample_spans_per_layer = sample_spans_per_layer
        self.sample_sampler_seed = sample_sampler_seed
        self.device = next(model.parameters()).device
        self._stop_event = threading.Event()

    def _clone_retained_positions(self) -> list[torch.Tensor]:
        return [pos.clone() for pos in self.retained_positions]

    def _check_interrupt(self) -> bool:
        return self._stop_event.is_set()

    def interrupt(self):
        self._stop_event.set()

    @staticmethod
    def normalize_phases(phases: Union[str, int, None]) -> Tuple[bool, bool, str]:
        if phases is None:
            return True, True, "1+2"
        key = str(phases).strip().lower()
        if key in {"1", "phase1"}:
            return True, False, "1"
        if key in {"2", "phase2"}:
            return False, True, "2"
        if key in {"1+2", "both", "all"}:
            return True, True, "1+2"
        raise ValueError(f"Unsupported phase selection: {phases!r}")

    @staticmethod
    def normalize_policy(policy: Optional[str]) -> Optional[str]:
        if policy is None:
            return None
        key = str(policy).strip().lower()
        if key not in {"shadow_only", "sampled_spans", "full_refresh"}:
            raise ValueError(f"Unsupported refinement policy: {policy!r}")
        return key

    def _parse_phase1_result(
        self,
        result,
        working_positions: list[torch.Tensor],
    ) -> tuple[object, list[torch.Tensor], int, int, int]:
        if isinstance(result, RepairPassResult):
            return (
                result.past_key_values,
                result.retained_positions,
                result.layers_touched,
                result.candidate_tokens,
                result.sampled_tokens,
            )
        return result, working_positions, 0, 0, 0

    def _parse_phase2_result(
        self,
        result,
        working_positions: list[torch.Tensor],
    ) -> tuple[object, list[torch.Tensor], int]:
        if isinstance(result, tuple) and len(result) == 3:
            kv, positions, layers_refreshed = result
            return kv, positions, layers_refreshed
        if isinstance(result, tuple) and len(result) == 2:
            kv, layers_refreshed = result
            return kv, working_positions, layers_refreshed
        return result, working_positions, 0

    def _check_factory(self, start: float, max_time_ms: Optional[float]):
        if max_time_ms is not None:
            deadline = start + max_time_ms / 1000.0

            def check():
                return self._stop_event.is_set() or time.perf_counter() > deadline
        else:
            check = self._check_interrupt
        return check

    @torch.no_grad()
    def run(
        self,
        past_key_values: tuple,
        generated_kv: Optional[list] = None,
        max_time_ms: Optional[float] = None,
        num_generated: int = 0,
        phases: Union[str, int, None] = "1+2",
        policy: Optional[str] = None,
    ) -> RefinementResult:
        self._stop_event.clear()
        start = time.perf_counter()
        check = self._check_factory(start, max_time_ms)
        normalized_policy = self.normalize_policy(policy)
        working_positions = self._clone_retained_positions()

        if normalized_policy is None:
            return self._run_legacy(
                past_key_values=past_key_values,
                generated_kv=generated_kv,
                max_time_ms=max_time_ms,
                num_generated=num_generated,
                phases=phases,
                check=check,
                start=start,
                working_positions=working_positions,
            )

        return self._run_policy(
            past_key_values=past_key_values,
            generated_kv=generated_kv or [],
            max_time_ms=max_time_ms,
            num_generated=num_generated,
            policy=normalized_policy,
            check=check,
            start=start,
            working_positions=working_positions,
        )

    def _run_legacy(
        self,
        *,
        past_key_values,
        generated_kv,
        max_time_ms,
        num_generated,
        phases,
        check,
        start,
        working_positions,
    ) -> RefinementResult:
        run_phase1, run_phase2, _ = self.normalize_phases(phases)

        kv = past_key_values
        p1_ran = False
        p1_time = 0.0
        if run_phase1:
            p1_start = time.perf_counter()
            phase1_out = phase1_rescore(
                past_key_values=kv,
                shadow_buffer=self.shadow_buffer,
                query_buffer=self.query_buffer,
                model=self.model,
                budget_per_layer=self.budget_per_layer,
                num_layers_arg=self.num_layers,
                interrupt_flag=check,
                num_generated=num_generated,
                retained_positions=working_positions,
            )
            kv, working_positions, _, _, _ = self._parse_phase1_result(phase1_out, working_positions)
            p1_time = (time.perf_counter() - p1_start) * 1000
            p1_ran = True

        p2_ran = False
        p2_layers = 0
        p2_time = 0.0
        phase2_allowed = run_phase2 and (max_time_ms is None or max_time_ms > 100.0)
        if phase2_allowed and not check() and len(self.full_kv_store) > 0:
            p2_start = time.perf_counter()
            phase2_out = phase2_refresh(
                past_key_values=kv,
                full_kv_store=self.full_kv_store,
                generated_kv=generated_kv or [],
                query_buffer=self.query_buffer,
                model=self.model,
                budget_per_layer=self.budget_per_layer,
                interrupt_flag=check,
                retained_positions=working_positions,
            )
            kv, working_positions, p2_layers = self._parse_phase2_result(phase2_out, working_positions)
            p2_time = (time.perf_counter() - p2_start) * 1000
            p2_ran = True

        total_time = (time.perf_counter() - start) * 1000
        return RefinementResult(
            past_key_values=kv,
            phase1_ran=p1_ran,
            phase1_time_ms=p1_time,
            phase2_ran=p2_ran,
            phase2_layers_refreshed=p2_layers,
            phase2_time_ms=p2_time,
            total_time_ms=total_time,
            was_interrupted=self._stop_event.is_set(),
            retained_positions=working_positions,
        )

    def _run_policy(
        self,
        *,
        past_key_values,
        generated_kv,
        max_time_ms,
        num_generated,
        policy,
        check,
        start,
        working_positions,
    ) -> RefinementResult:
        kv = past_key_values
        p1_ran = False
        p1_time = 0.0
        p2_ran = False
        p2_layers = 0
        p2_time = 0.0
        sampled_rounds = 0
        sampled_layers = 0
        sampled_tokens = 0
        cpu_bytes_loaded = 0
        sampled_time_ms = 0.0

        if max_time_ms is not None and max_time_ms < self.anytime_min_idle_ms:
            total_time = (time.perf_counter() - start) * 1000
            return RefinementResult(
                past_key_values=kv,
                phase1_ran=False,
                phase1_time_ms=0.0,
                phase2_ran=False,
                phase2_layers_refreshed=0,
                phase2_time_ms=0.0,
                total_time_ms=total_time,
                was_interrupted=self._stop_event.is_set(),
                policy=policy,
                retained_positions=working_positions,
            )

        if policy in {"shadow_only", "sampled_spans"} and self.query_buffer.count > 0 and not check():
            p1_start = time.perf_counter()
            phase1_out = phase1_rescore(
                past_key_values=kv,
                shadow_buffer=self.shadow_buffer,
                query_buffer=self.query_buffer,
                model=self.model,
                budget_per_layer=self.budget_per_layer,
                num_layers_arg=self.num_layers,
                interrupt_flag=check,
                num_generated=num_generated,
                retained_positions=working_positions,
            )
            kv, working_positions, _, _, _ = self._parse_phase1_result(phase1_out, working_positions)
            p1_time = (time.perf_counter() - p1_start) * 1000
            p1_ran = True

        if (
            policy == "sampled_spans"
            and self.query_buffer.count > 0
            and not check()
            and (max_time_ms is None or max_time_ms > self.anytime_shadow_only_max_ms)
        ):
            sampled_start = time.perf_counter()
            round_idx = 0
            while not check():
                touched_in_round = 0
                for layer_idx in range(self.num_layers):
                    if check():
                        break
                    if layer_idx >= len(working_positions) or layer_idx >= len(self.prefill_importance_scores):
                        continue

                    _, _, shadow_positions = self.shadow_buffer.get(layer_idx)
                    cold_batch = sample_cold_spans(
                        full_kv_store=self.full_kv_store,
                        layer_idx=layer_idx,
                        retained_positions=working_positions[layer_idx],
                        shadow_positions=shadow_positions,
                        importance_scores=self.prefill_importance_scores[layer_idx],
                        span_size=self.sample_span_size,
                        num_spans=self.sample_spans_per_layer,
                        seed=self.sample_sampler_seed + (round_idx * self.num_layers) + layer_idx,
                        device=self.device,
                    )
                    if cold_batch.positions.numel() == 0:
                        continue

                    current_k, current_v = get_layer_kv(kv, layer_idx)
                    recent_h = self.query_buffer.get(layer_idx=layer_idx)
                    shadow_k, shadow_v, shadow_positions = self.shadow_buffer.get(layer_idx)
                    outcome = repair_layer_pool(
                        current_k=current_k,
                        current_v=current_v,
                        retained_positions=working_positions[layer_idx],
                        shadow_k=shadow_k,
                        shadow_v=shadow_v,
                        shadow_positions=shadow_positions,
                        recent_h=recent_h,
                        model=self.model,
                        layer_idx=layer_idx,
                        budget_per_layer=self.budget_per_layer,
                        cold_candidates=cold_batch,
                    )

                    self.shadow_buffer.clear(layer_idx)
                    if outcome.shadow_positions.numel() > 0:
                        self.shadow_buffer.push(
                            layer_idx,
                            outcome.shadow_k,
                            outcome.shadow_v,
                            positions=outcome.shadow_positions,
                        )

                    kv = set_layer_kv(kv, layer_idx, outcome.new_k, outcome.new_v)
                    working_positions[layer_idx] = outcome.retained_positions
                    touched_in_round += 1
                    sampled_layers += 1
                    sampled_tokens += int(cold_batch.positions.numel())
                    cpu_bytes_loaded += int(cold_batch.bytes_loaded)

                if touched_in_round == 0:
                    break
                sampled_rounds += 1
                round_idx += 1

            sampled_time_ms = (time.perf_counter() - sampled_start) * 1000

        if policy == "full_refresh" and len(self.full_kv_store) > 0 and not check():
            p2_start = time.perf_counter()
            phase2_out = phase2_refresh(
                past_key_values=kv,
                full_kv_store=self.full_kv_store,
                generated_kv=generated_kv,
                query_buffer=self.query_buffer,
                model=self.model,
                budget_per_layer=self.budget_per_layer,
                interrupt_flag=check,
                retained_positions=working_positions,
            )
            kv, working_positions, p2_layers = self._parse_phase2_result(phase2_out, working_positions)
            self.shadow_buffer.clear()
            p2_time = (time.perf_counter() - p2_start) * 1000
            p2_ran = True

        total_time = (time.perf_counter() - start) * 1000
        return RefinementResult(
            past_key_values=kv,
            phase1_ran=p1_ran,
            phase1_time_ms=p1_time,
            phase2_ran=p2_ran,
            phase2_layers_refreshed=p2_layers,
            phase2_time_ms=p2_time,
            total_time_ms=total_time,
            was_interrupted=self._stop_event.is_set(),
            policy=policy,
            sampled_rounds=sampled_rounds,
            sampled_layers=sampled_layers,
            sampled_tokens=sampled_tokens,
            cpu_bytes_loaded=cpu_bytes_loaded,
            sampled_time_ms=sampled_time_ms,
            retained_positions=working_positions,
        )
