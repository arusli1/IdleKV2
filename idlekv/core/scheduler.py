"""
Tiered idle-time scheduler.

Decides which refinement operations to run based on estimated idle duration.
Both phases are anytime-safe: they can be interrupted at any layer boundary.

Tier 0 (<100ms): Phase 1 re-scoring only
Tier 1 (100ms-2s): Phase 1 + Phase 2 (progressive, as many layers as fit)
"""

import time
import threading
from typing import Optional
from dataclasses import dataclass

from idlekv.core.shadow_buffer import ShadowBuffer
from idlekv.core.query_buffer import QueryBuffer
from idlekv.core.phase1_rescore import phase1_rescore
from idlekv.core.phase2_refresh import phase2_refresh, CPUKVStore


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


class IdleScheduler:
    """
    Orchestrates idle-time refinement during tool-call pauses.

    Usage:
        scheduler = IdleScheduler(shadow_buffer, query_buffer, cpu_kv_store, model, ...)

        # When tool call is detected:
        result = scheduler.run(past_key_values, generated_kv)
        past_key_values = result.past_key_values
    """

    def __init__(
        self,
        shadow_buffer: ShadowBuffer,
        query_buffer: QueryBuffer,
        cpu_kv_store: CPUKVStore,
        model,
        budget_per_layer: int,
        num_layers: int,
    ):
        self.shadow_buffer = shadow_buffer
        self.query_buffer = query_buffer
        self.cpu_kv_store = cpu_kv_store
        self.model = model
        self.budget_per_layer = budget_per_layer
        self.num_layers = num_layers
        self._interrupted = False

    def _check_interrupt(self) -> bool:
        return self._interrupted

    def interrupt(self):
        """Call this when the tool returns to stop refinement."""
        self._interrupted = True

    @torch.no_grad()
    def run(
        self,
        past_key_values: tuple,
        generated_kv: Optional[list] = None,
        max_time_ms: Optional[float] = None,
    ) -> RefinementResult:
        """
        Run idle-time refinement. Blocks until interrupted or max_time reached.

        Args:
            past_key_values: current compressed KV cache
            generated_kv: KV pairs for tokens generated since last prefill
            max_time_ms: maximum time budget (for simulation). If None, runs
                         until interrupt() is called.

        Returns:
            RefinementResult with updated cache and timing info.
        """
        import torch
        self._interrupted = False
        start = time.perf_counter()

        # Set up timeout if specified
        if max_time_ms is not None:
            deadline = start + max_time_ms / 1000.0
            def check():
                return self._interrupted or time.perf_counter() > deadline
        else:
            check = self._check_interrupt

        # Phase 1: re-scoring (always runs first, <100ms)
        p1_start = time.perf_counter()
        kv = phase1_rescore(
            past_key_values=past_key_values,
            shadow_buffer=self.shadow_buffer,
            query_buffer=self.query_buffer,
            model=self.model,
            budget_per_layer=self.budget_per_layer,
            num_layers=self.num_layers,
            interrupt_flag=check,
        )
        p1_time = (time.perf_counter() - p1_start) * 1000
        p1_ran = True

        # Phase 2: progressive refresh (if time remains)
        p2_ran = False
        p2_layers = 0
        p2_start = time.perf_counter()

        if not check() and len(self.cpu_kv_store) > 0:
            p2_ran = True
            kv = phase2_refresh(
                past_key_values=kv,
                cpu_full_kv=self.cpu_kv_store.layers,
                generated_kv=generated_kv or [],
                query_buffer=self.query_buffer,
                model=self.model,
                budget_per_layer=self.budget_per_layer,
                num_layers=self.num_layers,
                interrupt_flag=check,
            )
            # Count how many layers were actually refreshed
            # (Phase 2 processes layers until interrupted)
            # We track this inside phase2_refresh via interrupt checks

        p2_time = (time.perf_counter() - p2_start) * 1000
        total_time = (time.perf_counter() - start) * 1000

        return RefinementResult(
            past_key_values=kv,
            phase1_ran=p1_ran,
            phase1_time_ms=p1_time,
            phase2_ran=p2_ran,
            phase2_layers_refreshed=p2_layers,
            phase2_time_ms=p2_time,
            total_time_ms=total_time,
            was_interrupted=self._interrupted,
        )


# Needed for type checking in run()
import torch
