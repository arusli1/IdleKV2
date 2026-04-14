"""
Tiered idle-time scheduler.

Decides which refinement operations to run based on estimated idle duration.
Both phases are anytime-safe: they can be interrupted at any layer boundary.

Tier 0 (<100ms): Phase 1 re-scoring only
Tier 1 (100ms-2s): Phase 1 + Phase 2 (progressive, as many layers as fit)
"""

import time
import threading
import torch
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from idlekv.core.shadow_buffer import ShadowBuffer
from idlekv.core.query_buffer import QueryBuffer
from idlekv.core.phase1_rescore import phase1_rescore
from idlekv.core.phase2_refresh import phase2_refresh, FullKVStore


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
        full_kv_store: FullKVStore,
        model,
        budget_per_layer: int,
        num_layers: int,
    ):
        self.shadow_buffer = shadow_buffer
        self.query_buffer = query_buffer
        self.full_kv_store = full_kv_store
        self.model = model
        self.budget_per_layer = budget_per_layer
        self.num_layers = num_layers

        # Fix Bug 5: Use threading.Event for thread safety
        self._stop_event = threading.Event()

    def _check_interrupt(self) -> bool:
        return self._stop_event.is_set()

    def interrupt(self):
        """Call this when the tool returns to stop refinement."""
        self._stop_event.set()

    @staticmethod
    def normalize_phases(phases: Union[str, int, None]) -> Tuple[bool, bool, str]:
        """
        Normalize a phase selection into explicit booleans.

        Supported values:
          - `1`, `"1"`, `"phase1"`
          - `2`, `"2"`, `"phase2"`
          - `"1+2"`, `"both"`, `None`
        """
        if phases is None:
            return True, True, "1+2"

        if isinstance(phases, int):
            key = str(phases)
        else:
            key = str(phases).strip().lower()

        if key in {"1", "phase1"}:
            return True, False, "1"
        if key in {"2", "phase2"}:
            return False, True, "2"
        if key in {"1+2", "both", "all"}:
            return True, True, "1+2"

        raise ValueError(f"Unsupported phase selection: {phases!r}")

    @torch.no_grad()
    def run(
        self,
        past_key_values: tuple,
        generated_kv: Optional[list] = None,
        max_time_ms: Optional[float] = None,
        num_generated: int = 0,
        phases: Union[str, int, None] = "1+2",
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
        self._stop_event.clear()  # Reset the event
        start = time.perf_counter()

        # Set up timeout if specified
        if max_time_ms is not None:
            deadline = start + max_time_ms / 1000.0
            def check():
                return self._stop_event.is_set() or time.perf_counter() > deadline
        else:
            check = self._check_interrupt

        run_phase1, run_phase2, _ = self.normalize_phases(phases)

        kv = past_key_values
        p1_ran = False
        p1_time = 0.0
        if run_phase1:
            p1_start = time.perf_counter()
            kv = phase1_rescore(
                past_key_values=past_key_values,
                shadow_buffer=self.shadow_buffer,
                query_buffer=self.query_buffer,
                model=self.model,
                budget_per_layer=self.budget_per_layer,
                num_layers_arg=self.num_layers,
                interrupt_flag=check,
                num_generated=num_generated,
            )
            p1_time = (time.perf_counter() - p1_start) * 1000
            p1_ran = True

        # Phase 2: progressive refresh (if time remains)
        p2_ran = False
        p2_layers = 0
        p2_time = 0.0
        phase2_allowed = run_phase2 and (max_time_ms is None or max_time_ms > 100.0)
        if phase2_allowed and not check() and len(self.full_kv_store) > 0:
            p2_start = time.perf_counter()
            p2_ran = True
            kv, p2_layers = phase2_refresh(
                past_key_values=kv,
                full_kv_store=self.full_kv_store,
                generated_kv=generated_kv or [],
                query_buffer=self.query_buffer,
                model=self.model,
                budget_per_layer=self.budget_per_layer,
                interrupt_flag=check,
            )
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
            was_interrupted=self._stop_event.is_set(),
        )
