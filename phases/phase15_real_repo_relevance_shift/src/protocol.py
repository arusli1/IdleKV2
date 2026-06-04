"""Protocol records for Phase 15 manifest-gated experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Phase15Protocol:
    """Frozen settings that must not change after a locked manifest is built."""

    model_dir: str = "models/Qwen2.5-7B-Instruct"
    tokenizer_dir: str = "models/Qwen2.5-7B-Instruct"
    context_tokens: int = 32768
    base_context_budget: int = 16384
    recency_window: int = 1024
    k_grid: tuple[int, ...] = (96, 192)
    primary_k: int = 192
    bootstrap_seed: int = 20260504
    conditions: tuple[str, ...] = (
        "A",
        "B",
        "B_match",
        "Random-K",
        "Oldest-K",
        "IdleKV-EventOnly-K",
        "StaleCue-K",
        "WrongEvent-K",
        "ToolFile-K",
        "AnchorWindow-K",
    )
    scoring_rule: str = "strict_identifier_first_line"
    source_task: str = "repodelta_edge"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["k_grid"] = list(self.k_grid)
        payload["conditions"] = list(self.conditions)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Phase15Protocol":
        return cls(
            model_dir=str(payload.get("model_dir", cls.model_dir)),
            tokenizer_dir=str(payload.get("tokenizer_dir", payload.get("model_dir", cls.tokenizer_dir))),
            context_tokens=int(payload.get("context_tokens", cls.context_tokens)),
            base_context_budget=int(payload.get("base_context_budget", cls.base_context_budget)),
            recency_window=int(payload.get("recency_window", cls.recency_window)),
            k_grid=tuple(int(value) for value in payload.get("k_grid", cls.k_grid)),
            primary_k=int(payload.get("primary_k", cls.primary_k)),
            bootstrap_seed=int(payload.get("bootstrap_seed", cls.bootstrap_seed)),
            conditions=tuple(str(value) for value in payload.get("conditions", cls.conditions)),
            scoring_rule=str(payload.get("scoring_rule", cls.scoring_rule)),
            source_task=str(payload.get("source_task", cls.source_task)),
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {})),
        )


def protocol_hash(protocol: Phase15Protocol) -> str:
    payload = json.dumps(protocol.to_dict(), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_protocol(path: Path | str) -> Phase15Protocol:
    return Phase15Protocol.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_protocol(path: Path | str, protocol: Phase15Protocol) -> None:
    Path(path).write_text(json.dumps(protocol.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
