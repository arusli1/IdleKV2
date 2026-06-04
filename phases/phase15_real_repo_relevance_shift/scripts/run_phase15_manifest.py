#!/usr/bin/env python3
"""Run Phase 15 RepoDelta-Edge from a frozen manifest.

This wrapper reuses the Phase 6 matched-budget cache machinery, but supplies
Phase 15's event-only repair cue and strict identifier scoring. It deliberately
loads the model once per process and prepares each manifest row exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_ROOT.parents[1]
for root in (PHASE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from phases.phase2_kv_cache.src.runtime import load_model, load_tokenizer  # noqa: E402
from phases.phase3_eviction.src.runtime import build_position_tracked_cache  # noqa: E402
from phases.phase6_repair.src.runner import Phase6Config, _run_one_split  # noqa: E402
from phases.phase15_real_repo_relevance_shift.src.manifest import (  # noqa: E402
    PHASE15_SPLIT_SPEC,
    encode_repair_signal,
    manifest_row_from_dict,
    split_prepared_from_manifest_row,
    stable_manifest_hash,
)
from phases.phase15_real_repo_relevance_shift.src.protocol import (  # noqa: E402
    Phase15Protocol,
    protocol_hash,
    read_protocol,
)
from phases.phase15_real_repo_relevance_shift.src.runner import with_wrong_event  # noqa: E402
from phases.phase15_real_repo_relevance_shift.src.scorer import score_identifier_prediction  # noqa: E402

SCHEMA_VERSION = "phase15-repodelta-edge-v1"
PHASE15_TO_PHASE6_CONDITION = {
    "A": "A",
    "B": "B",
    "B_match": "B_match",
    "IdleKV-EventOnly-K": "IdleKV",
    "Random-K": "Random-K",
    "Oldest-K": "Oldest-K",
    "StaleCue-K": "StaleQ-K",
    "WrongEvent-K": "WrongQ-K",
    "ToolFile-K": "ToolFile-K",
    "AnchorWindow-K": "AnchorWindow-K",
    "FileGatedIdleKV-K": "FileGatedIdleKV-K",
    "LexicalAnchor-K": "LexicalAnchor-K",
}
Q2_OUTPUT_SCORE_FIELDS = {
    "condition_a_output": "condition_a_score",
    "condition_b_output": "condition_b_score",
    "b_match_output": "b_match_score",
    "idlekv_output": "idlekv_score",
    "random_k_output": "random_k_score",
    "oldest_k_output": "oldest_k_score",
    "stale_q_k_output": "stale_q_k_score",
    "wrong_q_k_output": "wrong_q_k_score",
    "tool_file_k_output": "tool_file_k_score",
    "anchor_window_k_output": "anchor_window_k_score",
    "file_gated_idlekv_output": "file_gated_idlekv_score",
    "lexical_anchor_k_output": "lexical_anchor_k_score",
}


def load_manifest(path: Path, *, limit: int | None = None):
    """Load manifest rows in frozen file order."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(manifest_row_from_dict(json.loads(line)))
            if limit is not None and len(rows) >= int(limit):
                break
    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    return rows


def tokenizer_fingerprint(tokenizer) -> dict[str, Any]:
    """Record tokenizer metadata that affects manifest replay."""
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    }


def map_phase15_conditions(conditions: Iterable[str]) -> tuple[str, ...]:
    """Translate public Phase 15 condition names to the Phase 6 runner names."""
    mapped: list[str] = []
    for condition in conditions:
        name = str(condition)
        if name not in PHASE15_TO_PHASE6_CONDITION:
            raise ValueError(f"Unsupported Phase 15 condition: {name!r}")
        mapped.append(PHASE15_TO_PHASE6_CONDITION[name])
    return tuple(dict.fromkeys(mapped))


def needs_wrong_event_donor(conditions: Iterable[str]) -> bool:
    """Return whether the public Phase 15 condition list uses WrongEvent-K."""
    return "WrongEvent-K" in {str(condition) for condition in conditions}


def phase6_config_from_protocol(
    protocol: Phase15Protocol,
    *,
    stage: str,
    k_values: Iterable[int],
    conditions: Iterable[str],
    model_dir: Path | None,
    query_scoring_mode: str,
) -> Phase6Config:
    """Build the internal Phase 6 config without going through MQ-NIAH task aliases."""
    return Phase6Config(
        stage=str(stage),
        task="repodelta_edge",
        split_specs=(PHASE15_SPLIT_SPEC,),
        num_samples=0,
        context_length=int(protocol.context_tokens),
        dataset_seed_offset=0,
        k_values=tuple(dict.fromkeys(int(value) for value in k_values)),
        conditions=map_phase15_conditions(conditions),
        base_context_budget=int(protocol.base_context_budget),
        recency_window=int(protocol.recency_window),
        query_scoring_mode=str(query_scoring_mode),
        oracle_mode="gold_spans",
        wrong_query_mode="donor_q2",
        model_dir=str(model_dir or protocol.model_dir),
        initial_compressor="snapkv",
    )


def strict_rescore_row(row: dict[str, Any], *, q1_gold: str, q2_gold: str) -> dict[str, Any]:
    """Overwrite Phase 6 substring scores with strict identifier scores."""
    rescored = dict(row)
    q1_score = score_identifier_prediction(str(rescored.get("q1_output", "")), q1_gold)
    rescored["q1_score"] = round(q1_score.score, 6)
    rescored["q1_normalized_output"] = q1_score.normalized_prediction
    rescored["q1_failure_type"] = q1_score.failure_type
    for output_key, score_key in Q2_OUTPUT_SCORE_FIELDS.items():
        if output_key not in rescored:
            continue
        score = score_identifier_prediction(str(rescored.get(output_key, "")), q2_gold)
        rescored[score_key] = round(score.score, 6)
        prefix = score_key[: -len("_score")]
        rescored[f"{prefix}_normalized_output"] = score.normalized_prediction
        rescored[f"{prefix}_failure_type"] = score.failure_type
    return rescored


def _answer_overlap_fraction(
    kept_positions: Iterable[int],
    *,
    answer_token_start: int | None,
    answer_token_end: int | None,
) -> float:
    """Measure whether active-cache rows still include the gold answer tokens."""
    if answer_token_start is None or answer_token_end is None or answer_token_end <= answer_token_start:
        return 0.0
    answer_positions = set(range(int(answer_token_start), int(answer_token_end)))
    if not answer_positions:
        return 0.0
    kept = {int(position) for position in kept_positions}
    return len(answer_positions & kept) / len(answer_positions)


def add_answer_retention_fields(result: dict[str, Any], *, audit: Any) -> dict[str, Any]:
    """Record answer-token retention separately from whole-line span overlap."""
    enriched = dict(result)
    start = getattr(audit, "answer_token_start", None)
    end = getattr(audit, "answer_token_end", None)
    enriched["answer_token_start"] = start
    enriched["answer_token_end"] = end
    enriched["answer_token_count"] = getattr(audit, "answer_token_count", None)
    for prefix, field in (
        ("b", "b_kept_context_positions"),
        ("b_match", "b_match_kept_context_positions"),
        ("idlekv", "idlekv_selected_positions"),
        ("random_k", "random_k_selected_positions"),
        ("oldest_k", "oldest_k_selected_positions"),
        ("stale_q_k", "stale_q_k_selected_positions"),
        ("wrong_q_k", "wrong_q_k_selected_positions"),
        ("tool_file_k", "tool_file_k_selected_positions"),
        ("anchor_window_k", "anchor_window_k_selected_positions"),
        ("file_gated_idlekv", "file_gated_idlekv_selected_positions"),
        ("lexical_anchor_k", "lexical_anchor_k_selected_positions"),
    ):
        if field in enriched:
            enriched[f"{prefix}_answer_token_overlap_fraction"] = round(
                _answer_overlap_fraction(
                    enriched.get(field, []),
                    answer_token_start=start,
                    answer_token_end=end,
                ),
                6,
            )
    return enriched


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact summary by K for smoke gates."""
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_k.setdefault(int(row["k"]), []).append(row)
    summary: dict[str, Any] = {}
    score_keys = (
        "condition_a_score",
        "condition_b_score",
        "b_match_score",
        "idlekv_score",
        "random_k_score",
        "oldest_k_score",
        "stale_q_k_score",
        "wrong_q_k_score",
        "tool_file_k_score",
        "anchor_window_k_score",
        "file_gated_idlekv_score",
        "lexical_anchor_k_score",
        "cue_only_score",
    )
    for k, group in sorted(by_k.items()):
        payload: dict[str, Any] = {"n_examples": len(group)}
        for score_key in score_keys:
            values = [float(row[score_key]) for row in group if score_key in row]
            if values:
                payload[f"mean_{score_key[:-6]}"] = round(sum(values) / len(values), 6)
        for overlap_key in ("b_answer_token_overlap_fraction", "b_match_answer_token_overlap_fraction"):
            values = [float(row[overlap_key]) for row in group if overlap_key in row]
            if values:
                payload[f"mean_{overlap_key}"] = round(sum(values) / len(values), 6)
                payload[f"max_{overlap_key}"] = round(max(values), 6)
        if "idlekv_score" in group[0]:
            payload["mean_idlekv_minus_b_match"] = round(
                sum(float(row["idlekv_score"]) - float(row["b_match_score"]) for row in group) / len(group),
                6,
            )
        if "cue_only_score" in group[0]:
            payload["mean_full_minus_cue_only"] = round(
                sum(float(row["condition_a_score"]) - float(row["cue_only_score"]) for row in group) / len(group),
                6,
            )
        summary[f"k{k}"] = payload
    return summary


def run_cue_only(model, tokenizer, split) -> tuple[str, float, dict[str, Any]]:
    """Decode the final event+Q2 prompt with no repository context."""
    start = time.perf_counter()
    question_ids = split.q2_prepared.question_ids.to(model.device)
    with torch.inference_mode():
        outputs = model(input_ids=question_ids, use_cache=True, logits_to_keep=1)
        cache = outputs.past_key_values
        generated_ids = [outputs.logits[0, -1].argmax()]
        stop_ids = model.generation_config.eos_token_id
        if not isinstance(stop_ids, list):
            stop_ids = [stop_ids]
        for _step in range(int(split.q2_prepared.example.max_new_tokens) - 1):
            outputs = model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if int(new_id.item()) in stop_ids:
                break
    output = tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)
    elapsed = time.perf_counter() - start
    score = score_identifier_prediction(output, str(split.q2_prepared.example.outputs[0]))
    return output, float(score.score), {
        "cue_only_generation_s": round(elapsed, 6),
        "cue_only_normalized_output": score.normalized_prediction,
        "cue_only_failure_type": score.failure_type,
    }


def _donor_row(rows, index: int):
    """Choose a wrong-event donor that cannot be the same repo event in disguise."""
    current = rows[index]
    for offset in range(1, len(rows)):
        donor = rows[(index + offset) % len(rows)]
        if donor.example_id == current.example_id:
            continue
        if donor.repo.repo_id == current.repo.repo_id:
            continue
        if donor.tool_event == current.tool_event:
            continue
        if donor.answer == current.answer:
            continue
        return donor
    raise ValueError(
        "WrongEvent-K requires a donor from a different repo with a different event and answer."
    )


def validate_wrong_event_donors(rows) -> None:
    """Fail before model load if any row lacks a valid wrong-event donor."""
    for index, _row in enumerate(rows):
        _donor_row(rows, index)


def run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_protocol(Path(args.protocol)) if args.protocol else Phase15Protocol()
    tokenizer_dir = Path(args.tokenizer_dir or protocol.tokenizer_dir)
    k_values = tuple(args.k if args.k else protocol.k_grid)
    conditions = tuple(args.conditions if args.conditions else protocol.conditions)
    rows = load_manifest(Path(args.manifest), limit=args.limit)
    uses_wrong_event = needs_wrong_event_donor(conditions)
    if uses_wrong_event:
        validate_wrong_event_donors(rows)
    config = phase6_config_from_protocol(
        protocol,
        stage=args.stage,
        k_values=k_values,
        conditions=conditions,
        model_dir=args.model_dir,
        query_scoring_mode=args.query_scoring_mode,
    )
    if args.dry_run:
        tokenizer = load_tokenizer(tokenizer_dir)
        prepared_count = 0
        for row in rows:
            split_prepared_from_manifest_row(row, tokenizer)
            prepared_count += 1
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "prepared_rows": prepared_count,
            "manifest_hash": stable_manifest_hash(rows),
            "protocol_hash": protocol_hash(protocol),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "conditions": conditions,
            "k_values": list(k_values),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    overall_start = time.perf_counter()
    model = load_model(Path(config.model_dir))
    tokenizer = load_tokenizer(tokenizer_dir)
    result_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        print(
            f"[phase15] row {index + 1}/{len(rows)} "
            f"{row.repo.repo_id} answer={row.answer} conditions={','.join(conditions)} k={','.join(str(k) for k in k_values)}",
            file=sys.stderr,
            flush=True,
        )
        split = split_prepared_from_manifest_row(row, tokenizer)
        wrong_event_donor = _donor_row(rows, index) if uses_wrong_event else None
        wrong_row = with_wrong_event(row, wrong_event_donor) if wrong_event_donor is not None else None
        full_cache = build_position_tracked_cache(model, split.q1_prepared.context_ids)
        cue_only_output, cue_only_score, cue_only_meta = run_cue_only(model, tokenizer, split)
        example_rows = _run_one_split(
            model=model,
            tokenizer=tokenizer,
            config=config,
            split=split,
            full_cache=full_cache,
            index=index,
            wrong_q2_question_ids=(
                encode_repair_signal(tokenizer, wrong_row, mode="wrong_event")
                if wrong_row is not None
                else None
            ),
            repair_question_ids=encode_repair_signal(tokenizer, row, mode="event_only"),
            stale_question_ids=encode_repair_signal(tokenizer, row, mode="stale_event"),
            tool_file_q2_path=str(row.q2.get("path", "")),
        )
        for result in example_rows:
            rescored = strict_rescore_row(
                result,
                q1_gold=str(row.q1["answer"]),
                q2_gold=row.answer,
            )
            rescored = add_answer_retention_fields(rescored, audit=row.audit)
            metadata = {
                "example_id": row.example_id,
                "task": "repodelta_edge",
                "repo_id": row.repo.repo_id,
                "repo_commit_sha": row.repo.commit_sha,
                "source_task": row.source_task,
                "repair_signal_mode": "event_only",
                "decode_prompt_mode": "event_plus_q2",
                "q2_path": str(row.q2.get("path", "")),
                "q2_line_no": int(row.q2.get("line_no", 0)),
                "q2_answer": row.answer,
                "cue_only_output": cue_only_output,
                "cue_only_score": round(cue_only_score, 6),
                **cue_only_meta,
                "phase15_conditions": list(conditions),
                "phase15_manifest_audit": row.audit.to_dict(),
            }
            if wrong_event_donor is not None:
                metadata.update(
                    {
                        "wrong_event_donor_example_id": wrong_event_donor.example_id,
                        "wrong_event_donor_repo_id": wrong_event_donor.repo.repo_id,
                        "wrong_event_donor_answer": wrong_event_donor.answer,
                        "wrong_event_donor_tool_event_sha256": hashlib.sha256(
                            wrong_event_donor.tool_event.encode("utf-8")
                        ).hexdigest(),
                    }
                )
            rescored.update(metadata)
            result_rows.append(rescored)
        print(
            f"[phase15] row {index + 1}/{len(rows)} done; accumulated_result_rows={len(result_rows)}",
            file=sys.stderr,
            flush=True,
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dry_run": False,
        "stage": args.stage,
        "manifest_path": str(Path(args.manifest)),
        "manifest_hash": stable_manifest_hash(rows),
        "protocol_hash": protocol_hash(protocol),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "protocol": protocol.to_dict(),
        "phase6_config": {
            "k_values": list(config.k_values),
            "conditions": list(config.conditions),
            "base_context_budget": config.base_context_budget,
            "recency_window": config.recency_window,
            "query_scoring_mode": config.query_scoring_mode,
            "model_dir": config.model_dir,
            "tokenizer_dir": str(tokenizer_dir),
        },
        "rows": result_rows,
        "summary": summarize_rows(result_rows),
        "wall_s": round(time.perf_counter() - overall_start, 6),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Frozen Phase 15 manifest JSONL.")
    parser.add_argument("--protocol", help="Frozen Phase 15 protocol JSON.")
    parser.add_argument("--output", required=True, help="Output JSON artifact path.")
    parser.add_argument("--stage", default="ability_smoke")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", nargs="+", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--query-scoring-mode", choices=("proxy", "exact_q"), default="exact_q")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run_manifest(parse_args())
    print(json.dumps({key: payload[key] for key in ("dry_run", "manifest_hash", "protocol_hash")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
