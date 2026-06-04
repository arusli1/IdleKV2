"""Build a CPU-audited Phase 15 manifest from pinned repository snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from transformers import AutoTokenizer

from phases.phase15_real_repo_relevance_shift.src.manifest import (
    RepoSource,
    build_phase15_prepared_example,
    stable_manifest_hash,
)
from phases.phase15_real_repo_relevance_shift.src.protocol import (
    Phase15Protocol,
    protocol_hash,
    read_protocol,
)


def _read_registry(path: Path) -> list[RepoSource]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    repos = payload.get("repos", payload if isinstance(payload, list) else None)
    if not isinstance(repos, list):
        raise ValueError("Registry must be a list or an object with a 'repos' list.")
    sources = [RepoSource(**repo) for repo in repos]
    if not sources:
        raise ValueError("Registry did not contain any repos.")
    return sources


def _validate_sources(sources: list[RepoSource], *, allow_unpinned_for_dev: bool) -> None:
    project_root = Path(__file__).resolve().parents[3]
    for source in sources:
        repo_root = Path(source.repo_root).resolve()
        if repo_root == project_root or project_root in repo_root.parents:
            raise ValueError(f"Phase 15 sources must not come from this repository: {source.repo_root}")
        missing = [
            field
            for field, value in {
                "repo_url": source.repo_url,
                "commit_sha": source.commit_sha,
                "license_spdx": source.license_spdx,
                "archive_sha256": source.archive_sha256,
            }.items()
            if not value
        ]
        if missing and not allow_unpinned_for_dev:
            raise ValueError(
                f"Repo {source.repo_id!r} is missing frozen source fields: {', '.join(missing)}. "
                "Use --allow-unpinned-for-dev only for local generator debugging."
            )
        if not missing and not allow_unpinned_for_dev:
            _verify_git_source(source)


def _git_output(repo_root: Path, *args: str, binary: bool = False) -> str | bytes:
    cmd = ["git", "-C", str(repo_root), *args]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Git source verification failed for {repo_root}: {cmd!r}") from exc
    return completed.stdout if binary else completed.stdout.decode("utf-8").strip()


def _verify_git_source(source: RepoSource) -> None:
    """Verify that an on-disk source checkout matches the frozen registry fields."""
    repo_root = Path(source.repo_root).resolve()
    if not (repo_root / ".git").exists():
        raise ValueError(f"Repo {source.repo_id!r} is not a git checkout: {repo_root}")
    head = str(_git_output(repo_root, "rev-parse", "HEAD"))
    if head != source.commit_sha:
        raise ValueError(
            f"Repo {source.repo_id!r} HEAD mismatch: registry={source.commit_sha}, checkout={head}"
        )
    dirty = str(_git_output(repo_root, "status", "--porcelain", "--untracked-files=no"))
    if dirty:
        raise ValueError(f"Repo {source.repo_id!r} has tracked worktree changes.")
    archive = _git_output(repo_root, "archive", "--format=tar", "HEAD", binary=True)
    digest = hashlib.sha256(archive).hexdigest()
    if digest != source.archive_sha256:
        raise ValueError(
            f"Repo {source.repo_id!r} archive SHA256 mismatch: registry={source.archive_sha256}, checkout={digest}"
        )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer_fingerprint(tokenizer) -> dict[str, Any]:
    """Record enough tokenizer metadata to detect prompt/tokenization drift."""
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_protocol(args.protocol) if args.protocol else Phase15Protocol()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer or protocol.tokenizer_dir), trust_remote_code=True)
    registry_path = Path(args.registry)
    repos = _read_registry(registry_path)
    _validate_sources(repos, allow_unpinned_for_dev=bool(args.allow_unpinned_for_dev))
    max_k = max(int(value) for value in protocol.k_grid)
    min_context_tokens = (
        int(args.min_context_tokens)
        if args.min_context_tokens is not None
        else max(int(protocol.base_context_budget) + max_k, int(protocol.context_tokens * 0.75))
    )
    prepared_rows = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    allowed_edge_types = set(args.edge_types) if args.edge_types else None
    for repo in repos:
        repo_count = 0
        index = 0
        attempts = 0
        while repo_count < int(args.examples_per_repo) and attempts < int(args.max_attempts_per_repo):
            attempts += 1
            try:
                prepared = build_phase15_prepared_example(
                    repo=repo,
                    index=index,
                    tokenizer=tokenizer,
                    target_context_length=int(args.target_context_length),
                    max_context_tokens=int(protocol.context_tokens),
                    recency_window=int(protocol.recency_window),
                    k_max=max_k,
                    min_context_tokens=min_context_tokens,
                    seed_offset=int(args.seed_offset),
                    max_files=int(args.max_files),
                    allowed_edge_types=allowed_edge_types,
                    require_project_local_targets=not bool(args.allow_nonlocal_targets),
                    max_answer_boundary_occurrences=int(args.max_answer_boundary_occurrences),
                    min_q2_depth_fraction=float(args.min_q2_depth_fraction),
                    max_q2_depth_fraction=float(args.max_q2_depth_fraction),
                )
            except Exception as exc:  # noqa: BLE001 - manifest builder should record bad rows.
                failures.append({"repo_id": repo.repo_id, "index": index, "error": repr(exc)})
                index += 1
                continue
            if prepared.row.audit.passed or args.include_failed:
                prepared_rows.append(prepared.row)
                repo_count += 1
                if prepared.row.audit.warnings:
                    warnings.append(
                        {
                            "repo_id": repo.repo_id,
                            "index": index,
                            "example_id": prepared.row.example_id,
                            "warnings": list(prepared.row.audit.warnings),
                            "answer_isolated_token_sequence_occurrences": (
                                prepared.row.audit.answer_isolated_token_sequence_occurrences
                            ),
                        }
                    )
            else:
                failures.append(
                    {
                        "repo_id": repo.repo_id,
                        "index": index,
                        "example_id": prepared.row.example_id,
                        "flags": list(prepared.row.audit.flags),
                    }
                )
            index += 1
    manifest_rows = [row.to_dict() for row in prepared_rows]
    _write_jsonl(Path(args.output), manifest_rows)
    summary = {
        "rows": len(prepared_rows),
        "repos": sorted({row.repo.repo_id for row in prepared_rows}),
        "manifest_hash": stable_manifest_hash(prepared_rows),
        "protocol_hash": protocol_hash(protocol),
        "registry_hash": file_sha256(registry_path),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "min_context_tokens": min_context_tokens,
        "allowed_edge_types": sorted(allowed_edge_types) if allowed_edge_types else "all",
        "require_project_local_targets": not bool(args.allow_nonlocal_targets),
        "max_answer_boundary_occurrences": int(args.max_answer_boundary_occurrences),
        "min_q2_depth_fraction": float(args.min_q2_depth_fraction),
        "max_q2_depth_fraction": float(args.max_q2_depth_fraction),
        "failures": failures[: int(args.max_failures_in_summary)],
        "failure_count": len(failures),
        "warnings": warnings[: int(args.max_failures_in_summary)],
        "warning_count": len(warnings),
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="JSON repo registry path.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--summary", help="Optional summary JSON path.")
    parser.add_argument("--protocol", help="Optional phase15_protocol.json path.")
    parser.add_argument("--tokenizer", help="Tokenizer path override.")
    parser.add_argument("--examples-per-repo", type=int, default=4)
    parser.add_argument("--max-attempts-per-repo", type=int, default=64)
    parser.add_argument("--target-context-length", type=int, default=80_000)
    parser.add_argument("--min-context-tokens", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=24)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--edge-types",
        nargs="+",
        choices=("callsite_leaf_callee", "class_base_identifier", "exception_identifier"),
        default=None,
        help="Optional candidate edge-type allow-list.",
    )
    parser.add_argument(
        "--allow-nonlocal-targets",
        action="store_true",
        help="Allow targets that are not also declarations in the sampled repository context.",
    )
    parser.add_argument(
        "--max-answer-boundary-occurrences",
        type=int,
        default=1,
        help="Reject rows where the answer appears more than this many times in rendered context.",
    )
    parser.add_argument(
        "--min-q2-depth-fraction",
        type=float,
        default=0.35,
        help="Reject rows whose Q2 source line starts before this context depth fraction.",
    )
    parser.add_argument(
        "--max-q2-depth-fraction",
        type=float,
        default=0.86,
        help="Reject rows whose Q2 source line starts after this context depth fraction.",
    )
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--allow-unpinned-for-dev", action="store_true")
    parser.add_argument("--max-failures-in-summary", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    summary = build_manifest(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
