#!/usr/bin/env python3
"""Backfill deterministic WrongEvent donor provenance into Phase 15 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from phases.phase15_real_repo_relevance_shift.scripts.run_phase15_manifest import (
    _donor_row,
    load_manifest,
)
from phases.phase15_real_repo_relevance_shift.src.manifest import stable_manifest_hash


def donor_metadata_for_manifest(manifest_rows) -> dict[str, dict[str, Any]]:
    """Return deterministic wrong-event donor metadata by example id."""
    metadata: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(manifest_rows):
        donor = _donor_row(manifest_rows, index)
        metadata[row.example_id] = {
            "wrong_event_donor_example_id": donor.example_id,
            "wrong_event_donor_repo_id": donor.repo.repo_id,
            "wrong_event_donor_answer": donor.answer,
            "wrong_event_donor_tool_event_sha256": hashlib.sha256(
                donor.tool_event.encode("utf-8")
            ).hexdigest(),
        }
    return metadata


def backfill_wrong_event_metadata(
    *,
    manifest_rows,
    artifact_payload: dict[str, Any],
) -> dict[str, Any]:
    """Add donor provenance to every artifact row without changing scores."""
    manifest_hash = stable_manifest_hash(manifest_rows)
    artifact_hash = str(artifact_payload.get("manifest_hash", ""))
    if artifact_hash != manifest_hash:
        raise ValueError(f"Artifact manifest hash mismatch: {artifact_hash} != {manifest_hash}")

    donor_by_example = donor_metadata_for_manifest(manifest_rows)
    updated = dict(artifact_payload)
    output_rows: list[dict[str, Any]] = []
    changed = 0
    for row in artifact_payload.get("rows", []):
        example_id = str(row.get("example_id", ""))
        if example_id not in donor_by_example:
            raise ValueError(f"Artifact row has unknown example_id: {example_id}")
        enriched = dict(row)
        for key, value in donor_by_example[example_id].items():
            if enriched.get(key) != value:
                enriched[key] = value
                changed += 1
        output_rows.append(enriched)
    updated["rows"] = output_rows
    updated["wrong_event_donor_metadata_backfilled"] = True
    updated["wrong_event_donor_metadata_changed_fields"] = changed
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows = load_manifest(Path(args.manifest))
    artifact_payload = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    updated = backfill_wrong_event_metadata(
        manifest_rows=manifest_rows,
        artifact_payload=artifact_payload,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(updated.get("rows", [])),
                "changed_fields": updated["wrong_event_donor_metadata_changed_fields"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
