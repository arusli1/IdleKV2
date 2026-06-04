"""Unit tests for Phase 4 selector-quality diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from phases.phase4_eviction_buffer.src.buffer.quality import (
    evaluate_selection_quality,
    normalize_phase3_artifact_path,
)


class SelectionQualityTests(unittest.TestCase):
    def test_normalize_phase3_artifact_path_maps_old_repo_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            current_root = Path(tmpdir) / "phase3_eviction_logs"
            expected = current_root / "benchmark" / "run" / "VT4hop" / "snapkv" / "k512" / "ex001.json"
            normalized = normalize_phase3_artifact_path(
                "/home/ubuntu/IdleKV/phase_3_eviction_algorithm_validation/results/phase3_eviction_logs/"
                "benchmark/run/VT4hop/snapkv/k512/ex001.json",
                current_root,
            )
            self.assertEqual(normalized, expected)

    def test_quality_diagnostics_report_insufficient_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_root = root / "phase3_eviction_logs"
            raw_root = root / "phase3_raw_examples"
            task_dir = raw_root / "vt_4hop"
            log_dir = log_root / "benchmark" / "run" / "VT4hop" / "snapkv" / "k512"
            task_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "ex001.json").write_text(
                json.dumps({"evicted_positions": [10], "importance_scores": {"10": 0.2}}) + "\n",
                encoding="utf-8",
            )
            torch.save(torch.ones((2, 3, 4), dtype=torch.float32), log_dir / "ex001_qvecs.pt")
            payload = {
                "records": [
                    {
                        "method": "snapkv",
                        "k_budget": 512,
                        "correct": True,
                        "task_relevant_positions": [10],
                        "task_relevant_survived": [False],
                        "eviction_log_path": (
                            "/home/ubuntu/IdleKV/phase_3_eviction_algorithm_validation/results/phase3_eviction_logs/"
                            "benchmark/run/VT4hop/snapkv/k512/ex001.json"
                        ),
                        "q_vectors_path": (
                            "/home/ubuntu/IdleKV/phase_3_eviction_algorithm_validation/results/phase3_eviction_logs/"
                            "benchmark/run/VT4hop/snapkv/k512/ex001_qvecs.pt"
                        ),
                    }
                ]
            }
            (task_dir / "ex001.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")

            result = evaluate_selection_quality(
                log_root,
                raw_root,
                task_key="vt_4hop",
                method="snapkv",
                k_budget=512,
                max_examples=1,
            )

            self.assertEqual(result["status"], "insufficient_failures")
            self.assertEqual(result["matching_records"], 1)
            self.assertEqual(result["incorrect_records"], 0)


if __name__ == "__main__":
    unittest.main()
