from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from phases.phase15_real_repo_relevance_shift.scripts.build_phase15_manifest import _validate_sources
from phases.phase15_real_repo_relevance_shift.src.manifest import RepoSource


def test_manifest_cli_rejects_unpinned_sources_by_default(tmp_path: Path) -> None:
    source = RepoSource(repo_id="repo", repo_root=str(tmp_path))

    with pytest.raises(ValueError, match="missing frozen source fields"):
        _validate_sources([source], allow_unpinned_for_dev=False)

    _validate_sources([source], allow_unpinned_for_dev=True)


def test_manifest_cli_verifies_git_commit_and_archive(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    (repo_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True, capture_output=True)
    commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=repo_root)
    import hashlib

    source = RepoSource(
        repo_id="repo",
        repo_root=str(repo_root),
        repo_url="https://example.invalid/repo.git",
        commit_sha=commit_sha,
        license_spdx="MIT",
        archive_sha256=hashlib.sha256(archive).hexdigest(),
    )

    _validate_sources([source], allow_unpinned_for_dev=False)

    bad_source = RepoSource(
        repo_id="repo",
        repo_root=str(repo_root),
        repo_url="https://example.invalid/repo.git",
        commit_sha="0" * 40,
        license_spdx="MIT",
        archive_sha256=source.archive_sha256,
    )
    with pytest.raises(ValueError, match="HEAD mismatch"):
        _validate_sources([bad_source], allow_unpinned_for_dev=False)


def test_manifest_cli_rejects_this_repository() -> None:
    project_root = Path(__file__).resolve().parents[3]
    source = RepoSource(
        repo_id="self",
        repo_root=str(project_root),
        repo_url="https://github.com/arusli1/IdleKV",
        commit_sha="abc123",
        license_spdx="MIT",
        archive_sha256="deadbeef",
    )

    with pytest.raises(ValueError, match="must not come from this repository"):
        _validate_sources([source], allow_unpinned_for_dev=False)
