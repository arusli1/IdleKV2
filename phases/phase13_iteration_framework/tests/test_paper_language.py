from __future__ import annotations

from pathlib import Path
import re


REPO_DIR = Path(__file__).resolve().parents[3]
MAIN_TEX = REPO_DIR / "paper" / "main.tex"


def _paper_body_before_bibliography() -> str:
    text = MAIN_TEX.read_text(encoding="utf-8")
    return text.split(r"\begin{thebibliography}", maxsplit=1)[0].lower()


def test_main_paper_body_avoids_internal_run_vocabulary() -> None:
    body = _paper_body_before_bibliography()
    forbidden_patterns = (
        "phase 7",
        "phase 8",
        "phase 9",
        "phase 10",
        "phase 11",
        "phase 12",
        "phase 13",
        "matched footprint",
        "h2o-style",
        "streamingllm-style",
        "gold-k",
        "oracle-k",
        " oracle ",
        "clean suite",
        r"\bbridge\b",
        r"\bextension\b",
    )

    found = [pattern for pattern in forbidden_patterns if re.search(pattern, body)]

    assert found == []
