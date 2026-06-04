"""Python AST extraction for RepoDelta-Edge candidates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

COMMON_CALLEES = {
    "Exception",
    "ImportError",
    "IOError",
    "KeyError",
    "RuntimeError",
    "SystemExit",
    "TypeError",
    "ValueError",
    "add",
    "append",
    "array",
    "asarray",
    "close",
    "copy",
    "clear",
    "extend",
    "fixture",
    "format",
    "get",
    "items",
    "join",
    "keys",
    "len",
    "list",
    "loads",
    "open",
    "print",
    "range",
    "read",
    "run",
    "set",
    "setattr",
    "split",
    "startswith",
    "str",
    "update",
    "urandom",
    "values",
    "write",
}

IGNORED_DIR_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "doc",
    "docs",
    "example",
    "examples",
    "externals",
    "node_modules",
    "site-packages",
    "test",
    "tests",
    "testing",
    "vendor",
    "vendored",
}


@dataclass(frozen=True)
class DeclarationCandidate:
    """Simple exact identifier fact used for Q1 ability/biasing turns."""

    path: str
    kind: str
    answer: str
    line_no: int
    line_text: str
    answer_start_in_line: int
    answer_end_in_line: int


@dataclass(frozen=True)
class EdgeCandidate:
    """One callsite/cue candidate for RepoDelta-Edge."""

    path: str
    edge_type: str
    answer_kind: str
    answer: str
    line_no: int
    line_text: str
    answer_start_in_line: int
    answer_end_in_line: int
    anchor_name: str
    anchor_line_no: int


def read_text(path: Path) -> str | None:
    """Read source text with conservative encoding fallbacks."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return None
    except OSError:
        return None


def iter_python_files(repo_root: Path | str, *, max_bytes: int = 80_000) -> list[Path]:
    """Return deterministic Python files from a repository tree."""
    root = Path(repo_root)
    paths: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        try:
            rel = path.relative_to(root)
            if any(part in IGNORED_DIR_PARTS or part.startswith(".") for part in rel.parts):
                continue
            if path.stat().st_size > int(max_bytes):
                continue
        except OSError:
            continue
        paths.append(path)
    return paths


def number_file_card(rel_path: str, text: str) -> str:
    """Render a line-numbered source file card."""
    numbered = "\n".join(
        f"{line_no:04d}: {line}" for line_no, line in enumerate(text.rstrip().splitlines(), start=1)
    )
    return f"--- FILE: {rel_path} ---\n{numbered}\n"


def line_char_bounds(card_start: int, card: str, line_no: int) -> tuple[int, int]:
    """Return raw-context char bounds for a source line inside a numbered card."""
    offset = 0
    lines = card.splitlines(keepends=True)
    for current, line in enumerate(lines, start=1):
        next_offset = offset + len(line)
        if current == int(line_no) + 1:
            return card_start + offset, card_start + next_offset
        offset = next_offset
    raise ValueError(f"Line {line_no} is outside generated card.")


def _is_good_identifier(name: str) -> bool:
    if len(name) < 4 or len(name) > 64:
        return False
    if name.startswith("__") and name.endswith("__"):
        return False
    if name.startswith("test_"):
        return False
    if name.startswith("assert"):
        return False
    return name not in COMMON_CALLEES


def _line_col_span(lines: list[str], lineno: int, start_col: int, end_col: int) -> tuple[str, int, int]:
    line = lines[int(lineno) - 1]
    return line, int(start_col), int(end_col)


def extract_declarations(path: str, text: str) -> list[DeclarationCandidate]:
    """Extract simple Python declaration identifiers for Q1."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    candidates: list[DeclarationCandidate] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            answer = str(node.name)
            if not _is_good_identifier(answer):
                continue
            line = lines[int(node.lineno) - 1]
            start = line.find(answer, int(node.col_offset))
            if start < 0:
                continue
            end = start + len(answer)
            candidates.append(
                DeclarationCandidate(
                    path=path,
                    kind="class" if isinstance(node, ast.ClassDef) else "function",
                    answer=answer,
                    line_no=int(node.lineno),
                    line_text=line,
                    answer_start_in_line=start,
                    answer_end_in_line=end,
                )
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                answer = str(target.id)
                if not answer.isupper() or not _is_good_identifier(answer):
                    continue
                line, start, end = _line_col_span(lines, int(target.lineno), int(target.col_offset), int(target.end_col_offset or target.col_offset + len(answer)))
                candidates.append(
                    DeclarationCandidate(
                        path=path,
                        kind="constant",
                        answer=answer,
                        line_no=int(target.lineno),
                        line_text=line,
                        answer_start_in_line=start,
                        answer_end_in_line=end,
                    )
                )
    return sorted(candidates, key=lambda item: (item.path, item.line_no, item.answer))


class _CallsiteVisitor(ast.NodeVisitor):
    def __init__(self, lines: list[str], call_counts_by_line: dict[int, int]) -> None:
        self.lines = lines
        self.call_counts_by_line = call_counts_by_line
        self.stack: list[tuple[str, int]] = []
        self.edges: list[tuple[ast.Call, str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append((str(node.name), int(node.lineno)))
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.stack.append((str(node.name), int(node.lineno)))
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self.stack:
            anchor_name, anchor_line = self.stack[-1]
            self.edges.append((node, anchor_name, anchor_line))
        self.generic_visit(node)


def _leaf_callee_span(func: ast.AST, line: str) -> tuple[str, int, int] | None:
    if isinstance(func, ast.Name):
        if func.end_col_offset is None:
            return None
        return str(func.id), int(func.col_offset), int(func.end_col_offset)
    if isinstance(func, ast.Attribute):
        if func.end_col_offset is None:
            return None
        answer = str(func.attr)
        start = line.rfind(answer, 0, int(func.end_col_offset))
        if start < int(func.col_offset):
            return None
        return answer, start, start + len(answer)
    return None


def _leaf_expr_span(expr: ast.AST, line: str) -> tuple[str, int, int] | None:
    if isinstance(expr, ast.Name):
        if expr.end_col_offset is None:
            return None
        return str(expr.id), int(expr.col_offset), int(expr.end_col_offset)
    if isinstance(expr, ast.Attribute):
        if expr.end_col_offset is None:
            return None
        answer = str(expr.attr)
        start = line.rfind(answer, 0, int(expr.end_col_offset))
        if start < int(expr.col_offset):
            return None
        return answer, start, start + len(answer)
    return None


def _exception_expr_span(expr: ast.AST | None, line: str) -> tuple[str, int, int] | None:
    if expr is None:
        return None
    if isinstance(expr, ast.Call):
        return _leaf_expr_span(expr.func, line)
    return _leaf_expr_span(expr, line)


def _assert_raises_exception_span(call: ast.Call, line: str) -> tuple[str, int, int] | None:
    callee = _leaf_callee_span(call.func, line)
    if callee is None:
        return None
    callee_name, _start, _end = callee
    if callee_name not in {"assertRaises", "assertRaisesRegex", "assertRaisesMessage", "raises"}:
        return None
    if not call.args:
        return None
    return _exception_expr_span(call.args[0], line)


class _ExceptionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[tuple[str, int]] = []
        self.nodes: list[tuple[ast.AST, str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append((str(node.name), int(node.lineno)))
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.stack.append((str(node.name), int(node.lineno)))
        self.generic_visit(node)
        self.stack.pop()

    def _record(self, node: ast.AST) -> None:
        anchor_name, anchor_line = self.stack[-1] if self.stack else ("module_scope", 1)
        self.nodes.append((node, anchor_name, anchor_line))

    def visit_Raise(self, node: ast.Raise) -> None:
        self._record(node)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._record(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._record(node)
        self.generic_visit(node)


def extract_exception_edges(path: str, text: str) -> list[EdgeCandidate]:
    """Extract raise/except/assert-raises -> exception identifier candidates."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    visitor = _ExceptionVisitor()
    visitor.visit(tree)
    candidates: list[EdgeCandidate] = []
    for node, anchor_name, anchor_line in visitor.nodes:
        span: tuple[str, int, int] | None = None
        edge_kind = ""
        line_no = int(getattr(node, "lineno", -1))
        if line_no < 1 or line_no > len(lines):
            continue
        line = lines[line_no - 1]
        if isinstance(node, ast.Raise):
            span = _exception_expr_span(node.exc, line)
            edge_kind = "raised_exception_identifier"
        elif isinstance(node, ast.ExceptHandler):
            span = _exception_expr_span(node.type, line)
            edge_kind = "handled_exception_identifier"
        elif isinstance(node, ast.Call):
            span = _assert_raises_exception_span(node, line)
            edge_kind = "expected_exception_identifier"
        if span is None:
            continue
        answer, start, end = span
        if not _is_good_identifier(answer) or answer == anchor_name:
            continue
        candidates.append(
            EdgeCandidate(
                path=path,
                edge_type="exception_identifier",
                answer_kind=edge_kind,
                answer=answer,
                line_no=line_no,
                line_text=line,
                answer_start_in_line=int(start),
                answer_end_in_line=int(end),
                anchor_name=anchor_name,
                anchor_line_no=int(anchor_line),
            )
        )
    return sorted(candidates, key=lambda item: (item.path, item.line_no, item.answer))


def extract_class_base_edges(path: str, text: str) -> list[EdgeCandidate]:
    """Extract class declaration -> leaf base-class identifier candidates."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    candidates: list[EdgeCandidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or len(node.bases) != 1:
            continue
        base = node.bases[0]
        if int(getattr(base, "lineno", -1)) < 1 or int(getattr(base, "lineno", -1)) > len(lines):
            continue
        line = lines[int(base.lineno) - 1]
        leaf = _leaf_expr_span(base, line)
        if leaf is None:
            continue
        answer, start, end = leaf
        anchor_name = str(node.name)
        if not _is_good_identifier(answer) or answer == anchor_name:
            continue
        candidates.append(
            EdgeCandidate(
                path=path,
                edge_type="class_base_identifier",
                answer_kind="base_class_identifier",
                answer=answer,
                line_no=int(base.lineno),
                line_text=line,
                answer_start_in_line=int(start),
                answer_end_in_line=int(end),
                anchor_name=anchor_name,
                anchor_line_no=int(node.lineno),
            )
        )
    return sorted(candidates, key=lambda item: (item.path, item.line_no, item.answer))


def extract_callsite_edges(path: str, text: str) -> list[EdgeCandidate]:
    """Extract single-line callsite -> leaf callee identifier candidates."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    call_counts: dict[int, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_counts[int(node.lineno)] = call_counts.get(int(node.lineno), 0) + 1

    visitor = _CallsiteVisitor(lines, call_counts)
    visitor.visit(tree)
    candidates: list[EdgeCandidate] = []
    for node, anchor_name, anchor_line in visitor.edges:
        if int(getattr(node, "lineno", -1)) != int(getattr(node, "end_lineno", node.lineno)):
            continue
        if call_counts.get(int(node.lineno), 0) != 1:
            continue
        if int(node.lineno) < 1 or int(node.lineno) > len(lines):
            continue
        line = lines[int(node.lineno) - 1]
        leaf = _leaf_callee_span(node.func, line)
        if leaf is None:
            continue
        answer, start, end = leaf
        if not _is_good_identifier(answer) or answer == anchor_name:
            continue
        candidates.append(
            EdgeCandidate(
                path=path,
                edge_type="callsite_leaf_callee",
                answer_kind="callee_identifier",
                answer=answer,
                line_no=int(node.lineno),
                line_text=line,
                answer_start_in_line=int(start),
                answer_end_in_line=int(end),
                anchor_name=anchor_name,
                anchor_line_no=int(anchor_line),
            )
        )
    return sorted(candidates, key=lambda item: (item.path, item.line_no, item.answer))


def boundary_occurrences(text: str, identifier: str) -> list[tuple[int, int]]:
    """Find identifier occurrences bounded by non-identifier characters."""
    import re

    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])")
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def first_pair_by_files(
    q1_candidates: Iterable[DeclarationCandidate],
    q2_candidates: Iterable[EdgeCandidate],
) -> tuple[DeclarationCandidate, EdgeCandidate]:
    """Choose the first deterministic Q1/Q2 pair from different files."""
    q1_list = list(q1_candidates)
    q2_list = sorted(
        q2_candidates,
        key=lambda item: (
            {
                "callsite_leaf_callee": 0,
                "class_base_identifier": 1,
                "exception_identifier": 2,
            }.get(item.edge_type, 3),
            item.path,
            item.line_no,
            item.answer,
        ),
    )
    for q2 in q2_list:
        for q1 in q1_list:
            if q1.path != q2.path and q1.answer != q2.answer:
                return q1, q2
    raise ValueError("Need Q1 declaration and Q2 edge candidates from different files.")
