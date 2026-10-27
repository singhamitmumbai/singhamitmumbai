"""Workspace code search — lexical + symbol index (no embeddings/LLM)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Optional

from .ast_tools import analyze_file


SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "models",
    "dist",
    "build",
    ".eggs",
}


def iter_py_files(root: Path | str) -> Iterator[Path]:
    root = Path(root)
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def search_text(
    root: Path | str,
    pattern: str,
    *,
    regex: bool = False,
    max_hits: int = 50,
) -> list[dict[str, Any]]:
    root = Path(root)
    hits: list[dict[str, Any]] = []
    cre = re.compile(pattern) if regex else None
    for path in iter_py_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            found = bool(cre.search(line)) if cre else (pattern in line)
            if found:
                hits.append(
                    {
                        "path": str(path.relative_to(root) if root in path.parents or path == root else path),
                        "lineno": i,
                        "line": line.strip()[:240],
                    }
                )
                if len(hits) >= max_hits:
                    return hits
    return hits


def build_symbol_index(root: Path | str) -> dict[str, list[dict[str, Any]]]:
    root = Path(root)
    index: dict[str, list[dict[str, Any]]] = {}
    for path in iter_py_files(root):
        analysis = analyze_file(path)
        if not analysis.ok:
            continue
        rel = str(path.relative_to(root)) if root in path.parents or path.parent == root else str(path)
        for sym in analysis.symbols:
            index.setdefault(sym.name, []).append(
                {"path": rel, "kind": sym.kind, "lineno": sym.lineno}
            )
    return index


def find_symbol(root: Path | str, name: str) -> list[dict[str, Any]]:
    index = build_symbol_index(root)
    # exact then suffix match
    hits = list(index.get(name) or [])
    if not hits:
        for key, locs in index.items():
            if key.endswith("." + name) or key == name:
                hits.extend(locs)
    return hits
