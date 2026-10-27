"""Safe structured file edits under a workspace root."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .ast_tools import analyze_source, rename_symbol_in_source


class CodeEditor:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def resolve(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if not str(path).startswith(str(self.root)):
            raise PermissionError(f"path_escapes_workspace:{rel}")
        return path

    def read(self, rel: str) -> str:
        return self.resolve(rel).read_text(encoding="utf-8")

    def write(self, rel: str, content: str, *, create_parents: bool = True) -> dict[str, Any]:
        path = self.resolve(rel)
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        analysis = analyze_source(content, path=str(path))
        return {
            "ok": analysis.ok,
            "path": str(path.relative_to(self.root)),
            "bytes": len(content.encode("utf-8")),
            "analysis": analysis.to_dict() if analysis.ok else {"error": analysis.error},
        }

    def append(self, rel: str, content: str) -> dict[str, Any]:
        path = self.resolve(rel)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        return self.write(rel, existing + content)

    def replace_span(self, rel: str, start_line: int, end_line: int, new_text: str) -> dict[str, Any]:
        path = self.resolve(rel)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        start = max(1, start_line) - 1
        end = min(len(lines), end_line)
        if start >= end:
            return {"ok": False, "error": "invalid_span"}
        replacement = new_text if new_text.endswith("\n") else new_text + "\n"
        new_lines = lines[:start] + [replacement] + lines[end:]
        return self.write(rel, "".join(new_lines))

    def rename_symbol(self, rel: str, old: str, new: str) -> dict[str, Any]:
        src = self.read(rel)
        out, meta = rename_symbol_in_source(src, old, new)
        if not meta.get("ok"):
            return meta
        written = self.write(rel, out)
        written.update(meta)
        return written
