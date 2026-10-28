"""Safe structured file edits under a workspace root — atomic writes for stability."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from .ast_tools import analyze_source, rename_symbol_in_source


class CodeEditor:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def resolve(self, rel: str) -> Path:
        # Disallow absolute escapes
        raw = Path(rel)
        if raw.is_absolute():
            path = raw.resolve()
        else:
            path = (self.root / rel).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"path_escapes_workspace:{rel}") from exc
        return path

    def read(self, rel: str) -> str:
        return self.resolve(rel).read_text(encoding="utf-8")

    def write(
        self,
        rel: str,
        content: str,
        *,
        create_parents: bool = True,
        atomic: bool = True,
        backup: bool = False,
    ) -> dict[str, Any]:
        path = self.resolve(rel)
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        analysis = analyze_source(content, path=str(path))
        if not analysis.ok:
            return {
                "ok": False,
                "path": str(path.relative_to(self.root)),
                "error": analysis.error,
                "analysis": analysis.to_dict(),
            }
        if backup and path.exists():
            bak = path.with_suffix(path.suffix + ".bak")
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        if atomic:
            fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(content)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        else:
            path.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "path": str(path.relative_to(self.root)),
            "bytes": len(content.encode("utf-8")),
            "atomic": atomic,
            "analysis": analysis.to_dict(),
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

    def insert_after_symbol(self, rel: str, symbol: str, snippet: str) -> dict[str, Any]:
        """Insert snippet after a top-level function/class definition."""
        from .ast_tools import analyze_source

        src = self.read(rel)
        analysis = analyze_source(src)
        if not analysis.ok:
            return {"ok": False, "error": analysis.error}
        target = next((s for s in analysis.symbols if s.name == symbol or s.name.endswith("." + symbol)), None)
        if not target or not target.end_lineno:
            return {"ok": False, "error": f"symbol_not_found:{symbol}"}
        lines = src.splitlines(keepends=True)
        idx = target.end_lineno
        snippet = snippet if snippet.endswith("\n") else snippet + "\n"
        new_src = "".join(lines[:idx]) + "\n" + snippet + "".join(lines[idx:])
        return self.write(rel, new_src)

    def format_source(self, rel: str) -> dict[str, Any]:
        """Normalize via ast.parse + unparse (stable, no black dependency)."""
        import ast

        src = self.read(rel)
        try:
            tree = ast.parse(src)
            out = ast.unparse(tree)
            if not out.endswith("\n"):
                out += "\n"
        except SyntaxError as exc:
            return {"ok": False, "error": str(exc)}
        return self.write(rel, out)

    def rename_symbol(self, rel: str, old: str, new: str) -> dict[str, Any]:
        src = self.read(rel)
        out, meta = rename_symbol_in_source(src, old, new)
        if not meta.get("ok"):
            return meta
        written = self.write(rel, out)
        written.update(meta)
        return written
