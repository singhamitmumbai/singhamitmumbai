"""CodingSystem — advanced non-LLM coding brain for analysis, generation, edit, test.

Capabilities (symbolic only):
- AST analysis & complexity metrics
- Workspace symbol index / text search
- Template generation (function/class/module/test/dataclass)
- Safe structured edits & symbol rename
- py_compile + pytest runners

Not an LLM coding assistant. Not AGI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .ast_tools import analyze_file, analyze_source
from .editor import CodeEditor
from .generator import (
    generate_class,
    generate_dataclass,
    generate_function,
    generate_module,
    generate_pytest,
    parse_generate_request,
)
from .runner import CodeRunner
from .search import find_symbol, search_text


class CodingSystem:
    def __init__(self, root: Optional[Path | str] = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.editor = CodeEditor(self.root)
        self.runner = CodeRunner(self.root)

    def handle(self, instruction: str, *, metadata: Optional[dict] = None) -> dict[str, Any]:
        """Route a coding instruction to the appropriate tool."""
        meta = metadata or {}
        text = instruction.strip()
        lower = text.lower()

        # Explicit structured action from metadata
        action = meta.get("action")
        if action:
            return self._dispatch(action, text, meta)

        # Natural-ish patterns
        if re.search(r"\b(?:analyze|inspect)\s+(?:code|file|module)\b", lower) or lower.startswith("analyze "):
            path = meta.get("path") or _extract_path(text) or "."
            return self.analyze(path)

        if re.search(r"\brename symbol\b", lower):
            old = meta.get("old")
            new = meta.get("new")
            path = meta.get("path")
            if not (old and new and path):
                m = re.search(r"rename symbol\s+(\w+)\s+to\s+(\w+)(?:\s+in\s+(\S+))?", lower)
                if m:
                    old, new, path = m.group(1), m.group(2), m.group(3) or meta.get("path")
            if not (old and new and path):
                return {"ok": False, "error": "need old, new, path"}
            return self.editor.rename_symbol(path, old, new)

        if re.search(r"\bfind symbol\b", lower):
            name = meta.get("name") or _extract_ident(text, after=("symbol", "find"))
            return {"ok": True, "action": "find_symbol", "name": name, "hits": find_symbol(self.root, name or "")}

        if re.search(r"\b(?:code search|search code|search for)\b", lower):
            pat = meta.get("pattern") or _extract_quoted(text) or _last_token(text)
            return {"ok": True, "action": "search", "pattern": pat, "hits": search_text(self.root, pat or "")}

        if re.search(r"\brun tests?\b|\bpytest\b", lower):
            args = meta.get("pytest_args") or []
            return {"ok": True, "action": "pytest", **self.runner.run_pytest(args)}

        if re.search(r"\bcompile\b", lower):
            path = meta.get("path") or _extract_path(text)
            if path and path.endswith(".py"):
                return {"ok": True, "action": "compile", **self.runner.compile_file(path)}
            return {"ok": True, "action": "compile_tree", **self.runner.compile_tree()}

        plan = parse_generate_request(text)
        if plan:
            return self._generate_from_plan(plan, meta)

        if re.search(r"\bwrite code\b|\bgenerate code\b|\bcreate module\b", lower):
            name = meta.get("name") or _extract_ident(text, after=("module", "code")) or "generated_module"
            return self._generate_from_plan({"action": "generate_module", "name": name}, meta)

        return {
            "ok": False,
            "success": False,
            "error": "unsupported_coding_instruction",
            "hint": (
                "Try: analyze file X.py | find symbol name | search for 'pattern' | "
                "generate function foo(a, b) | generate class Bar | generate test for foo | "
                "rename symbol old to new in path.py | run tests | compile"
            ),
            "llm_dependency": False,
        }

    def analyze(self, path: str) -> dict[str, Any]:
        p = Path(path)
        if p.is_dir() or path in (".", ""):
            # summarize tree
            from .search import iter_py_files

            files = list(iter_py_files(self.root if path in (".", "") else self.root / path))
            sample = []
            total_complexity = 0
            for f in files[:40]:
                a = analyze_file(f)
                if a.ok:
                    total_complexity += int(a.metrics.get("cyclomatic_approx") or 0)
                    sample.append({"path": str(f.relative_to(self.root)), "metrics": a.metrics})
            return {
                "ok": True,
                "action": "analyze_tree",
                "files": len(files),
                "sample": sample[:20],
                "total_complexity_approx": total_complexity,
            }
        target = path if (self.root / path).exists() else path
        rel = path
        try:
            analysis = analyze_file(self.root / rel if not Path(rel).is_absolute() else rel)
        except Exception:
            analysis = analyze_file(target)
        return {"ok": analysis.ok, "action": "analyze_file", **analysis.to_dict()}

    def _generate_from_plan(self, plan: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        action = plan["action"]
        name = plan.get("name") or meta.get("name") or "generated"
        write_path = meta.get("path")
        code = ""
        if action == "generate_function":
            code = generate_function(name, args=plan.get("args") or meta.get("args") or [])
            write_path = write_path or f"generated/{name}.py"
        elif action == "generate_class":
            code = generate_class(name, methods=meta.get("methods"))
            write_path = write_path or f"generated/{name}.py"
        elif action == "generate_module":
            code = generate_module(
                name,
                functions=meta.get("functions") or [{"name": "main", "args": [], "body": "    return 0\n", "returns": "int"}],
                classes=meta.get("classes"),
            )
            write_path = write_path or f"generated/{name}.py"
        elif action == "generate_test":
            code = generate_pytest(name, module=meta.get("module"), cases=meta.get("cases"))
            write_path = write_path or f"generated/test_{name}.py"
        elif action == "generate_dataclass":
            fields = meta.get("fields") or {"id": "str", "value": "int"}
            code = generate_dataclass(name, fields)
            write_path = write_path or f"generated/{name}.py"
        else:
            return {"ok": False, "error": f"unknown_generate_action:{action}"}

        if meta.get("dry_run"):
            return {"ok": True, "action": action, "path": write_path, "code": code, "dry_run": True}

        written = self.editor.write(write_path, code)
        compiled = self.runner.compile_file(write_path) if written.get("ok") else {"ok": False}
        return {
            "ok": bool(written.get("ok") and compiled.get("ok")),
            "success": bool(written.get("ok") and compiled.get("ok")),
            "action": action,
            "path": write_path,
            "code": code,
            "write": written,
            "compile": compiled,
            "llm_dependency": False,
        }

    def _dispatch(self, action: str, text: str, meta: dict[str, Any]) -> dict[str, Any]:
        if action == "analyze":
            return self.analyze(meta.get("path") or ".")
        if action == "search":
            return {"ok": True, "hits": search_text(self.root, meta.get("pattern") or text)}
        if action == "find_symbol":
            return {"ok": True, "hits": find_symbol(self.root, meta.get("name") or text)}
        if action.startswith("generate"):
            return self._generate_from_plan({"action": action, "name": meta.get("name", "generated")}, meta)
        if action == "pytest":
            return {"ok": True, **self.runner.run_pytest(meta.get("pytest_args"))}
        if action == "compile":
            return {"ok": True, **self.runner.compile_file(meta["path"])}
        if action == "rename_symbol":
            return self.editor.rename_symbol(meta["path"], meta["old"], meta["new"])
        return {"ok": False, "error": f"unknown_action:{action}"}

    def status(self) -> dict[str, Any]:
        return {
            "name": "CodingSystem",
            "root": str(self.root),
            "capabilities": [
                "ast_analyze",
                "symbol_index",
                "text_search",
                "generate_function",
                "generate_class",
                "generate_module",
                "generate_test",
                "generate_dataclass",
                "structured_edit",
                "rename_symbol",
                "py_compile",
                "pytest",
            ],
            "llm_at_runtime": False,
            "human_level_agi": False,
            "notes": "Symbolic/template coding tools — not an LLM pair-programmer.",
        }


def _extract_path(text: str) -> Optional[str]:
    m = re.search(r"([\w./-]+\.py)", text)
    return m.group(1) if m else None


def _extract_quoted(text: str) -> Optional[str]:
    m = re.search(r"['\"]([^'\"]+)['\"]", text)
    return m.group(1) if m else None


def _extract_ident(text: str, after: tuple[str, ...] = ()) -> Optional[str]:
    lower = text.lower()
    for a in after:
        m = re.search(rf"{a}\s+(\w+)", lower)
        if m:
            return m.group(1)
    m = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", text)
    return m[-1] if m else None


def _last_token(text: str) -> Optional[str]:
    parts = text.strip().split()
    return parts[-1] if parts else None
