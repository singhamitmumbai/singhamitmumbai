"""CodingSystem — stronger, more stable non-LLM coding brain.

Upgrades vs v1:
- Smart pure-function implementations (add/mul/…) instead of NotImplementedError stubs
- Atomic writes + refuse invalid syntax before save
- Multi-stage verifier (AST + py_compile + soft lint + optional smoke exec)
- Richer intents: format, verify, explain, implement, enum, fix-syntax
- Consistent response envelope: ok/success/action/text/verify

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
    generate_enum,
    generate_function,
    generate_module,
    generate_pytest,
    parse_generate_request,
)
from .runner import CodeRunner
from .search import find_symbol, search_text
from .verifier import CodeVerifier


class CodingSystem:
    version = "coding_system_v2"

    def __init__(self, root: Optional[Path | str] = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.editor = CodeEditor(self.root)
        self.runner = CodeRunner(self.root)
        self.verifier = CodeVerifier(self.root)

    def handle(self, instruction: str, *, metadata: Optional[dict] = None) -> dict[str, Any]:
        """Route a coding instruction to the appropriate tool."""
        meta = dict(metadata or {})
        text = (instruction or "").strip()
        lower = text.lower()

        try:
            if meta.get("action"):
                return self._pack(self._dispatch(meta["action"], text, meta))

            if re.search(r"\b(?:coding\s+)?(?:status|capabilities)\b", lower) or lower in ("help", "coding help"):
                return self._pack({**self.status(), "ok": True, "action": "status"})

            if re.search(r"\b(?:analyze|inspect)\s+(?:code|file|module)\b", lower) or lower.startswith("analyze "):
                path = meta.get("path") or _extract_path(text) or "."
                return self._pack(self.analyze(path))

            if re.search(r"\bexplain\s+(?:file|code|module)\b", lower) or lower.startswith("explain "):
                path = meta.get("path") or _extract_path(text) or "."
                return self._pack(self.explain(path))

            if re.search(r"\b(?:verify|validate|check)\s+(?:file|code|module)?\b", lower) or lower.startswith("verify "):
                path = meta.get("path") or _extract_path(text)
                if not path:
                    return self._pack({"ok": False, "error": "need_path_for_verify", "action": "verify"})
                return self._pack({"action": "verify", **self.verifier.verify_file(path)})

            if re.search(r"\bformat\s+(?:file|code)?\b", lower) or lower.startswith("format "):
                path = meta.get("path") or _extract_path(text)
                if not path:
                    return self._pack({"ok": False, "error": "need_path_for_format", "action": "format"})
                return self._pack({"action": "format", **self.editor.format_source(path)})

            if re.search(r"\brename symbol\b", lower):
                old = meta.get("old")
                new = meta.get("new")
                path = meta.get("path")
                if not (old and new and path):
                    m = re.search(r"rename symbol\s+(\w+)\s+to\s+(\w+)(?:\s+in\s+(\S+))?", lower)
                    if m:
                        old, new, path = m.group(1), m.group(2), m.group(3) or meta.get("path")
                if not (old and new and path):
                    return self._pack({"ok": False, "error": "need old, new, path", "action": "rename_symbol"})
                out = self.editor.rename_symbol(path, old, new)
                if out.get("ok"):
                    out["verify"] = self.verifier.verify_file(path)
                return self._pack({"action": "rename_symbol", **out})

            if re.search(r"\bfind symbol\b", lower):
                name = meta.get("name") or _extract_ident(text, after=("symbol", "find"))
                return self._pack(
                    {"ok": True, "action": "find_symbol", "name": name, "hits": find_symbol(self.root, name or "")}
                )

            if re.search(r"\b(?:code search|search code|search for)\b", lower):
                pat = meta.get("pattern") or _extract_quoted(text) or _last_token(text)
                return self._pack(
                    {"ok": True, "action": "search", "pattern": pat, "hits": search_text(self.root, pat or "")}
                )

            if re.search(r"\brun tests?\b|\bpytest\b", lower):
                args = meta.get("pytest_args") or []
                return self._pack({"action": "pytest", **self.runner.run_pytest(args)})

            if re.search(r"\bcompile\b", lower):
                path = meta.get("path") or _extract_path(text)
                if path and path.endswith(".py"):
                    return self._pack({"action": "compile", **self.runner.compile_file(path)})
                return self._pack({"action": "compile_tree", **self.runner.compile_tree()})

            if re.search(r"\brun(?:\s+python)?\s+[\w./-]+\.py\b", lower) or meta.get("run"):
                path = meta.get("path") or _extract_path(text)
                if not path:
                    return self._pack({"ok": False, "error": "need_path_to_run", "action": "run"})
                return self._pack({"action": "run", **self.runner.run_python(path, meta.get("argv"))})

            plan = parse_generate_request(text)
            if plan:
                return self._pack(self._generate_from_plan(plan, meta))

            if re.search(r"\bwrite code\b|\bgenerate code\b|\bcreate module\b", lower):
                name = meta.get("name") or _extract_ident(text, after=("module", "code")) or "generated_module"
                return self._pack(self._generate_from_plan({"action": "generate_module", "name": name}, meta))

            return self._pack(
                {
                    "ok": False,
                    "success": False,
                    "error": "unsupported_coding_instruction",
                    "hint": (
                        "Try: analyze file X.py | explain file X.py | verify file X.py | format file X.py | "
                        "find symbol name | search for 'pattern' | "
                        "generate function add(a, b) | generate class Bar | generate test for foo | "
                        "generate enum Status | rename symbol old to new in path.py | "
                        "run tests | compile | run path.py"
                    ),
                    "llm_dependency": False,
                }
            )
        except PermissionError as exc:
            return self._pack({"ok": False, "success": False, "error": str(exc), "action": "security"})
        except Exception as exc:  # noqa: BLE001 — stability: never crash the product router
            return self._pack(
                {
                    "ok": False,
                    "success": False,
                    "error": f"coding_internal_error:{exc}",
                    "action": "error",
                    "stable": True,
                }
            )

    def analyze(self, path: str) -> dict[str, Any]:
        p = Path(path)
        if p.is_dir() or path in (".", ""):
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
        try:
            target = self.editor.resolve(path)
        except PermissionError as exc:
            return {"ok": False, "action": "analyze_file", "error": str(exc), "path": path}
        analysis = analyze_file(target)
        return {"ok": analysis.ok, "action": "analyze_file", **analysis.to_dict()}

    def explain(self, path: str) -> dict[str, Any]:
        analysis = self.analyze(path)
        if not analysis.get("ok"):
            return {**analysis, "action": "explain", "text": analysis.get("error") or "analyze_failed"}
        if analysis.get("action") == "analyze_tree":
            text = (
                f"Workspace has {analysis.get('files')} Python files; "
                f"approx complexity sum={analysis.get('total_complexity_approx')}."
            )
            return {**analysis, "action": "explain_tree", "text": text}
        syms = analysis.get("symbols") or []
        metrics = analysis.get("metrics") or {}
        lines = [
            f"File `{analysis.get('path')}`:",
            f"- {metrics.get('functions', 0)} functions, {metrics.get('classes', 0)} classes, "
            f"{metrics.get('methods', 0)} methods",
            f"- ~{metrics.get('cyclomatic_approx', '?')} cyclomatic complexity, {metrics.get('lines', '?')} lines",
        ]
        if analysis.get("imports"):
            lines.append("- imports: " + ", ".join(analysis["imports"][:12]))
        for s in syms[:12]:
            lines.append(f"- {s.get('kind')}: `{s.get('name')}` @ L{s.get('lineno')}")
        return {**analysis, "action": "explain_file", "text": "\n".join(lines)}

    def _generate_from_plan(self, plan: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        action = plan["action"]
        name = plan.get("name") or meta.get("name") or "generated"
        write_path = meta.get("path")
        code = ""
        implement = bool(plan.get("implement", True) if meta.get("implement") is None else meta.get("implement"))

        if action == "generate_function":
            code = generate_function(
                name,
                args=plan.get("args") or meta.get("args") or [],
                implement=implement,
                returns=meta.get("returns"),
                body=meta.get("body"),
                docstring=meta.get("docstring"),
            )
            write_path = write_path or f"generated/{name}.py"
        elif action == "generate_class":
            code = generate_class(name, methods=meta.get("methods"))
            write_path = write_path or f"generated/{name}.py"
        elif action == "generate_module":
            code = generate_module(
                name,
                functions=meta.get("functions")
                or [{"name": "main", "args": [], "body": "    return 0\n", "returns": "int"}],
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
        elif action == "generate_enum":
            code = generate_enum(name, members=meta.get("members"))
            write_path = write_path or f"generated/{name}.py"
        else:
            return {"ok": False, "error": f"unknown_generate_action:{action}"}

        # Pre-verify before write (stability)
        pre = self.verifier.verify_source(code, path=write_path or "<plan>")
        if not pre.get("ok"):
            return {
                "ok": False,
                "success": False,
                "action": action,
                "path": write_path,
                "code": code,
                "verify": pre,
                "error": pre.get("error") or "pre_verify_failed",
            }

        if meta.get("dry_run"):
            return {
                "ok": True,
                "action": action,
                "path": write_path,
                "code": code,
                "dry_run": True,
                "verify": pre,
            }

        written = self.editor.write(write_path, code)
        if not written.get("ok"):
            return {
                "ok": False,
                "success": False,
                "action": action,
                "path": write_path,
                "code": code,
                "write": written,
                "error": written.get("error") or "write_failed",
            }

        compiled = self.runner.compile_file(write_path)
        verify = self.verifier.verify_file(write_path)
        smoke = None
        if action == "generate_function" and (plan.get("args") is not None or True):
            # Smoke only for known pure ops with numeric defaults
            args = plan.get("args") or []
            if name.lower() in ("add", "mul", "multiply", "sub", "subtract", "div", "divide") and len(args) >= 2:
                smoke = self.verifier.safe_exec_function(code, name, [2, 3])

        ok = bool(written.get("ok") and compiled.get("ok") and verify.get("ok"))
        text = f"Generated `{write_path}` ({action})"
        if smoke and smoke.get("ok"):
            text += f" · smoke {name}{tuple(smoke.get('args') or ())} => {smoke.get('result')}"
        return {
            "ok": ok,
            "success": ok,
            "action": action,
            "path": write_path,
            "code": code,
            "write": written,
            "compile": compiled,
            "verify": verify,
            "smoke": smoke,
            "text": text,
            "llm_dependency": False,
        }

    def _dispatch(self, action: str, text: str, meta: dict[str, Any]) -> dict[str, Any]:
        if action == "analyze":
            return self.analyze(meta.get("path") or ".")
        if action == "explain":
            return self.explain(meta.get("path") or ".")
        if action == "verify":
            return {"action": "verify", **self.verifier.verify_file(meta["path"])}
        if action == "format":
            return {"action": "format", **self.editor.format_source(meta["path"])}
        if action == "search":
            return {"ok": True, "hits": search_text(self.root, meta.get("pattern") or text)}
        if action == "find_symbol":
            return {"ok": True, "hits": find_symbol(self.root, meta.get("name") or text)}
        if action.startswith("generate"):
            return self._generate_from_plan(
                {"action": action, "name": meta.get("name", "generated"), "args": meta.get("args")},
                meta,
            )
        if action == "pytest":
            return {"action": "pytest", **self.runner.run_pytest(meta.get("pytest_args"))}
        if action == "compile":
            return {"action": "compile", **self.runner.compile_file(meta["path"])}
        if action == "run":
            return {"action": "run", **self.runner.run_python(meta["path"], meta.get("argv"))}
        if action == "rename_symbol":
            return self.editor.rename_symbol(meta["path"], meta["old"], meta["new"])
        return {"ok": False, "error": f"unknown_action:{action}"}

    def _pack(self, out: dict[str, Any]) -> dict[str, Any]:
        result = dict(out)
        result.setdefault("ok", bool(result.get("success", result.get("ok", False))))
        result.setdefault("success", bool(result.get("ok")))
        result.setdefault("llm_dependency", False)
        result.setdefault("human_level_agi", False)
        result.setdefault("version", self.version)
        result.setdefault("product", "CodingSystem")
        # Friendly text for CompleteAI chat surface
        if "text" not in result:
            if result.get("ok") and result.get("code"):
                result["text"] = f"```python\n{result['code']}```"
            elif result.get("ok") and result.get("action"):
                result["text"] = f"Coding {result['action']}: ok"
            elif result.get("error"):
                result["text"] = f"Coding error: {result['error']}"
            else:
                result["text"] = result.get("hint") or "Done."
        return result

    def status(self) -> dict[str, Any]:
        return {
            "name": "CodingSystem",
            "version": self.version,
            "root": str(self.root),
            "capabilities": [
                "ast_analyze",
                "explain_file",
                "verify",
                "format",
                "symbol_index",
                "text_search",
                "generate_function",
                "generate_class",
                "generate_module",
                "generate_test",
                "generate_dataclass",
                "generate_enum",
                "smart_pure_impls",
                "atomic_write",
                "structured_edit",
                "rename_symbol",
                "py_compile",
                "pytest",
                "run_python",
                "smoke_exec",
            ],
            "llm_at_runtime": False,
            "human_level_agi": False,
            "notes": "Symbolic/template coding tools with verification — not an LLM pair-programmer.",
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
