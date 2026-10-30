"""Multi-stage code verification — syntax, AST, optional safe exec (non-LLM)."""

from __future__ import annotations

import ast
import py_compile
import tempfile
from pathlib import Path
from typing import Any, Optional


class CodeVerifier:
    """Stable checks after generate/edit — never claims AGI."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def verify_source(self, source: str, *, path: str = "<memory>") -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        # 1) Parse
        try:
            tree = ast.parse(source)
            checks.append({"name": "ast_parse", "ok": True})
        except SyntaxError as exc:
            return {
                "ok": False,
                "path": path,
                "checks": [{"name": "ast_parse", "ok": False, "error": str(exc)}],
                "error": f"SyntaxError: {exc}",
            }

        # 2) py_compile via temp file
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
                fh.write(source)
                tmp = fh.name
            py_compile.compile(tmp, doraise=True)
            checks.append({"name": "py_compile", "ok": True})
            Path(tmp).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": "py_compile", "ok": False, "error": str(exc)})
            return {"ok": False, "path": path, "checks": checks, "error": str(exc)}

        # 3) Structural expectations
        funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        checks.append(
            {
                "name": "structure",
                "ok": True,
                "functions": len(funcs),
                "classes": len(classes),
                "has_docstring": bool(ast.get_docstring(tree)),
            }
        )

        # 4) Soft lint — empty body / bare except
        issues: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [b for b in node.body if not isinstance(b, ast.Expr)]
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    issues.append(f"empty_pass:{node.name}")
                if len(node.body) == 1 and isinstance(node.body[0], ast.Raise) and isinstance(
                    getattr(node.body[0], "exc", None), ast.Call
                ):
                    # NotImplementedError stub is acceptable for scaffolds
                    pass
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(f"bare_except:L{node.lineno}")
        checks.append({"name": "soft_lint", "ok": True, "issues": issues[:20]})

        return {
            "ok": all(c.get("ok") for c in checks if c["name"] != "soft_lint"),
            "path": path,
            "checks": checks,
            "issues": issues[:20],
            "human_level_agi": False,
        }

    def verify_file(self, rel: str) -> dict[str, Any]:
        path = (self.root / rel).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return {"ok": False, "error": "path_escapes_workspace", "path": rel}
        if not path.exists():
            return {"ok": False, "error": "missing_file", "path": rel}
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc), "path": rel}
        out = self.verify_source(source, path=rel)
        out["bytes"] = len(source.encode("utf-8"))
        return out

    def safe_exec_function(
        self,
        source: str,
        func_name: str,
        args: Optional[list[Any]] = None,
        *,
        timeout_hint: float = 1.0,
    ) -> dict[str, Any]:
        """Execute a pure generated function in a restricted globals dict.

        Only for smoke-testing arithmetic / simple returns — not a sandbox VM.
        """
        args = args if args is not None else []
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {"ok": False, "error": str(exc)}
        # Reject obvious side-effect imports
        banned = {"os", "sys", "subprocess", "socket", "pathlib", "shutil", "ctypes"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in banned:
                        return {"ok": False, "error": f"banned_import:{a.name}"}
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in banned:
                return {"ok": False, "error": f"banned_import:{node.module}"}
        glb: dict[str, Any] = {"__builtins__": {"range": range, "len": len, "abs": abs, "min": min, "max": max, "sum": sum, "int": int, "float": float, "str": str, "bool": bool, "list": list, "dict": dict, "tuple": tuple, "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError, "NotImplementedError": NotImplementedError}}
        loc: dict[str, Any] = {}
        try:
            exec(compile(tree, "<verify>", "exec"), glb, loc)  # noqa: S102 — intentional restricted smoke
            fn = loc.get(func_name) or glb.get(func_name)
            if not callable(fn):
                return {"ok": False, "error": f"function_missing:{func_name}"}
            result = fn(*args)
            return {"ok": True, "result": result, "func": func_name, "args": args}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "func": func_name}
