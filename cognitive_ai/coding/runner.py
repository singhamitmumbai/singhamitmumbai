"""Compile / test runners sandboxed to the workspace (non-LLM) — hardened."""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


class CodeRunner:
    def __init__(self, root: Path | str, *, timeout_sec: float = 60.0) -> None:
        self.root = Path(root).resolve()
        self.timeout_sec = timeout_sec

    def _safe(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("path_escapes_workspace") from exc
        return path

    def compile_file(self, rel: str) -> dict[str, Any]:
        try:
            path = self._safe(rel)
        except PermissionError:
            return {"ok": False, "error": "path_escapes_workspace", "path": rel}
        if not path.exists():
            return {"ok": False, "error": "missing_file", "path": rel}
        try:
            py_compile.compile(str(path), doraise=True)
            return {"ok": True, "path": rel, "compiled": True}
        except py_compile.PyCompileError as exc:
            return {"ok": False, "path": rel, "error": str(exc), "stable": True}

    def compile_tree(self, subdir: str = ".") -> dict[str, Any]:
        try:
            base = self._safe(subdir)
        except PermissionError:
            return {"ok": False, "error": "path_escapes_workspace"}
        results = []
        ok_all = True
        for p in base.rglob("*.py"):
            if any(x in p.parts for x in (".venv", "venv", "__pycache__", "models", ".git")):
                continue
            rel = str(p.relative_to(self.root))
            r = self.compile_file(rel)
            results.append(r)
            if not r.get("ok"):
                ok_all = False
        return {
            "ok": ok_all,
            "files": len(results),
            "failures": [r for r in results if not r.get("ok")],
            "passed": sum(1 for r in results if r.get("ok")),
        }

    def run_pytest(
        self,
        args: Optional[list[str]] = None,
        *,
        max_fail: int = 5,
    ) -> dict[str, Any]:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--maxfail={max_fail}",
            "--tb=line",
            *(args or []),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "pytest_timeout", "timeout_sec": self.timeout_sec}
        out = (proc.stdout or "") + (proc.stderr or "")
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "output_tail": out[-4000:],
            "command": cmd,
        }

    def run_python(self, rel: str, argv: Optional[list[str]] = None) -> dict[str, Any]:
        try:
            path = self._safe(rel)
        except PermissionError:
            return {"ok": False, "error": "path_escapes_workspace"}
        if not path.exists():
            return {"ok": False, "error": "missing_file", "path": rel}
        cmd = [sys.executable, str(path), *(argv or [])]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": os.environ.get("HOME", ""),
        }
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=min(self.timeout_sec, 30.0),
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "run_timeout"}
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-2000:],
            "stderr": (proc.stderr or "")[-2000:],
        }
