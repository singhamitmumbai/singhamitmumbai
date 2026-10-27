"""Compile / test runners sandboxed to the workspace (non-LLM)."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


class CodeRunner:
    def __init__(self, root: Path | str, *, timeout_sec: float = 60.0) -> None:
        self.root = Path(root).resolve()
        self.timeout_sec = timeout_sec

    def compile_file(self, rel: str) -> dict[str, Any]:
        path = (self.root / rel).resolve()
        if not str(path).startswith(str(self.root)):
            return {"ok": False, "error": "path_escapes_workspace"}
        try:
            py_compile.compile(str(path), doraise=True)
            return {"ok": True, "path": rel, "compiled": True}
        except py_compile.PyCompileError as exc:
            return {"ok": False, "path": rel, "error": str(exc)}

    def compile_tree(self, subdir: str = ".") -> dict[str, Any]:
        base = (self.root / subdir).resolve()
        results = []
        ok_all = True
        for p in base.rglob("*.py"):
            if any(x in p.parts for x in (".venv", "venv", "__pycache__", "models")):
                continue
            rel = str(p.relative_to(self.root))
            r = self.compile_file(rel)
            results.append(r)
            if not r.get("ok"):
                ok_all = False
        return {"ok": ok_all, "files": len(results), "failures": [r for r in results if not r.get("ok")]}

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
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
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
        path = (self.root / rel).resolve()
        if not str(path).startswith(str(self.root)):
            return {"ok": False, "error": "path_escapes_workspace"}
        cmd = [sys.executable, str(path), *(argv or [])]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "run_timeout"}
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-2000:],
            "stderr": (proc.stderr or "")[-2000:],
        }
