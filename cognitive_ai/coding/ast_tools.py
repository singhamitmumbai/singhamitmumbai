"""AST-based code analysis — symbols, metrics, imports (stdlib ast only)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SymbolInfo:
    name: str
    kind: str  # function|async_function|class|method|assign
    lineno: int
    end_lineno: Optional[int] = None
    args: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "args": list(self.args),
            "docstring": self.docstring,
            "decorators": list(self.decorators),
        }


@dataclass
class ModuleAnalysis:
    path: str
    ok: bool
    error: Optional[str] = None
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "error": self.error,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": list(self.imports),
            "metrics": dict(self.metrics),
        }


def analyze_source(source: str, path: str = "<memory>") -> ModuleAnalysis:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ModuleAnalysis(path=path, ok=False, error=f"SyntaxError: {exc}")

    symbols: list[SymbolInfo] = []
    imports: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(_func_info(node))
        elif isinstance(node, ast.ClassDef):
            symbols.append(
                SymbolInfo(
                    name=node.name,
                    kind="class",
                    lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", None),
                    docstring=ast.get_docstring(node),
                    decorators=[_name_of(d) for d in node.decorator_list],
                )
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info = _func_info(item)
                    info.kind = "method"
                    info.name = f"{node.name}.{item.name}"
                    symbols.append(info)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.extend(_import_names(node))

    metrics = {
        "lines": source.count("\n") + 1,
        "symbols": len(symbols),
        "imports": len(imports),
        "classes": sum(1 for s in symbols if s.kind == "class"),
        "functions": sum(1 for s in symbols if s.kind in ("function", "async_function")),
        "methods": sum(1 for s in symbols if s.kind == "method"),
        "cyclomatic_approx": _approx_complexity(tree),
    }
    return ModuleAnalysis(path=path, ok=True, symbols=symbols, imports=imports, metrics=metrics)


def analyze_file(path: Path | str) -> ModuleAnalysis:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except OSError as exc:
        return ModuleAnalysis(path=str(p), ok=False, error=str(exc))
    return analyze_source(source, path=str(p))


def find_symbol(source: str, name: str) -> Optional[SymbolInfo]:
    analysis = analyze_source(source)
    if not analysis.ok:
        return None
    for s in analysis.symbols:
        if s.name == name or s.name.endswith("." + name):
            return s
    return None


def rename_symbol_in_source(source: str, old: str, new: str) -> tuple[str, dict[str, Any]]:
    """Rename a top-level function/class definition and simple Name loads/stores."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return source, {"ok": False, "error": str(e)}

    class Renamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):  # noqa: N802
            self.generic_visit(node)
            if node.name == old:
                node.name = new
            return node

        def visit_AsyncFunctionDef(self, node):  # noqa: N802
            return self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):  # noqa: N802
            self.generic_visit(node)
            if node.name == old:
                node.name = new
            return node

        def visit_Name(self, node):  # noqa: N802
            if node.id == old:
                return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)
            return node

    new_tree = Renamer().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        out = ast.unparse(new_tree)
    except Exception as exc:  # noqa: BLE001
        return source, {"ok": False, "error": f"unparse_failed:{exc}"}
    return out + ("\n" if not out.endswith("\n") else ""), {"ok": True, "renamed": old, "to": new}


def _func_info(node: ast.FunctionDef | ast.AsyncFunctionDef) -> SymbolInfo:
    args = [a.arg for a in node.args.args]
    return SymbolInfo(
        name=node.name,
        kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
        lineno=node.lineno,
        end_lineno=getattr(node, "end_lineno", None),
        args=args,
        docstring=ast.get_docstring(node),
        decorators=[_name_of(d) for d in node.decorator_list],
    )


def _import_names(node: ast.AST) -> list[str]:
    out = []
    if isinstance(node, ast.Import):
        for a in node.names:
            out.append(a.name)
    elif isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        for a in node.names:
            out.append(f"{mod}.{a.name}" if mod else a.name)
    return out


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return type(node).__name__


def _approx_complexity(tree: ast.AST) -> int:
    """McCabe-ish: +1 per branch / loop / boolean op / except."""
    score = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.AsyncWith)):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += max(0, len(node.values) - 1)
        elif isinstance(node, ast.comprehension):
            score += 1
    return score
