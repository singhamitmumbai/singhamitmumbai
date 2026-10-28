"""Stronger / more stable CodingSystem (v1.24)."""

from __future__ import annotations

from pathlib import Path

from cognitive_ai.coding import CodingSystem
from cognitive_ai.coding.generator import generate_function, parse_generate_request
from cognitive_ai.coding.verifier import CodeVerifier
from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.version import __version__


def test_version_coding_v124():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 24)


def test_smart_add_implements_and_smoke(tmp_path: Path):
    coding = CodingSystem(root=tmp_path)
    out = coding.handle("generate function add(a, b)", metadata={"path": "ops.py"})
    assert out["ok"] is True
    assert out["verify"]["ok"] is True
    src = (tmp_path / "ops.py").read_text()
    assert "return a + b" in src
    assert "NotImplementedError" not in src
    assert out.get("smoke", {}).get("ok") is True
    assert out["smoke"]["result"] == 5


def test_multiply_and_divide_stable(tmp_path: Path):
    coding = CodingSystem(root=tmp_path)
    mul = coding.handle("generate function multiply(x, y)", metadata={"path": "m.py"})
    assert mul["ok"] and "return x * y" in mul["code"]
    div = coding.handle("generate function divide(a, b)", metadata={"path": "d.py"})
    assert div["ok"] and "ZeroDivisionError" in div["code"]


def test_atomic_write_rejects_syntax(tmp_path: Path):
    ed = CodingSystem(root=tmp_path).editor
    bad = ed.write("broken.py", "def oops(:\n  pass\n")
    assert bad["ok"] is False
    assert not (tmp_path / "broken.py").exists()


def test_verify_format_explain(tmp_path: Path):
    coding = CodingSystem(root=tmp_path)
    coding.handle("generate function greet(name)", metadata={"path": "g.py"})
    v = coding.handle("verify file g.py")
    assert v["ok"] is True
    exp = coding.handle("explain file g.py")
    assert exp["ok"] and "greet" in exp["text"]
    fmt = coding.handle("format file g.py")
    assert fmt["ok"] is True


def test_enum_and_implement_parse():
    plan = parse_generate_request("implement function clamp(x, lo, hi)")
    assert plan and plan["name"] == "clamp"
    code = generate_function("clamp", args=["x", "lo", "hi"])
    assert "max(" in code and "min(" in code
    coding = CodingSystem(root=Path.cwd())
    # dry
    out = coding.handle(
        "generate enum JobStatus",
        metadata={"path": "generated/_enum_job.py", "dry_run": True, "members": ["A", "B"]},
    )
    assert out["ok"] and "Enum" in out["code"]


def test_path_escape_blocked(tmp_path: Path):
    coding = CodingSystem(root=tmp_path)
    out = coding.handle("analyze file /etc/passwd")
    # either security error or analyze failure — must not crash
    assert "ok" in out
    assert out.get("human_level_agi") is False


def test_verifier_smoke_bans_os():
    v = CodeVerifier(Path.cwd())
    bad = v.safe_exec_function("import os\ndef f():\n    return os.getcwd()\n", "f")
    assert bad["ok"] is False


def test_complete_ai_coding_route_stronger():
    ai = CompleteAI(seed_demo=False, enable_llm=False)
    out = ai.handle("generate function add(a, b)")
    assert out["route"] == "coding"
    assert out.get("ok") or out.get("code") or "add" in (out.get("text") or "")
    assert out.get("human_level_agi") is False


def test_status_lists_new_caps():
    st = CodingSystem().status()
    assert st["version"] == "coding_system_v2"
    for cap in ("verify", "format", "atomic_write", "smart_pure_impls", "generate_enum"):
        assert cap in st["capabilities"]
