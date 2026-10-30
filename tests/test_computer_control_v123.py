"""Detailed full computer control — gated, sandboxed, no keylogging."""

from __future__ import annotations

from pathlib import Path

from cognitive_ai.assistant.computer import ComputerControl, ALLOWED
from cognitive_ai.assistant.computer_nlu import parse_computer_command
from cognitive_ai.assistant.computer_policy import (
    policy_snapshot,
    resolve_in_sandbox,
    shell_allowed,
)
from cognitive_ai.assistant.finder_assistant import FinderbayAssistant
from cognitive_ai.version import __version__


def test_version_computer_v123():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 23)


def test_disabled_by_default_and_status():
    cc = ComputerControl(allow=False)
    st = cc.status()
    assert st["enabled"] is False
    assert st["no_keylogging"] is True
    assert st["no_background_clipboard_monitor"] is True
    assert "list_dir" in st["allowed_actions"]
    assert "screenshot" in ALLOWED
    r = cc.run("open_url", url="https://example.com")
    assert r["ok"] is False
    assert r["error"] == "computer_control_disabled"


def test_help_and_policy_without_enable():
    cc = ComputerControl(allow=False)
    h = cc.run("help")
    assert h["ok"] and "examples" in h["result"]
    p = cc.run("policy")
    assert p["ok"] and p["result"]["no_keylogging"] is True
    snap = policy_snapshot()
    assert "calculator" in snap["apps"]


def test_sandbox_blocks_etc_and_allows_cwd():
    path, err = resolve_in_sandbox("/etc/passwd", must_exist=False)
    assert path is None
    assert err and ("forbidden" in err or "outside_sandbox" in err)
    p2, err2 = resolve_in_sandbox("README.md", must_exist=True)
    assert err2 is None and p2 is not None


def test_shell_denylist():
    ok, reason = shell_allowed("rm -rf /")
    assert ok is False
    ok2, _ = shell_allowed("echo hello")
    assert ok2 is True


def test_enabled_filesystem_ops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cc = ComputerControl(allow=True, sandbox_roots=[tmp_path])
    assert cc.run("mkdir", path="data/demo_dir")["ok"]
    w = cc.run(
        "write_file",
        path="data/demo_dir/note.txt",
        content="hello control",
        confirm=True,
    )
    assert w["ok"]
    r = cc.run("read_file", path="data/demo_dir/note.txt")
    assert r["ok"] and "hello control" in r["result"]["text"]
    ls = cc.run("list_dir", path="data/demo_dir")
    assert ls["ok"] and ls["result"]["count"] >= 1
    cp = cc.run("copy_path", src="data/demo_dir/note.txt", dst="data/demo_dir/note2.txt")
    assert cp["ok"]
    # delete requires confirm
    bad = cc.run("delete_path", path="data/demo_dir/note2.txt")
    assert bad["ok"] is False
    good = cc.run("delete_path", path="data/demo_dir/note2.txt", confirm=True)
    assert good["ok"]


def test_open_url_and_notify_and_audit():
    cc = ComputerControl(allow=True)
    r = cc.run("open_url", url="https://example.com")
    assert r["ok"] is True
    n = cc.run("notify", title="t", message="m")
    assert n["ok"] is True
    audit = cc.run("audit_log", limit=5)
    assert audit["ok"] and audit["result"]["total"] >= 2


def test_confirm_gates_for_input_and_clipboard():
    cc = ComputerControl(allow=True)
    assert cc.run("type_text", text="hi")["ok"] is False
    assert cc.run("clipboard_get")["ok"] is False
    # With confirm but missing deps — either ok or clear unavailable (not crash)
    out = cc.run("clipboard_get", confirm=True)
    assert "ok" in out


def test_shell_extra_gate():
    cc = ComputerControl(allow=True, allow_shell=False)
    r = cc.run("run_command", command="echo hi", confirm=True)
    assert r["ok"] is False and r["error"] == "shell_disabled"
    cc2 = ComputerControl(allow=True, allow_shell=True)
    r2 = cc2.run("run_command", command="echo hi", confirm=True)
    assert r2["ok"] is True
    assert "hi" in (r2["result"].get("stdout") or "")


def test_nlu_parses_natural_commands():
    cases = [
        ("computer help", "help"),
        ("computer open https://example.com", "open_url"),
        ("take a screenshot", "screenshot"),
        ("list files in data", "list_dir"),
        ("read file README.md", "read_file"),
        ("launch calculator", "launch_app"),
        ("notify hello there", "notify"),
        ("type Hello", "type_text"),
        ("press ctrl+s", "hotkey"),
        ("click 10 20", "mouse_click"),
        ("copy a.txt to b.txt", "copy_path"),
        ("delete file tmp.txt", "delete_path"),
        ("computer env", "env_info"),
        ("computer processes", "list_processes"),
    ]
    for text, expect in cases:
        parsed = parse_computer_command(text)
        assert parsed["action"] == expect, (text, parsed)


def test_handle_text_and_assistant_route(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cc = ComputerControl(allow=True, sandbox_roots=[tmp_path])
    out = cc.handle_text("computer list files in data")
    assert out["ok"] is True
    a = FinderbayAssistant(
        seed_demo=False,
        enable_llm=False,
        allow_computer_control=True,
    )
    a.computer.sandbox_roots = [tmp_path]
    res = a.handle("computer help")
    assert res["route"] == "computer_control"
    assert "ok" in res["text"].lower() or "Computer help" in res["text"]


def test_unknown_action_rejected():
    cc = ComputerControl(allow=True)
    r = cc.run("keylogger_start")
    assert r["ok"] is False
    assert "allowlisted" in r["error"]
