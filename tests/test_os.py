"""Tests for Cognitive OS boot, shell commands, and services."""

from pathlib import Path

from cognitive_ai.os.shell import OSShell, launch_os
from cognitive_ai.os.system import CognitiveOS
from cognitive_ai.os.services import ConsoleService, StoreService


def test_boot_and_status(tmp_path: Path):
    cos = CognitiveOS(root=tmp_path / "cos", domain="animals")
    splash = cos.boot()
    assert cos.booted
    assert "ready" in splash
    assert cos.brain.kb.domain == "animals"
    text = cos.status_text()
    assert "booted:     True" in text
    assert "animals" in text


def test_shell_commands_session(tmp_path: Path, capsys):
    cos = CognitiveOS(root=tmp_path / "cos2", domain="animals")
    shell = OSShell(cos)
    script = [
        "status",
        "assert rover is_a dog",
        "query rover is_a *",
        "goal Feed tweety | tweety fed true",
        "ps",
        "think 3",
        "ask Who are you?",
        "why tweety can fly",
        "mem 5",
        "dmesg 5",
        "save",
        "shutdown",
    ]
    rc = shell.run_script(script)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Cognitive OS" in out or "COGNITIVE" in out
    assert "tweety" in out.lower() or "goal" in out.lower()
    assert (tmp_path / "cos2" / "knowledge.json").exists()


def test_persist_reload(tmp_path: Path):
    root = tmp_path / "persist"
    cos = CognitiveOS(root=root, domain="animals")
    cos.boot()
    cos.brain.assert_fact("alpha", "rel", "beta", learn=True)
    cos.store.save_knowledge(cos.brain)
    cos.shutdown()

    cos2 = CognitiveOS(root=root, domain="kitchen")
    cos2.boot(load_persisted=True)
    assert cos2.brain.kb.has_fact("alpha", "rel", "beta")


def test_ps_and_mem(tmp_path: Path):
    cos = CognitiveOS(root=tmp_path / "ps", domain="animals")
    cos.boot()
    cos.brain.set_goal("Feed tweety", "tweety", "fed", "true")
    assert "Feed tweety" in cos.ps_text()
    assert "Domain: animals" in cos.mem_text(5)


def test_scheduler_tick(tmp_path: Path):
    cos = CognitiveOS(root=tmp_path / "sched", domain="animals", auto_sched=False)
    cos.boot()
    cos.brain.set_goal("Feed tweety", "tweety", "fed", "true")
    report = cos.scheduler.tick_once(steps=2)
    assert hasattr(report, "stopped_reason")
    assert cos.scheduler.ticks == 1
    cos.scheduler.start()
    assert cos.scheduler.running
    cos.scheduler.pause()
    assert cos.scheduler.paused
    cos.scheduler.resume()
    cos.scheduler.stop()
    assert not cos.scheduler.running


def test_console_service(tmp_path: Path):
    log = tmp_path / "c.log"
    c = ConsoleService(log_path=log)
    c.info("hello")
    c.warn("careful")
    assert len(c.recent()) == 2
    assert log.exists()
    assert "hello" in log.read_text()


def test_launch_os_demo(tmp_path: Path, capsys):
    rc = launch_os(
        root=str(tmp_path / "demo_os"),
        domain="animals",
        script=[
            "status",
            "think 1",
            "shutdown",
        ],
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "shutdown" in out.lower() or "goodbye" in out.lower()


def test_store_summary(tmp_path: Path):
    store = StoreService(tmp_path / "s")
    store.ensure()
    s = store.summary()
    assert s["knowledge_exists"] is False
    assert "knowledge.json" in s["knowledge"]
