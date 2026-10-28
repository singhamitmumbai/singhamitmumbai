"""24/7 continuous self-learning — verified only, human_level_agi=false."""

from __future__ import annotations

import time
from pathlib import Path

from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.learning_sys.continuous import ContinuousLearner
from cognitive_ai.learning_sys.journal import LearningJournal
from cognitive_ai.version import __version__


def test_version_learning_v125():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 25)


def test_journal_append(tmp_path: Path):
    j = LearningJournal(tmp_path / "j.jsonl")
    j.append("obs", {"route": "coding"}, source="test")
    rows = j.iter_recent(10)
    assert len(rows) == 1
    assert rows[0]["human_level_agi"] is False
    assert j.count() == 1


def test_rejects_unverified_self_label(tmp_path: Path):
    cl = ContinuousLearner(store_dir=tmp_path / "cl", enabled=True, interval_sec=5)
    bad = cl.observe(text="x", route="test", source="self", verified=False)
    assert bad["ok"] is False
    assert "rejected" in bad["error"]


def test_tick_drains_and_learns(tmp_path: Path):
    cl = ContinuousLearner(store_dir=tmp_path / "cl2", enabled=True, interval_sec=5)
    for i in range(5):
        cl.observe(text=f"Explain gravity {i}", route="open_domain", ok=True, source="session")
    cl.observe(text="fail case", route="coding", ok=False, source="session")
    report = cl.tick()
    assert report["observations_consumed"] >= 5
    assert "drain_inbox" in report["jobs_run"]
    assert "consolidate" in report["jobs_run"]
    assert report["human_level_agi"] is False
    assert cl.total_ticks >= 1
    st = cl.status()
    assert st["enabled"] is True
    assert st["journal_events"] >= 1


def test_daemon_start_stop(tmp_path: Path):
    cl = ContinuousLearner(store_dir=tmp_path / "cl3", enabled=True, interval_sec=5)
    cl.observe(text="hello", route="open_domain", ok=True)
    started = cl.start(daemon=True)
    assert started["ok"] is True
    assert cl.status()["running"] is True
    # force a tick while running
    cl.tick()
    stopped = cl.stop()
    assert stopped["ok"] is True
    assert cl.status()["running"] is False


def test_complete_ai_learning_api(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("COMPLETE_AI_CONTINUOUS_LEARNING", "0")
    # Use isolated learner via property override after construct
    ai = CompleteAI(seed_demo=False, enable_llm=False)
    ai._continuous = ContinuousLearner(store_dir=tmp_path / "cai", enabled=False, interval_sec=5)
    out = ai.handle("Explain gravity")
    assert out.get("thinking") or out.get("text")
    # queued even if daemon off
    assert ai.continuous.status()["inbox"] >= 1 or ai.continuous.stats.get("observations", 0) >= 1
    ai.continuous.enable()
    tick = ai.learning_tick()
    assert tick.get("jobs_run")
    st = ai.learning_status()
    assert st["human_level_agi"] is False
    started = ai.start_learning(interval_sec=5)
    assert started.get("ok") is True
    ai.stop_learning()


def test_assistant_learning_commands(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("COMPLETE_AI_CONTINUOUS_LEARNING", "0")
    from cognitive_ai.assistant import FinderbayAssistant

    a = FinderbayAssistant(seed_demo=False, enable_llm=False)
    a.complete._continuous = ContinuousLearner(
        store_dir=tmp_path / "asst", enabled=False, interval_sec=5
    )
    st = a.handle("learning status")
    assert st["route"] == "continuous_learning"
    start = a.handle("start continuous learning")
    assert start["route"] == "continuous_learning"
    tick = a.handle("learning tick")
    assert tick["route"] == "continuous_learning"
    stop = a.handle("stop continuous learning")
    assert stop["route"] == "continuous_learning"


def test_snapshot_persists(tmp_path: Path):
    d = tmp_path / "snap"
    cl = ContinuousLearner(store_dir=d, enabled=True, interval_sec=5)
    cl.observe(text="x", route="open_domain", ok=True)
    cl.tick()
    assert (d / "snapshot.json").exists()
    cl2 = ContinuousLearner(store_dir=d, enabled=True, interval_sec=5)
    assert cl2.total_ticks >= 1
