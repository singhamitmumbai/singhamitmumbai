"""Complete AI v1.6 — persistence, export, completeness, Hindi."""

from __future__ import annotations

from pathlib import Path

from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.response.hindi import (
    realize_alerts_hi,
    realize_materials_hi,
    realize_status_hi,
    realize_vision_hi,
)
from cognitive_ai.runtime.completeness import run_completeness_check
from cognitive_ai.runtime.persist import StateStore


def test_persist_and_export(tmp_path: Path):
    store = tmp_path / "state.json"
    ai = CompleteAI(seed_demo=True, enable_llm=False, persist_path=str(store), autosave=True)
    ai.remember("user_name", "Amit")
    ai.handle("Who are you?")
    assert store.exists()
    md = ai.export_session(fmt="markdown")
    assert "Complete AI" in md
    assert "Amit" in md or "user_name" in md
    js = ai.export_session(fmt="json")
    assert '"version"' in js

    ai2 = CompleteAI(seed_demo=False, enable_llm=False, persist_path=str(store), autosave=False)
    assert ai2.recall("user_name") == "Amit"
    assert len(ai2.session) >= 2


def test_completeness_checker():
    report = run_completeness_check()
    assert report["ok"] is True
    assert report["complete_code"] is True
    assert report["version"].startswith("1.6")


def test_hindi_realizers():
    assert "Complete AI" in realize_status_hi()
    assert "सामग्री" in realize_materials_hi({"count": 2})
    assert "अलर्ट" in realize_alerts_hi({"count": 3, "by_severity": {"critical": 1, "warn": 1, "info": 1}})
    assert "व्यक्ति" in realize_vision_hi({"people_count": 3, "engine": "stub"}, "count_people")


def test_diagnostics_includes_completeness(tmp_path: Path):
    ai = CompleteAI(
        seed_demo=True,
        enable_llm=False,
        persist_path=str(tmp_path / "s.json"),
        autosave=False,
    )
    d = ai.diagnostics()
    assert d["ok"] is True
    assert d["completeness"]["ok"] is True
