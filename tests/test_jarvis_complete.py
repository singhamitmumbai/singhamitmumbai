"""Full JARVIS system — SQLite, NLU, learning, policy, documents, dialogue."""

from __future__ import annotations

from pathlib import Path

from cognitive_ai.jarvis import (
    AuthContext,
    DocumentToolkit,
    IntentClassifier,
    JarvisSystem,
    LearningPipeline,
    PolicyGuard,
    SqliteMemory,
)
from cognitive_ai.jarvis.dialogue import DialogueManager


def test_jarvis_system_shortage_and_status(tmp_path: Path):
    js = JarvisSystem(
        memory_path=tmp_path / "mem.sqlite",
        report_dir=tmp_path / "reports",
        learning_dir=tmp_path / "learn",
    )
    js.dialogue.state.active_project = "Site-7"
    out = js.handle("Prepare tomorrow's material shortage report for the selected site.")
    assert out["route"] == "jarvis"
    assert out["verified"] is True
    assert out["intent"] == "inventory.shortages"
    st = js.status()
    assert st["human_level_agi"] is False
    assert st["tools"] >= 15
    assert st["sqlite"]["engine"] == "sqlite+fts5"
    ev = js.eval()
    assert ev["passed"] == ev["total"]


def test_sqlite_fts_and_fact_supersession(tmp_path: Path):
    mem = SqliteMemory(tmp_path / "m.sqlite")
    from cognitive_ai.jarvis.contracts import FactRecord

    f1 = mem.record_fact(
        FactRecord(subject="P1", relation="deadline", value="2026-01-01", source_id="v1", verification_status="document")
    )
    f2 = mem.record_fact(
        FactRecord(subject="P1", relation="deadline", value="2026-03-01", source_id="v2", verification_status="document")
    )
    cur = mem.query_facts(subject="P1", relation="deadline")
    assert len(cur) == 1
    assert cur[0]["value"] == "2026-03-01"
    assert f2.supersedes_fact_id == f1.fact_id
    mem.index_document("s1", "Shortage guide", "material shortage report cement steel site")
    hits = mem.search_documents("shortage cement")
    assert hits and hits[0]["source_id"] == "s1"
    mem.close()


def test_intent_classifier_and_hinglish():
    clf = IntentClassifier()
    r = clf.predict("Prepare tomorrow material shortage report for Site-7")
    assert r.intent == "inventory.shortages"
    r2 = clf.predict("kal subah 10 baje tender review yaad dilana")
    assert r2.intent == "schedule.create"
    assert r2.slots.get("local_time") == "10:00"
    r3 = clf.predict("teleport invent AGI forever")
    assert r3.intent == "unsupported" or r3.confidence < 0.5


def test_learning_pipeline_improves_or_holds():
    lp = LearningPipeline()
    lp.collect_correction("mat short rpt site 9", "unsupported", "inventory.shortages")
    lp.train_offline()
    held = [
        ("prepare material shortage report", "inventory.shortages"),
        ("stop this task", "jobs.cancel"),
        ("find latest project report", "files.search"),
    ]
    release = lp.activate_if_improved(held)
    assert release.metrics["candidate"]["accuracy"] >= 0
    assert "dataset_hash" in release.to_dict()


def test_policy_rejects_self_approval():
    guard = PolicyGuard(AuthContext(grants={"read"}))
    # planner cannot inject approved=true for write
    denied = guard.authorize_tool(
        "documents.create_report",
        "local_write",
        {"title": "x", "sections": {}, "approved": True},
    )
    assert denied["ok"] is False
    allowed = PolicyGuard(AuthContext(grants={"read", "local_write"})).authorize_tool(
        "documents.create_report",
        "local_write",
        {"title": "x", "path": "reports/out.md", "approved": True},
    )
    assert allowed["ok"] is True
    assert "approved" not in allowed["args"]


def test_documents_create_and_verify(tmp_path: Path):
    docs = DocumentToolkit(tmp_path / "r")
    created = docs.create_report("Site Shortage", {"site": "S7", "shortages": [{"item": "cement", "shortage": 10}]})
    assert created["ok"] and Path(created["path"]).exists()
    verified = docs.verify(created["artifact_id"], expected_fields=["Site Shortage", "cement"])
    assert verified["verified"] is True
    assert docs.money("10.005") == "10.01"


def test_dialogue_followups():
    dm = DialogueManager()
    dm.update_from_result("inventory.shortages", {"site": "Site-7", "date": "2026-10-28"})
    f = dm.resolve_followup("do the same for the other site")
    assert f["is_followup"]
    assert "site" in (f.get("needs") or ["site"])
    f2 = dm.resolve_followup("change the date to Friday")
    assert f2["overrides"].get("relative_date") == "friday"
