"""JARVIS Tools & System Blueprint — Finderbay integration tests."""

from __future__ import annotations

from cognitive_ai.assistant import FinderbayAssistant
from cognitive_ai.jarvis import JarvisPipeline, run_jarvis_eval
from cognitive_ai.jarvis.contracts import JobState, ToolContract
from cognitive_ai.jarvis.provenance import ProvenanceMemory
from cognitive_ai.jarvis.skills import MaterialShortageSkill
from cognitive_ai.jarvis.voice import VoicePipeline


def test_tool_contract_validation():
    pipe = JarvisPipeline()
    bad = pipe.tools.invoke("documents.create_report", {})
    assert bad["ok"] is False
    assert bad["error"] == "validation_failed"
    good = pipe.tools.invoke(
        "documents.create_report",
        {"title": "Test", "sections": {"a": 1}},
    )
    assert good["ok"] is True
    assert good.get("verified") is True


def test_job_states_and_cancel():
    pipe = JarvisPipeline()
    r = pipe.run("Stop this task")
    assert r.stage == "cancelled"
    job = pipe.jobs.get(r.job_id)
    assert job is not None
    assert job.state == JobState.CANCELLED.value


def test_idempotent_attempt_id():
    pipe = JarvisPipeline()
    a = pipe.run("are you ready", attempt_id="x-1")
    b = pipe.run("are you ready", attempt_id="x-1")
    assert a.job_id == b.job_id


def test_material_shortage_skill_verified():
    pipe = JarvisPipeline()
    skill = MaterialShortageSkill(pipe.tools)
    result = skill.run(
        site="Site-7",
        date="2026-10-28",
        demand=[{"item": "cement", "qty": 100}, {"item": "steel", "qty": 20}],
        stock=[{"item": "cement", "qty": 90}, {"item": "steel", "qty": 20}],
        committed=[{"item": "cement", "qty": 10}],
        expected_deliveries=[{"item": "cement", "qty": 50, "eta": "2026-10-29"}],
    )
    assert result.verified
    cement = next(s for s in result.output["shortages"] if s["item"] == "cement")
    # available = 90-10 = 80; shortage = 20
    assert cement["shortage"] == 20.0
    assert result.output["expected_deliveries"][0]["qty"] == 50


def test_pipeline_shortage_report():
    pipe = JarvisPipeline()
    pipe.memory.set_working(active_project="Site-7")
    r = pipe.run("Prepare tomorrow's material shortage report for the selected site.")
    assert r.intent == "inventory.shortages"
    assert r.verified
    assert r.job_id
    job = pipe.jobs.get(r.job_id)
    assert job.state == JobState.COMPLETED.value


def test_provenance_supersession():
    mem = ProvenanceMemory()
    f1 = mem.record_fact("P1", "deadline", "2026-01-01", source_id="v1", verification_status="document")
    f2 = mem.correct_fact("P1", "deadline", "2026-03-01")
    cur = mem.query(subject="P1", relation="deadline")
    assert len(cur) == 1
    assert cur[0].value == "2026-03-01"
    assert f2.supersedes_fact_id == f1.fact_id
    assert mem.facts[f1.fact_id].valid_until is not None


def test_hinglish_schedule_slots():
    pipe = JarvisPipeline()
    r = pipe.run("Jarvis, kal subah 10 baje tender review yaad dilana.")
    # stripped jarvis prefix happens in assistant; pipeline gets full text
    # Intent should still match yaad dilana
    intent, slots, _ = pipe.interpret(pipe.capture("kal subah 10 baje tender review yaad dilana"))
    assert intent == "schedule.create"
    assert slots.get("local_time") == "10:00"
    assert slots.get("relative_date") == "tomorrow"
    assert "tender review" in (slots.get("task") or "").lower()


def test_voice_push_to_talk_first():
    v = VoicePipeline()
    st = v.status()
    assert st["push_to_talk"] is True
    assert st["wake_word"] == "disabled_prefer_ptt"
    v.speak("hello")
    v.cancel()
    assert v.session.cancelled
    assert len(v.speech_queue) == 0


def test_jarvis_eval_suite():
    report = run_jarvis_eval()
    assert report["human_level_agi"] is False
    assert report["passed"] == report["total"]
    assert report["pass_rate"] == 1.0


def test_assistant_jarvis_route():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    a.jarvis.memory.set_working(active_project="Site-7")
    out = a.handle("Prepare tomorrow's material shortage report for the selected site.")
    assert out["route"] == "jarvis"
    assert out["verified"] is True
    assert "shortage" in out["text"].lower() or "verified" in out["text"].lower()
    assert "jarvis" in a.status()
    assert "jarvis" in a.complete.tools.handlers
    ev = a.handle("jarvis eval")
    assert ev["route"] == "jarvis_eval"
    assert ev["structured"]["passed"] == ev["structured"]["total"]


def test_tool_contract_manifest_fields():
    c = ToolContract(
        name="documents.create_project_report",
        version="1.0",
        input_schema={"title": "str"},
        effect="local_write",
        scope="approved_report_folder",
        verification="reopen_and_check_fields",
    )
    d = c.to_dict()
    assert d["effect"] == "local_write"
    assert d["retry_policy"] == "reconcile_before_retry"
