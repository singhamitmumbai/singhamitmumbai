"""JARVIS evaluation criteria (blueprint §14)."""

from __future__ import annotations

from typing import Any

from .pipeline import JarvisPipeline
from .provenance import ProvenanceMemory


def run_jarvis_eval(pipeline: JarvisPipeline | None = None) -> dict[str, Any]:
    pipe = pipeline or JarvisPipeline()
    cases: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        cases.append({"name": name, "ok": ok, "detail": detail})

    # Intent classification
    r = pipe.run("Prepare tomorrow's material shortage report for the selected site.")
    check("intent_shortage", r.intent == "inventory.shortages", r.intent)
    check("shortage_verified", r.verified and r.ok, r.response)

    r2 = pipe.run("Find the latest project report")
    check("intent_files", r2.intent == "files.search", r2.intent)

    r3 = pipe.run("Remind me tomorrow morning tender review yaad dilana")
    # may match schedule via yaad dilana
    check("intent_schedule", r3.intent == "schedule.create", r3.intent)

    r4 = pipe.run("Stop this task")
    check("stop_cancel", r4.stage == "cancelled", r4.stage)

    r5 = pipe.run("Calculate 12 + 30")
    check("math", r5.ok or r5.intent == "math.calculate", r5.intent)

    r6 = pipe.run("teleport to mars and invent AGI")
    check("reject_unsupported", r6.intent == "unsupported" or not r6.ok, r6.response[:80])

    # Provenance / supersession
    mem = ProvenanceMemory()
    f1 = mem.record_fact("ProjectA", "deadline", "2026-01-01", source_id="doc_v1", verification_status="document")
    f2 = mem.correct_fact("ProjectA", "deadline", "2026-02-15", source_id="doc_v2")
    current = mem.query(subject="ProjectA", relation="deadline")
    check("fact_supersession", len(current) == 1 and current[0].value == "2026-02-15", str(current[0].value if current else None))
    check("supersedes_link", f2.supersedes_fact_id == f1.fact_id, str(f2.supersedes_fact_id))

    # Tool contract validation
    bad = pipe.tools.invoke("inventory.shortages", {"site": "X"})  # missing date/demand/stock
    check("contract_rejects_bad_input", not bad.get("ok"), str(bad.get("details")))

    # Idempotent job attempt
    a = pipe.run("system health check are you ready", attempt_id="att-1")
    b = pipe.run("system health check are you ready", attempt_id="att-1")
    check("idempotent_attempt", a.job_id == b.job_id, f"{a.job_id} vs {b.job_id}")

    # Shortage arithmetic with committed stock
    skill = pipe.skills.run(
        "materials.shortage_report",
        site="Site-7",
        date="2026-10-28",
        demand=[{"item": "cement", "qty": 100}],
        stock=[{"item": "cement", "qty": 80}],
        committed=[{"item": "cement", "qty": 20}],
        expected_deliveries=[{"item": "cement", "qty": 40, "eta": "2026-10-29"}],
    )
    short_row = skill.output["shortages"][0]
    # available = 80-20 = 60; shortage = 40
    check("shortage_math", short_row["shortage"] == 40.0, str(short_row))
    check("deliveries_separate", len(skill.output.get("expected_deliveries") or []) == 1, "")
    check("skill_verified", skill.verified, skill.message)

    # Voice pipeline status
    from .voice import VoicePipeline

    voice = VoicePipeline()
    voice.start_listening()
    st = voice.status()
    check("voice_ptt_default", st["push_to_talk"] is True and st["wake_word"] == "disabled_prefer_ptt", str(st["backend"]))
    voice.cancel()
    check("voice_cancel_clears_queue", voice.session.cancelled and len(voice.speech_queue) == 0, "")

    passed = sum(1 for c in cases if c["ok"])
    return {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": passed / max(1, len(cases)),
        "cases": cases,
        "controller": "non_llm",
        "human_level_agi": False,
    }
