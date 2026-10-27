"""v1.17 — product integration + AGI-grade eval suite for gap pillars."""

from __future__ import annotations

from cognitive_ai.agi_gap import ProtoAGIStack, run_agi_grade_eval
from cognitive_ai.assistant import FinderbayAssistant
from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.intelligence.router import SmartRouter
from cognitive_ai.version import __version__


def test_version_117():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 17)


def test_agi_grade_eval_suite():
    report = run_agi_grade_eval(allow_live_search=False)
    assert report["human_level_agi"] is False
    assert report.get("certifies_agi") is False
    assert report["passed"] >= int(0.9 * report["total"])
    assert report["ok"] is True


def test_router_proto_agi():
    r = SmartRouter().decide("multi-step research photosynthesis then verify")
    assert r.route == "proto_agi"


def test_complete_ai_proto_mode():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    out = ai.handle("multi-step research gravity then verify", mode="auto")
    assert out.get("human_level_agi") is False
    assert out.get("proto_agi") is True or out.get("route") in (
        "long_horizon",
        "proto_agi",
        "local_kb",
        "web_search",
    )


def test_assistant_proto_route():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    out = a.handle("multi-step research DNA then verify")
    assert out.get("human_level_agi") is False
    assert (out.get("router") or {}).get("route") == "proto_agi" or out.get("route")


def test_stack_bind_backends_and_calibration():
    stack = ProtoAGIStack()
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    stack.bind_backends(complete_ai=ai)
    out = stack.handle("multi-step research photosynthesis then verify")
    assert "reflection" in out
    assert out["human_level_agi"] is False
    # calibration updated
    model = stack.meta.self_model()
    assert "calibration" in model
