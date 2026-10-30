"""Intelligence layer — router, confidence, self-check, stronger NLU/retrieval."""

from __future__ import annotations

from cognitive_ai.assistant import FinderbayAssistant
from cognitive_ai.intelligence import SelfCheck, SmartRouter, harden_text, score_response
from cognitive_ai.jarvis import JarvisPipeline
from cognitive_ai.memory.retrieval import BM25Index


def test_harden_text():
    assert harden_text("  hello\t\tworld  ") == "hello world"
    assert "\x00" not in harden_text("a\x00b")
    assert harden_text("x" * 9000).endswith("…")


def test_smart_router_routes():
    r = SmartRouter()
    assert r.decide("Prepare material shortage report for Site-7").route == "jarvis"
    assert r.decide("Analyze thermal image ./panel_thermal.jpg").route == "vision_thermal"
    assert r.decide("Understand the scene in image ./crowd_people.jpg").route in (
        "vision_understand",
        "vision",
    )
    assert r.decide("generate function add(a, b)").route == "coding"
    assert r.decide("Show unpaid invoices for Project A").route == "finderbay"
    assert r.decide("remember that warehouse is Central").route == "memory"


def test_confidence_bands():
    high = score_response(verified=True, supported=True, intent_confidence=0.9)
    assert high.band == "high"
    low = score_response(verified=False, supported=True, missing_fields=["site", "date", "qty"], engine_stub=True)
    assert low.band in ("low", "medium", "abstain")
    abs_ = score_response(supported=False)
    assert abs_.band == "abstain"


def test_self_check():
    report = SelfCheck().run_core_imports().run_invariants().summary()
    assert report["ok"] is True
    assert report["human_level_agi"] is False


def test_jarvis_hybrid_interpret():
    pipe = JarvisPipeline()
    intent, slots, _ = pipe.interpret(pipe.capture("Prepare tomorrow material shortage report for Site-7"))
    assert intent == "inventory.shortages"
    assert slots.get("_interpret_method") == "hybrid"
    assert float(slots.get("_intent_confidence") or 0) > 0


def test_bm25_fuzzy_and_char_boost():
    idx = BM25Index()
    idx.build(
        [{"id": 1, "text": "cement shortage report site seven"}, {"id": 2, "text": "invoice unpaid project a"}],
        text_fn=lambda d: d["text"],
    )
    hits = idx.search("cement shortge site", top_k=2)  # typo shortge
    assert hits
    assert hits[0][1]["id"] == 1


def test_assistant_uses_router_and_self_check():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    out = a.handle("Show unpaid invoices for Project A")
    assert out.get("router")
    assert out["router"]["route"] == "finderbay"
    sc = a.handle("self check")
    assert sc["route"] == "self_check"
    assert sc["structured"]["ok"] is True
