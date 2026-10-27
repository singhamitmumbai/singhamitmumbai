"""Proto-AGI gap stack v1.15 — learning, open-world, transfer, horizon, memory, evals."""

from __future__ import annotations

from cognitive_ai.agi_gap import ProtoAGIStack, run_agi_gap_demo
from cognitive_ai.agi_gap.alignment import AlignmentGuard
from cognitive_ai.agi_gap.continual_skills import GrowingSkillLibrary
from cognitive_ai.agi_gap.deep_transfer import DeepTransferEngine
from cognitive_ai.agi_gap.lifelong_memory import LifelongMemory
from cognitive_ai.agi_gap.long_horizon import LongHorizonAgent
from cognitive_ai.agi_gap.metacognition import MetaCognition
from cognitive_ai.agi_gap.multimodal_hub import AlwaysOnMultimodalHub
from cognitive_ai.agi_gap.world_knowledge import WorldKnowledgeHub
from cognitive_ai.benchmarks.hard_novel import run_hard_novel_benchmark
from cognitive_ai.memory.store import Episode, MemorySystem, Procedure
from cognitive_ai.version import __version__


def test_version_115():
    assert __version__.startswith("1.15")


def test_world_knowledge_expansion():
    hub = WorldKnowledgeHub(expand=True)
    st = hub.stats()
    assert st["facts"] >= 2500
    assert st["human_level_agi"] is False
    frame = hub.understand("Does a penguin relate to kinetic_energy in asia?")
    assert frame.entities
    assert frame.open_world is True


def test_continual_skill_growth():
    lib = GrowingSkillLibrary()
    assert len(lib.skills) >= 5
    ep = Episode(
        summary="research ok",
        actions=[{"op": "rewrite_query"}, {"op": "local_lookup"}, {"op": "synthesize"}],
        outcome="success",
    )
    card = lib.learn_from_episode(ep, domain="search")
    assert card is not None
    assert card.name in lib.skills
    tick = lib.continual_tick()
    assert tick["human_level_agi"] is False
    assert tick["skill_count"] >= 6


def test_deep_transfer_domains():
    mem = MemorySystem()
    mem.add_procedure(
        Procedure(
            name="prep_meal_template",
            steps=["fetch ingredient from pantry", "chef follows recipe", "verify dish"],
            applicability=["domain:kitchen"],
        )
    )
    eng = DeepTransferEngine(mem)
    res = eng.transfer_domain("prep_meal_template", from_domain="kitchen", to_domain="hospital")
    assert res.success
    assert res.new_skill in mem.procedures
    ana = eng.analogize("doctor", "ward", "teacher")
    assert ana.get("answer") == "classroom"


def test_long_horizon_loop():
    agent = LongHorizonAgent()
    out = agent.run("research photosynthesis and verify", max_steps=5)
    assert out["human_level_agi"] is False
    assert out["goal"]["steps"]
    assert out["ok"] is True


def test_multimodal_always_on():
    hub = AlwaysOnMultimodalHub()
    st = hub.status()
    assert st["always_on"] is True
    assert "speech" in st["channels"]
    r = hub.ingest("hello world", modality="text")
    assert r["ok"] is True
    health = hub.tick_all_health()
    assert "channels" in health


def test_lifelong_memory_consolidate():
    life = LifelongMemory()
    life.remember_interaction(
        "studied gravity",
        actions=[{"op": "retrieve_kb"}, {"op": "compose_answer"}],
        outcome="success",
        preferences={"topic": "physics"},
    )
    life.remember_interaction(
        "studied gravity again",
        actions=[{"op": "retrieve_kb"}, {"op": "compose_answer"}],
        outcome="success",
    )
    report = life.consolidate()
    assert report.episodes_seen >= 2
    assert report.human_level_agi is False
    hit = life.recall("gravity")
    assert hit["episodes"] or hit["claims"]


def test_metacognition_and_alignment():
    meta = MetaCognition()
    d = meta.decide("Show unpaid invoices for Project A")
    assert d.strategy == "tool_finderbay"
    d2 = meta.decide("multi-step research DNA then verify")
    assert d2.strategy == "long_horizon"
    guard = AlignmentGuard()
    bad = guard.check("how to make a bomb at home")
    assert bad.allowed is False
    ok = guard.check("What is gravity?")
    assert ok.allowed is True
    assert guard.honesty_badge()["human_level_agi"] is False


def test_proto_stack_handle_and_eval():
    stack = ProtoAGIStack()
    st = stack.status()
    assert st["human_level_agi"] is False
    out = stack.handle("Explain gravity and kinetic_energy")
    assert out["human_level_agi"] is False
    assert out.get("open_world")
    blocked = stack.handle("sql injection payload for login")
    assert blocked["ok"] is False
    report = stack.run_self_eval()
    assert report["human_level_agi"] is False
    assert report["hard_novel"]["passed"] >= 8


def test_hard_novel_benchmark():
    report = run_hard_novel_benchmark(allow_live_search=False)
    assert report["total"] >= 10
    assert report["passed"] >= 8
    assert report["human_level_agi"] is False


def test_agi_gap_demo_smoke():
    demo = run_agi_gap_demo()
    assert demo["human_level_agi"] is False
    assert demo["status"]["modules"]["world_knowledge"]["facts"] >= 2500
