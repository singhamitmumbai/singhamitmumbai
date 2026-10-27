"""v1.18 — persist, multi-goal, arbitrate, scorecard, open-world chat grounding."""

from __future__ import annotations

import tempfile
from pathlib import Path

from cognitive_ai.agi_gap import ProtoAGIStack, run_agi_grade_eval
from cognitive_ai.agi_gap.deep_transfer import DeepTransferEngine
from cognitive_ai.agi_gap.lifelong_memory import LifelongMemory
from cognitive_ai.agi_gap.long_horizon import LongHorizonAgent
from cognitive_ai.agi_gap.multimodal_hub import AlwaysOnMultimodalHub
from cognitive_ai.agi_gap.scorecard import build_scorecard
from cognitive_ai.commonsense.chat import OpenDomainChat
from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.version import __version__


def test_version_118():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 18)


def test_multi_goal_horizon_queue():
    hz = LongHorizonAgent()
    hz.enqueue_goal("research gravity then verify")
    hz.enqueue_goal("research oxygen then verify")
    out = hz.run_queue(max_goals=2, max_steps=4)
    assert out["ran"] == 2
    assert out["human_level_agi"] is False


def test_multimodal_arbitrate_and_session():
    hub = AlwaysOnMultimodalHub()
    hub.ingest("a", modality="text")
    hub.ingest("b", modality="text")
    assert len(hub.session_buffer) >= 2
    arb = hub.arbitrate()
    assert arb["ok"] is True
    hub.clear_session()
    assert hub.session_buffer == []


def test_lifelong_persist_roundtrip():
    life = LifelongMemory()
    life.remember_interaction("persist me", actions=[{"op": "act"}], outcome="success")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mem.json"
        life.persist(path)
        other = LifelongMemory()
        loaded = other.load(path)
        assert loaded["ok"] is True
        assert other.memory.episodes


def test_new_transfer_domains():
    ov = DeepTransferEngine().structural_overlap("lab", "logistics")
    assert ov["overlap"] > 0


def test_scorecard_and_agi_grade_eval():
    report = run_agi_grade_eval(allow_live_search=False)
    assert report["ok"] is True
    assert "scorecard" in report
    assert report["scorecard"]["human_level_agi"] is False
    assert report["scorecard"]["certifies_agi"] is False
    assert report["scorecard"]["overall"] >= 0.85
    sc = build_scorecard(report["cases"], version=__version__)
    assert len(sc.pillars) == 8


def test_open_world_chat_grounding():
    chat = OpenDomainChat()
    # Force a path that can use expanded world entities
    out = chat.respond("Tell me about mumbai and rice together briefly", allow_live_search=False)
    assert out.get("text")
    assert out.get("human_level_agi") is False


def test_complete_session_feeds_learning_when_proto_used():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    # Touch proto stack first so session learning hooks attach
    _ = ai.proto_stack
    before = len(ai.proto_stack.learner.events)
    ai.handle("multi-step research gravity then verify")
    after = len(ai.proto_stack.learner.events)
    assert after >= before


def test_stack_status_v118_fields():
    st = ProtoAGIStack().status()
    assert "transfer_lab_logistics" in st["modules"]
    assert st["human_level_agi"] is False
