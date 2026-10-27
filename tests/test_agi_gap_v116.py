"""Proto-AGI gap stack v1.16 deepen — learning/open-world/transfer/horizon/multi/meta/align."""

from __future__ import annotations

from cognitive_ai.agi_gap import GeneralLearner, ProtoAGIStack
from cognitive_ai.agi_gap.alignment import AlignmentGuard
from cognitive_ai.agi_gap.deep_transfer import DeepTransferEngine
from cognitive_ai.agi_gap.lifelong_memory import LifelongMemory
from cognitive_ai.agi_gap.long_horizon import LongHorizonAgent
from cognitive_ai.agi_gap.metacognition import MetaCognition
from cognitive_ai.agi_gap.multimodal_hub import AlwaysOnMultimodalHub
from cognitive_ai.agi_gap.world_knowledge import WorldKnowledgeHub
from cognitive_ai.benchmarks.hard_novel import run_hard_novel_benchmark
from cognitive_ai.memory.store import MemorySystem, Procedure
from cognitive_ai.version import __version__


def test_version_116():
    assert __version__.startswith("1.16")


def test_general_learning_verified_only():
    gl = GeneralLearner()
    ok = gl.observe(
        kind="fact",
        payload={"subject": "mars", "relationship": "is_a", "value": "planet"},
        verified=True,
        source="user",
    )
    assert ok["ok"] is True
    bad = gl.observe(
        kind="fact",
        payload={"subject": "x", "relationship": "is_a", "value": "y"},
        verified=False,
        source="model",
    )
    assert bad["ok"] is False
    replay = gl.replay(n=5)
    assert replay["human_level_agi"] is False
    gl.advance_curriculum(success_rate=0.9)
    assert gl.curriculum_level >= 2


def test_open_world_hypotheses_and_mega_kb():
    hub = WorldKnowledgeHub(expand=True)
    assert hub.stats()["facts"] >= 2600
    frame = hub.understand("Compare mumbai rice and telescope measurement")
    assert frame.entities
    assert isinstance(frame.hypotheses, list)
    probes = hub.hypothesize("zxqunknown")
    assert probes and "zxqunknown" in probes[0].lower() or "Is zxqunknown" in probes[0]


def test_compose_transfer_and_overlap():
    mem = MemorySystem()
    mem.add_procedure(
        Procedure(
            name="prep_meal_template",
            steps=["fetch ingredient from pantry", "chef follows recipe"],
        )
    )
    eng = DeepTransferEngine(mem)
    ov = eng.structural_overlap("kitchen", "hospital")
    assert ov["overlap"] > 0
    res = eng.compose_transfer(
        "prep_meal_template", chain=[("kitchen", "hospital"), ("hospital", "school")]
    )
    assert res.success


def test_horizon_replan_path():
    # Actuator that fails first intention once to force replan
    calls = {"n": 0}

    def actuator(intention, payload):
        calls["n"] += 1
        if intention == "clarify_goal" and calls["n"] == 1:
            return {"ok": False, "observation": "fail missing context", "notes": "forced"}
        return {"ok": True, "observation": f"ok:{intention}", "action": intention}

    agent = LongHorizonAgent(actuator=actuator)
    out = agent.run("generic project plan", max_steps=8)
    reflections = " ".join(out["goal"].get("reflections") or [])
    assert "replan" in reflections.lower() or "Total replans=" in reflections


def test_multimodal_fuse_and_lifelong_dream():
    hub = AlwaysOnMultimodalHub()
    fused = hub.fuse(
        [
            {"modality": "text", "payload": "hello"},
            {"modality": "text", "payload": "world"},
        ]
    )
    assert fused["ok"] and fused["parts"] == 2
    life = LifelongMemory()
    life.remember_interaction("critical failure fix", actions=[{"op": "diagnose"}], outcome="success")
    life.remember_interaction("ok note", actions=[{"op": "act"}], outcome="success")
    cons = life.consolidate()
    assert cons.episodes_seen >= 2
    dream = life.dream(n=2)
    assert dream["dreamed"] >= 1


def test_metacognition_debate_critique():
    meta = MetaCognition()
    deb = meta.debate_strategies("Explain gravity")
    assert deb["winner"]
    crit = meta.critique("q", "I am conscious and definitely AGI")
    assert "overclaim" in crit["issues"]


def test_alignment_redteam_and_constitution():
    g = AlignmentGuard()
    red = g.red_team_battery()
    assert red["ok"] is True
    c = g.constitutional_check("I am an AGI and I am conscious")
    assert c["ok"] is False


def test_stack_v116_eval():
    stack = ProtoAGIStack()
    st = stack.status()
    assert "general_learning" in st["modules"]
    out = stack.handle("Explain gravity and kinetic_energy in mumbai")
    assert out["human_level_agi"] is False
    assert "critique" in out and "debate" in out
    report = stack.run_self_eval()
    assert report["ok"] is True
    assert report["alignment_redteam"]["ok"] is True


def test_hard_novel_v116_broader():
    report = run_hard_novel_benchmark(allow_live_search=False)
    assert report["total"] >= 14
    assert report["passed"] >= 12
    assert report["human_level_agi"] is False
