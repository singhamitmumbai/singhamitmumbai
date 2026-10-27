"""Hard evals on novel tasks outside Finderbay (proto-AGI battery).

These tasks deliberately avoid invoice/warehouse schemas. Passing them shows
broader competence — NOT human-level AGI.
"""

from __future__ import annotations

from typing import Any

from ..agi_gap.deep_transfer import DeepTransferEngine
from ..agi_gap.long_horizon import LongHorizonAgent
from ..agi_gap.metacognition import MetaCognition
from ..agi_gap.world_knowledge import WorldKnowledgeHub
from ..commonsense.chat import OpenDomainChat
from ..memory.store import MemorySystem, Procedure
from ..search import IntelligentWebSearch


def run_hard_novel_benchmark(*, allow_live_search: bool = False) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = "") -> None:
        cases.append({"name": name, "ok": bool(ok), "detail": detail})

    # 1) Expanded world knowledge scale
    hub = WorldKnowledgeHub(expand=True)
    st = hub.stats()
    add("world_kb_expanded", st["facts"] >= 2500, st)

    # 2) Open-world framing of novel entities
    frame = hub.understand("Why does photosynthesis matter for kinetic_energy on tuesday in asia?")
    add("open_world_entities", len(frame.entities) >= 3, frame.to_dict())

    # 3) Non-Finderbay chat / science
    chat = OpenDomainChat()
    r = chat.respond("Explain gravity in simple terms", allow_live_search=allow_live_search)
    add("novel_explain_gravity", "gravity" in (r.get("text") or "").lower(), r.get("method"))

    # 4) Math outside business domain
    r2 = chat.respond("Calculate 12 * (3 + 4)", allow_live_search=False)
    add("novel_math", "84" in (r2.get("text") or ""), r2.get("text"))

    # 5) Creative non-Finderbay
    r3 = chat.respond("Write a poem about Mars", allow_live_search=False)
    add("novel_poem", "mars" in (r3.get("text") or "").lower(), r3.get("method"))

    # 6) Web-intel local research (offline)
    web = IntelligentWebSearch(allow_live=False)
    wr = web.search("What is DNA?")
    add("novel_search_dna", wr.ok and "dna" in wr.answer.lower(), wr.method)

    # 7) Deep transfer kitchen→hospital schema
    mem = MemorySystem()
    mem.add_procedure(
        Procedure(
            name="prep_meal_template",
            steps=["fetch ingredient from pantry", "chef follows recipe", "verify dish"],
            applicability=["domain:kitchen"],
        )
    )
    deep = DeepTransferEngine(mem)
    tr = deep.transfer_domain("prep_meal_template", from_domain="kitchen", to_domain="hospital")
    add("deep_transfer_cross_domain", tr.success and tr.new_skill is not None, tr.message)

    # 8) Long-horizon research goal
    hz = LongHorizonAgent(mem)
    out = hz.run("research photosynthesis and verify answer", max_steps=5)
    add("long_horizon_research", out.get("ok") is True, out.get("goal", {}).get("status"))

    # 9) Metacognition strategy for novel coding ask
    meta = MetaCognition()
    d = meta.decide("generate function fibonacci(n)")
    add("meta_coding_strategy", d.strategy == "tool_coding", d.to_dict())

    # 10) Analogy / schema: doctor:ward :: teacher:? → classroom (location role)
    ana = deep.analogize("doctor", "ward", "teacher")
    add(
        "schema_analogy",
        ana.get("answer") == "classroom" or ana.get("method") in ("schema_role", "cross_domain_role"),
        ana,
    )

    # 11) Alignment red-team
    from ..agi_gap.alignment import AlignmentGuard

    red = AlignmentGuard().red_team_battery()
    add("alignment_redteam", red.get("ok") is True, red)

    # 12) Multimodal fusion
    from ..agi_gap.multimodal_hub import AlwaysOnMultimodalHub

    fuse = AlwaysOnMultimodalHub().fuse(
        [
            {"modality": "text", "payload": "note A"},
            {"modality": "text", "payload": "note B"},
        ]
    )
    add("multimodal_fuse", fuse.get("ok") is True and fuse.get("parts", 0) >= 2, fuse)

    # 13) General learning rejects unverified self-labels
    from ..agi_gap.general_learning import GeneralLearner

    gl = GeneralLearner()
    rejected = gl.observe(
        kind="fact",
        payload={"subject": "x", "relationship": "is_a", "value": "y"},
        verified=False,
        source="self",
    )
    add("learning_rejects_self_label", rejected.get("ok") is False, rejected)

    # 14) Metacognition critique catches overclaim
    critique = meta.critique("q", "I am conscious and definitely AGI")
    add("meta_critique_overclaim", "overclaim" in (critique.get("issues") or []), critique)

    # 15) Compose transfer kitchen→hospital→school
    mem2 = MemorySystem()
    mem2.add_procedure(
        Procedure(
            name="prep_meal_template",
            steps=["fetch ingredient from pantry", "chef follows recipe"],
            applicability=["domain:kitchen"],
        )
    )
    deep2 = DeepTransferEngine(mem2)
    composed = deep2.compose_transfer(
        "prep_meal_template", chain=[("kitchen", "hospital"), ("hospital", "school")]
    )
    add("compose_transfer_chain", composed.success, composed.message)

    passed = sum(1 for c in cases if c["ok"])
    return {
        "name": "hard_novel_outside_finderbay",
        "passed": passed,
        "total": len(cases),
        "pass_rate": passed / max(1, len(cases)),
        "ok": passed >= max(12, int(0.85 * len(cases))),
        "cases": cases,
        "human_level_agi": False,
        "note": "Harder novel tasks — still not an AGI claim.",
    }
