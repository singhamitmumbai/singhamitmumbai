"""Unified Proto-AGI stack facade — wires all gap upgrades together (v1.16)."""

from __future__ import annotations

from typing import Any, Optional

from ..memory.store import MemorySystem
from ..version import __version__
from .alignment import AlignmentGuard
from .continual_skills import GrowingSkillLibrary
from .deep_transfer import DeepTransferEngine
from .general_learning import GeneralLearner
from .lifelong_memory import LifelongMemory
from .long_horizon import LongHorizonAgent
from .metacognition import MetaCognition
from .multimodal_hub import AlwaysOnMultimodalHub
from .world_knowledge import WorldKnowledgeHub


class ProtoAGIStack:
    """Progress toward broader general intelligence — NOT human-level AGI."""

    def __init__(self, memory: Optional[MemorySystem] = None) -> None:
        self.version = __version__
        self.memory = memory or MemorySystem()
        self.world = WorldKnowledgeHub(expand=True)
        self.skills = GrowingSkillLibrary(self.memory)
        self.learner = GeneralLearner(self.memory, self.skills)
        self.transfer = DeepTransferEngine(self.memory)
        self.horizon = LongHorizonAgent(self.memory)
        self.multimodal = AlwaysOnMultimodalHub()
        self.lifelong = LifelongMemory(self.memory)
        self.meta = MetaCognition()
        self.align = AlignmentGuard()

    def status(self) -> dict[str, Any]:
        return {
            "name": "ProtoAGIStack",
            "version": self.version,
            "human_level_agi": False,
            "modules": {
                "world_knowledge": self.world.stats(),
                "general_learning": self.learner.summary(),
                "continual_skills": self.skills.summary(),
                "lifelong_memory": self.lifelong.summary(),
                "multimodal": self.multimodal.status(),
                "metacognition": self.meta.self_model(),
                "alignment": self.align.honesty_badge(),
                "transfer_overlap_demo": self.transfer.structural_overlap("kitchen", "hospital"),
            },
            "note": (
                "v1.16 deepens learning, open-world, transfer, long-horizon, multimodal, "
                "lifelong memory, metacognition, alignment — still not AGI."
            ),
        }

    def handle(self, text: str, *, context: Optional[dict] = None) -> dict[str, Any]:
        """Meta-routed handle: align → open-world → debate → execute → critique → learn."""
        verdict = self.align.check(text)
        if not verdict.allowed:
            return {
                "text": self.align.wrap_response("", verdict),
                "ok": False,
                "route": "alignment_block",
                "alignment": verdict.to_dict(),
                "human_level_agi": False,
            }

        ow = self.world.understand(text)
        debate = self.meta.debate_strategies(text, open_world=ow.to_dict())
        decision = self.meta.decide(text, open_world=ow.to_dict())
        # Prefer debate winner when confidence close
        if debate.get("winner") and decision.confidence < 0.7:
            decision.strategy = str(debate["winner"])
            decision.reasons = list(decision.reasons) + ["debate_winner"]
        ctx = dict(context or {})

        if decision.should_abstain or decision.strategy == "abstain":
            text_out = self.align.wrap_response(
                decision.next_questions[0] if decision.next_questions else "I should not proceed.",
                verdict,
            )
            return self._finish(text_out, decision, ow, verdict, debate, route="abstain", ok=False)

        if decision.strategy == "ask_clarify":
            q = decision.next_questions[0] if decision.next_questions else "Could you clarify?"
            return self._finish(q, decision, ow, verdict, debate, route="clarify", ok=True)

        if decision.strategy == "long_horizon":
            result = self.horizon.run(text, context=ctx)
            body = (
                f"Long-horizon run **{result['goal']['status']}** "
                f"({len(result['goal']['steps'])} steps, replans noted in reflections). "
                f"Reflections: {'; '.join((result['goal'].get('reflections') or [])[-3:]) or 'n/a'}"
            )
            body = self.align.wrap_response(body, verdict)
            if result.get("ok"):
                eps = self.memory.episodes[-1] if self.memory.episodes else None
                if eps:
                    self.skills.learn_from_episode(eps, domain="autonomy")
                    self.learner.observe(
                        kind="skill",
                        payload={
                            "name": f"horizon_{eps.episode_id[:8]}",
                            "steps": [a.get("op") for a in eps.actions if isinstance(a, dict)],
                            "domain": "autonomy",
                            "summary": eps.summary,
                        },
                        verified=True,
                        source="horizon",
                    )
                    self.lifelong.remember_interaction(
                        eps.summary, actions=eps.actions, outcome=eps.outcome
                    )
            return self._finish(
                body,
                decision,
                ow,
                verdict,
                debate,
                route="long_horizon",
                ok=result.get("ok"),
                extra={"horizon": result},
            )

        if decision.strategy == "transfer":
            overlap = self.transfer.structural_overlap("finderbay", "coding")
            msg = (
                "Deep transfer ready. "
                f"Example overlap finderbay↔coding roles={overlap.get('shared_roles')}. "
                "Seed a procedure then call transfer_domain / compose_transfer."
            )
            return self._finish(msg, decision, ow, verdict, debate, route="transfer", ok=True)

        if decision.strategy == "tool_vision":
            health = self.multimodal.tick_all_health()
            body = (
                "Vision/multimodal channel status: "
                + ", ".join(
                    f"{k}={v.get('engine')}/{'on' if v.get('available') else 'off'}"
                    for k, v in (health.get("channels") or {}).items()
                )
            )
            return self._finish(body, decision, ow, verdict, debate, route="multimodal", ok=True)

        # Default open-world sketch
        if ow.known_facts:
            body = (
                f"**Open-world read** (strategy={decision.strategy}):\n"
                + "\n".join(f"- {f}" for f in ow.known_facts[:8])
            )
            if ow.hypotheses:
                body += "\n\nHypotheses:\n" + "\n".join(f"- {h}" for h in ow.hypotheses[:4])
            if ow.unknowns:
                body += "\n\nUnknowns to probe: " + ", ".join(ow.unknowns[:5])
            ok = True
        else:
            probes = []
            for u in ow.unknowns[:3]:
                probes.extend(self.world.hypothesize(u)[:1])
            body = (
                "Limited local world knowledge for that. "
                + (" Probes: " + "; ".join(probes) if probes else "")
                + " Try web search, Finderbay/vision/coding tools, or a long-horizon research goal."
            )
            ok = False

        body = self.align.wrap_response(body, verdict)
        self.learner.observe(
            kind="fact" if ok else "preference",
            payload=(
                {"subject": (ow.entities[0] if ow.entities else "topic"), "relationship": "discussed", "value": text[:80]}
                if ok
                else {"key": "last_miss", "value": text[:80]}
            ),
            verified=True,
            source="stack",
        )
        self.lifelong.remember_interaction(
            summary=f"proto:{text[:80]}",
            actions=[{"op": decision.strategy}],
            outcome="success" if ok else "partial",
        )
        return self._finish(body, decision, ow, verdict, debate, route=decision.strategy, ok=ok)

    def run_self_eval(self) -> dict[str, Any]:
        from ..benchmarks.hard_novel import run_hard_novel_benchmark

        hard = run_hard_novel_benchmark(allow_live_search=False)
        tick = self.skills.continual_tick()
        cons = self.lifelong.consolidate()
        dream = self.lifelong.dream(n=3)
        multi = self.multimodal.tick_all_health()
        fused = self.multimodal.fuse(
            [
                {"modality": "text", "payload": "scene note: warehouse aisle"},
                self.multimodal.ingest("thermal hint: warm motor", modality="text"),
            ]
        )
        red = self.align.red_team_battery()
        replay = self.learner.replay(n=10)
        return {
            "hard_novel": hard,
            "continual": tick,
            "consolidation": cons.to_dict(),
            "dream": dream,
            "multimodal_health": multi,
            "multimodal_fuse": fused,
            "alignment_redteam": red,
            "learning_replay": replay,
            "world": self.world.stats(),
            "alignment": self.align.honesty_badge(),
            "human_level_agi": False,
            "ok": bool(hard.get("ok")) and bool(red.get("ok")),
            "version": self.version,
        }

    def _finish(
        self,
        text: str,
        decision,
        ow,
        verdict,
        debate,
        *,
        route: str,
        ok: bool,
        extra: Optional[dict] = None,
    ) -> dict[str, Any]:
        critique = self.meta.critique(ow.text if hasattr(ow, "text") else "", text, decision=decision)
        constitution = self.align.constitutional_check(text)
        if not constitution.get("ok"):
            text = (
                text
                + "\n\n_(Alignment note: answer adjusted for policy — human_level_agi=false.)_"
            )
        out = {
            "text": text,
            "ok": ok,
            "route": route,
            "strategy": decision.strategy,
            "meta": decision.to_dict(),
            "debate": debate,
            "critique": critique,
            "constitution": constitution,
            "open_world": ow.to_dict(),
            "alignment": verdict.to_dict(),
            "human_level_agi": False,
            "llm_dependency": False,
            "version": self.version,
            "product": "Complete AI Proto-AGI Stack",
        }
        if extra:
            out.update(extra)
        return out

    # backward-compatible alias used by older code paths
    def _pack(self, *args, **kwargs):  # pragma: no cover
        return self._finish(*args, debate={}, **kwargs)


def run_agi_gap_demo() -> dict[str, Any]:
    stack = ProtoAGIStack()
    samples = [
        stack.handle("Explain gravity and kinetic_energy in mumbai"),
        stack.handle("multi-step research photosynthesis then verify"),
        stack.handle("generate function fibonacci(n)"),
    ]
    eval_report = stack.run_self_eval()
    return {
        "status": stack.status(),
        "samples": samples,
        "eval": eval_report,
        "human_level_agi": False,
    }
