"""Unified Proto-AGI stack facade — wires all gap upgrades together."""

from __future__ import annotations

from typing import Any, Optional

from ..memory.store import MemorySystem
from ..version import __version__
from .alignment import AlignmentGuard
from .continual_skills import GrowingSkillLibrary
from .deep_transfer import DeepTransferEngine
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
                "continual_skills": self.skills.summary(),
                "lifelong_memory": self.lifelong.summary(),
                "multimodal": self.multimodal.status(),
                "metacognition": self.meta.self_model(),
                "alignment": self.align.honesty_badge(),
            },
            "note": (
                "Upgrades learning, open-world, transfer, long-horizon, multimodal, "
                "lifelong memory, metacognition, alignment — still not AGI."
            ),
        }

    def handle(self, text: str, *, context: Optional[dict] = None) -> dict[str, Any]:
        """Meta-routed handle: align → open-world → strategy → execute scaffold."""
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
        decision = self.meta.decide(text, open_world=ow.to_dict())
        ctx = dict(context or {})

        if decision.should_abstain or decision.strategy == "abstain":
            text_out = self.align.wrap_response(
                decision.next_questions[0] if decision.next_questions else "I should not proceed.",
                verdict,
            )
            return self._pack(text_out, decision, ow, verdict, route="abstain", ok=False)

        if decision.strategy == "ask_clarify":
            q = decision.next_questions[0] if decision.next_questions else "Could you clarify?"
            return self._pack(q, decision, ow, verdict, route="clarify", ok=True)

        if decision.strategy == "long_horizon":
            result = self.horizon.run(text, context=ctx)
            body = (
                f"Long-horizon run **{result['goal']['status']}** "
                f"({len(result['goal']['steps'])} steps). "
                f"Reflections: {'; '.join(result['goal'].get('reflections') or []) or 'n/a'}"
            )
            body = self.align.wrap_response(body, verdict)
            # Grow skill from episode if succeeded
            if result.get("ok"):
                eps = self.memory.episodes[-1] if self.memory.episodes else None
                if eps:
                    self.skills.learn_from_episode(eps, domain="autonomy")
                    self.lifelong.remember_interaction(
                        eps.summary, actions=eps.actions, outcome=eps.outcome
                    )
            return self._pack(
                body, decision, ow, verdict, route="long_horizon", ok=result.get("ok"), extra={"horizon": result}
            )

        if decision.strategy == "transfer":
            # Demo: project rename if pattern present
            msg = (
                "Transfer engine ready. Seed a procedure then call "
                "transfer_domain / transfer_project. "
                f"Known domains: finderbay, kitchen, hospital, school, coding."
            )
            return self._pack(msg, decision, ow, verdict, route="transfer", ok=True)

        # Default: open-world answer sketch from KB
        if ow.known_facts:
            body = (
                f"**Open-world read** (strategy={decision.strategy}):\n"
                + "\n".join(f"- {f}" for f in ow.known_facts[:8])
            )
            if ow.unknowns:
                body += "\n\nUnknowns to probe: " + ", ".join(ow.unknowns[:5])
            ok = True
        else:
            body = (
                "Limited local world knowledge for that. "
                "Try web search, a Finderbay/vision/coding tool, or a long-horizon research goal."
            )
            ok = False
        body = self.align.wrap_response(body, verdict)
        reflection = self.meta.reflect(decision, {"ok": ok})
        self.lifelong.remember_interaction(
            summary=f"proto:{text[:80]}",
            actions=[{"op": decision.strategy}],
            outcome="success" if ok else "partial",
        )
        return self._pack(
            body,
            decision,
            ow,
            verdict,
            route=decision.strategy,
            ok=ok,
            extra={"reflection": reflection},
        )

    def run_self_eval(self) -> dict[str, Any]:
        from ..benchmarks.hard_novel import run_hard_novel_benchmark

        hard = run_hard_novel_benchmark(allow_live_search=False)
        # Continual tick
        tick = self.skills.continual_tick()
        cons = self.lifelong.consolidate()
        multi = self.multimodal.tick_all_health()
        return {
            "hard_novel": hard,
            "continual": tick,
            "consolidation": cons.to_dict(),
            "multimodal_health": multi,
            "world": self.world.stats(),
            "alignment": self.align.honesty_badge(),
            "human_level_agi": False,
            "ok": bool(hard.get("ok")),
            "version": self.version,
        }

    def _pack(
        self,
        text: str,
        decision,
        ow,
        verdict,
        *,
        route: str,
        ok: bool,
        extra: Optional[dict] = None,
    ) -> dict[str, Any]:
        out = {
            "text": text,
            "ok": ok,
            "route": route,
            "strategy": decision.strategy,
            "meta": decision.to_dict(),
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


def run_agi_gap_demo() -> dict[str, Any]:
    stack = ProtoAGIStack()
    samples = [
        stack.handle("Explain gravity and kinetic_energy"),
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
