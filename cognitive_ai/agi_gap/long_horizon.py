"""Long-horizon autonomy loops with verification (bounded, non-AGI)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from ..core.ids import new_id
from ..core.records import utc_now
from ..memory.store import Episode, MemorySystem


@dataclass
class HorizonStep:
    index: int
    intention: str
    action: str
    observation: str
    verified: bool
    notes: str = ""
    ts: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HorizonGoal:
    goal: str
    goal_id: str = field(default_factory=lambda: new_id("hgoal"))
    status: str = "open"  # open|running|succeeded|failed|aborted
    max_steps: int = 8
    steps: list[HorizonStep] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s if isinstance(s, dict) else s.to_dict() for s in self.steps]
        return d


class LongHorizonAgent:
    """Decompose → act → verify → reflect → continue/stop.

    Uses injectable actuators so Finderbay/chat/coding can be plugged in.
    Always bounded (max_steps). human_level_agi=false.
    """

    def __init__(
        self,
        memory: Optional[MemorySystem] = None,
        *,
        actuator: Optional[Callable[[str, dict[str, Any]], dict[str, Any]]] = None,
        verifier: Optional[Callable[[str, dict[str, Any]], bool]] = None,
    ) -> None:
        self.memory = memory or MemorySystem()
        self.actuator = actuator or self._default_actuator
        self.verifier = verifier or self._default_verifier
        self.history: list[HorizonGoal] = []

    def decompose(self, goal: str) -> list[str]:
        g = goal.strip()
        lower = g.lower()
        # Heuristic task templates (non-LLM)
        if "shortage" in lower or "material" in lower:
            return [
                "identify_site_and_window",
                "query_inventory_shortages",
                "draft_shortage_report",
                "verify_counts",
                "store_episode",
            ]
        if "invoice" in lower:
            return [
                "parse_project_filters",
                "query_invoices",
                "aggregate_totals",
                "verify_against_register",
                "respond",
            ]
        if "research" in lower or "search" in lower or "learn about" in lower:
            return [
                "rewrite_query",
                "local_knowledge_lookup",
                "optional_web_search",
                "synthesize_with_citations",
                "verify_nonempty_answer",
            ]
        if "compare" in lower:
            return ["extract_entities", "fetch_facts_a", "fetch_facts_b", "compose_comparison", "verify"]
        # Generic planning scaffold
        return [
            "clarify_goal",
            "gather_context",
            "propose_plan",
            "execute_next_action",
            "verify_progress",
            "reflect_and_adapt",
            "finalize",
        ]

    def run(
        self,
        goal: str,
        *,
        max_steps: Optional[int] = None,
        context: Optional[dict[str, Any]] = None,
        success_criteria: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        ctx = dict(context or {})
        intentions = self.decompose(goal)
        hg = HorizonGoal(
            goal=goal,
            max_steps=max_steps or min(8, max(3, len(intentions))),
            success_criteria=list(success_criteria or ["verified_final_step"]),
        )
        hg.status = "running"
        t0 = time.perf_counter()

        for i, intention in enumerate(intentions[: hg.max_steps]):
            result = self.actuator(intention, {"goal": goal, "context": ctx, "index": i})
            obs = str(result.get("observation") or result.get("text") or result.get("ok"))
            verified = bool(self.verifier(intention, result))
            step = HorizonStep(
                index=i,
                intention=intention,
                action=str(result.get("action") or intention),
                observation=obs[:500],
                verified=verified,
                notes=str(result.get("notes") or ""),
            )
            hg.steps.append(step)
            ctx[f"step_{i}"] = result
            if not verified and result.get("fatal"):
                hg.status = "failed"
                hg.reflections.append(f"Fatal at {intention}: {obs}")
                break
            # Simple reflect every 2 steps
            if (i + 1) % 2 == 0:
                n_ok = sum(1 for s in hg.steps if s.verified)
                hg.reflections.append(
                    f"After {i+1} steps: {n_ok}/{i+1} verified; continue toward '{goal}'."
                )
        else:
            # completed loop without break
            if hg.steps and all(s.verified for s in hg.steps[-2:]):
                hg.status = "succeeded"
            elif hg.steps and sum(1 for s in hg.steps if s.verified) >= max(1, len(hg.steps) // 2):
                hg.status = "succeeded"
                hg.reflections.append("Majority steps verified — accepting bounded success.")
            else:
                hg.status = "failed"
                hg.reflections.append("Insufficient verified steps.")

        # Persist episode
        ep = Episode(
            summary=f"long_horizon:{goal}",
            observations=[{"goal": goal}],
            actions=[{"op": s.intention, "verified": s.verified} for s in hg.steps],
            outcome="success" if hg.status == "succeeded" else hg.status,
        )
        self.memory.add_episode(ep)
        self.history.append(hg)

        return {
            "ok": hg.status == "succeeded",
            "goal": hg.to_dict(),
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            "episode_id": ep.episode_id,
            "human_level_agi": False,
            "llm_dependency": False,
            "note": "Bounded long-horizon loop — not open-ended AGI autonomy.",
        }

    def _default_actuator(self, intention: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Stub actuator: marks planned intentions as attempted
        return {
            "ok": True,
            "action": f"do:{intention}",
            "observation": f"Simulated progress on '{intention}' for goal '{payload.get('goal')}'.",
            "notes": "default_actuator",
        }

    def _default_verifier(self, intention: str, result: dict[str, Any]) -> bool:
        if result.get("ok") is False:
            return False
        obs = str(result.get("observation") or "")
        return bool(obs) and "fail" not in obs.lower()
