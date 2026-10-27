"""Lifelong memory — consolidate episodic → semantic/procedural/preferences."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.records import Claim, EvidenceStatus, utc_now
from ..memory.store import Episode, MemorySystem, Procedure


@dataclass
class ConsolidationReport:
    episodes_seen: int = 0
    claims_added: int = 0
    procedures_reinforced: int = 0
    preferences_updated: int = 0
    summaries: list[str] = field(default_factory=list)
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LifelongMemory:
    """Durable multi-store memory with consolidation passes."""

    def __init__(self, memory: Optional[MemorySystem] = None) -> None:
        self.memory = memory or MemorySystem()
        self.consolidation_runs: list[dict[str, Any]] = []
        self._cursor = 0  # episodes already consolidated

    def remember_interaction(
        self,
        summary: str,
        *,
        observations: Optional[list[dict]] = None,
        actions: Optional[list[dict]] = None,
        outcome: str = "",
        preferences: Optional[dict[str, Any]] = None,
    ) -> Episode:
        ep = Episode(
            summary=summary,
            observations=list(observations or []),
            actions=list(actions or []),
            outcome=outcome,
        )
        self.memory.add_episode(ep)
        if preferences:
            for k, v in preferences.items():
                self.memory.set_preference(k, v)
        return ep

    def importance(self, episode: Episode) -> float:
        """Heuristic importance for retention / dream replay."""
        score = 0.3
        if str(episode.outcome).lower() in ("ok", "success", "succeeded", "verified", "true"):
            score += 0.3
        score += min(0.3, 0.05 * len(episode.actions))
        if any(
            k in episode.summary.lower()
            for k in ("fail", "error", "correct", "critical", "align")
        ):
            score += 0.2
        return min(1.0, score)

    def consolidate(self, *, limit: int = 50) -> ConsolidationReport:
        """Promote recurring episode patterns into claims/procedures."""
        report = ConsolidationReport()
        eps = self.memory.episodes[self._cursor : self._cursor + limit]
        report.episodes_seen = len(eps)
        action_counter: Counter[str] = Counter()
        outcome_ok = 0
        for ep in eps:
            imp = self.importance(ep)
            if str(ep.outcome).lower() in ("ok", "success", "succeeded", "verified", "true"):
                outcome_ok += 1
            for a in ep.actions:
                if isinstance(a, dict):
                    action_counter[str(a.get("op") or a.get("step") or "act")] += 1
                else:
                    action_counter[str(a)] += 1
            # Semantic: store summary claim (importance in notes)
            claim = Claim(
                subject="episode",
                relationship="summary",
                value=ep.summary[:200],
                source="lifelong_consolidation",
                evidence_status=EvidenceStatus.ASSERTED,
                notes=f"episode:{ep.episode_id}|importance={imp:.2f}",
            )
            self.memory.remember_claim(claim)
            report.claims_added += 1
            if imp >= 0.7:
                self.memory.set_preference(f"salient:{ep.episode_id}", ep.summary[:120])
                report.preferences_updated += 1

        # Reinforce top action sequences as micro-procedures
        for op, cnt in action_counter.most_common(5):
            if cnt < 2:
                continue
            name = f"lifelong_{op}"
            existing = self.memory.procedures.get(name)
            if existing:
                existing.success_count += cnt
                report.procedures_reinforced += 1
            else:
                self.memory.add_procedure(
                    Procedure(
                        name=name,
                        steps=[op, "verify"],
                        applicability=["lifelong", "auto"],
                        version="lifelong_v1",
                        success_count=cnt,
                    )
                )
                report.procedures_reinforced += 1

        if eps:
            rate = outcome_ok / max(1, len(eps))
            self.memory.set_preference("lifelong_success_rate", round(rate, 3))
            self.memory.set_preference("lifelong_last_consolidation", utc_now())
            report.preferences_updated = 2
            report.summaries.append(
                f"Consolidated {len(eps)} episodes; top actions={action_counter.most_common(3)}; success_rate={rate:.0%}."
            )

        self._cursor += len(eps)
        payload = report.to_dict()
        self.consolidation_runs.append(payload)
        return report

    def recall(self, query: str, *, limit: int = 10) -> dict[str, Any]:
        q = (query or "").lower()
        episodes = [
            e.__dict__
            for e in reversed(self.memory.episodes)
            if q in e.summary.lower() or any(q in str(a).lower() for a in e.actions)
        ][:limit]
        claims = [
            c.to_dict()
            for c in self.memory.current_claims()
            if q in str(c.value).lower() or q in c.subject.lower()
        ][:limit]
        procs = [
            p.__dict__
            for name, p in self.memory.procedures.items()
            if q in name.lower() or any(q in s.lower() for s in p.steps)
        ][:limit]
        return {
            "query": query,
            "episodes": episodes,
            "claims": claims,
            "procedures": procs,
            "preferences": dict(self.memory.preferences),
            "human_level_agi": False,
        }

    def dream(self, *, n: int = 5) -> dict[str, Any]:
        """Spaced replay of salient episodes (offline consolidation aid)."""
        ranked = sorted(self.memory.episodes, key=self.importance, reverse=True)[:n]
        replayed = []
        for ep in ranked:
            replayed.append(
                {
                    "episode_id": ep.episode_id,
                    "summary": ep.summary,
                    "importance": self.importance(ep),
                }
            )
            # Soft reinforce top action
            if ep.actions:
                a0 = ep.actions[0]
                op = a0.get("op") if isinstance(a0, dict) else str(a0)
                name = f"lifelong_{op}"
                proc = self.memory.procedures.get(name)
                if proc:
                    proc.success_count += 1
        return {"dreamed": len(replayed), "items": replayed, "human_level_agi": False}

    def summary(self) -> dict[str, Any]:
        ms = self.memory.summary()
        return {
            **ms,
            "consolidations": len(self.consolidation_runs),
            "cursor": self._cursor,
            "human_level_agi": False,
            "note": "Lifelong consolidation is heuristic — not human autobiographical memory.",
        }
