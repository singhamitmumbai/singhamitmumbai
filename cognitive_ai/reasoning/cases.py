"""Case-based reasoning: reuse earlier verified episodes (blueprint §8)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..memory.store import MemorySystem, Episode


@dataclass
class CaseMatch:
    episode_id: str
    score: float
    summary: str
    outcome: str
    reused_steps: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "score": self.score,
            "summary": self.summary,
            "outcome": self.outcome,
            "reused_steps": list(self.reused_steps),
            "message": self.message,
        }


class CaseBasedReasoner:
    """Retrieve similar past episodes and suggest reusable action patterns."""

    def retrieve(
        self,
        memory: MemorySystem,
        query: str,
        *,
        operation: Optional[str] = None,
        top_k: int = 3,
    ) -> list[CaseMatch]:
        episodes = list(getattr(memory, "episodes", []) or [])
        if not episodes:
            return []
        q_tokens = set(_tokens(query))
        if operation:
            q_tokens.add(operation.lower())
        scored: list[CaseMatch] = []
        for ep in episodes:
            if not isinstance(ep, Episode):
                continue
            text = (ep.summary or "") + " " + " ".join(str(a) for a in (ep.actions or [])[:5])
            e_tokens = set(_tokens(text))
            if not e_tokens:
                continue
            overlap = len(q_tokens & e_tokens)
            if overlap == 0:
                continue
            score = overlap / max(1, len(q_tokens | e_tokens))
            # Prefer satisfied outcomes
            if (ep.outcome or "").lower() in ("satisfied", "success", "confirmed"):
                score += 0.15
            steps = []
            for a in ep.actions or []:
                if isinstance(a, dict):
                    op = a.get("operator") or a.get("name") or (a.get("step") or {}).get("operator")
                    if op:
                        steps.append(str(op))
            scored.append(
                CaseMatch(
                    episode_id=ep.episode_id,
                    score=score,
                    summary=ep.summary,
                    outcome=ep.outcome or "",
                    reused_steps=steps[:8],
                    message=f"Similar past case ({ep.outcome}): {ep.summary[:120]}",
                )
            )
        scored.sort(key=lambda c: -c.score)
        return scored[:top_k]

    def suggest_plan_hint(self, matches: list[CaseMatch]) -> Optional[dict[str, Any]]:
        if not matches:
            return None
        best = matches[0]
        if best.score < 0.12:
            return None
        return {
            "from_case": best.episode_id,
            "score": best.score,
            "suggested_operators": list(best.reused_steps),
            "message": best.message,
        }


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) > 2]
