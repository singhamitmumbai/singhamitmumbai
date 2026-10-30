"""General learning controller — curriculum, replay, multi-source updates.

Learns facts, skills, and preferences from verified experience only.
Never promotes unverified model self-labels. human_level_agi=false.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import Claim, EvidenceStatus, LearningExample, utc_now
from ..memory.store import Episode, MemorySystem
from .continual_skills import GrowingSkillLibrary


@dataclass
class LearningEvent:
    kind: str  # fact|skill|preference|correction
    payload: dict[str, Any]
    verified: bool = True
    source: str = "user"
    event_id: str = field(default_factory=lambda: new_id("le"))
    ts: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneralLearner:
    """Unified learning API sitting above skill library + memory claims."""

    def __init__(
        self,
        memory: Optional[MemorySystem] = None,
        skills: Optional[GrowingSkillLibrary] = None,
    ) -> None:
        self.memory = memory or MemorySystem()
        self.skills = skills or GrowingSkillLibrary(self.memory)
        self.events: list[LearningEvent] = []
        self.replay_buffer: list[dict[str, Any]] = []
        self.curriculum_level: int = 1

    def observe(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        verified: bool = True,
        source: str = "user",
    ) -> dict[str, Any]:
        if not verified and source in ("model", "self", "unverified"):
            return {
                "ok": False,
                "error": "rejected_unverified_self_label",
                "human_level_agi": False,
            }
        ev = LearningEvent(kind=kind, payload=payload, verified=verified, source=source)
        self.events.append(ev)
        self.replay_buffer.append(ev.to_dict())
        if len(self.replay_buffer) > 500:
            self.replay_buffer = self.replay_buffer[-500:]

        if kind == "fact" and verified:
            claim = Claim(
                subject=str(payload.get("subject") or "entity"),
                relationship=str(payload.get("relationship") or "is_a"),
                value=payload.get("value") or payload.get("object") or "",
                source=source,
                evidence_status=EvidenceStatus.ASSERTED,
            )
            self.memory.remember_claim(claim)
            return {"ok": True, "learned": "fact", "claim_id": claim.claim_id, "human_level_agi": False}

        if kind == "skill" and verified:
            steps = list(payload.get("steps") or [])
            name = str(payload.get("name") or f"skill_{hashlib.sha1(json.dumps(steps).encode()).hexdigest()[:8]}")
            ep = Episode(
                summary=str(payload.get("summary") or name),
                actions=[{"op": s} for s in steps],
                outcome="success",
            )
            card = self.skills.learn_from_episode(ep, domain=str(payload.get("domain") or "general"))
            return {
                "ok": True,
                "learned": "skill",
                "skill": None if card is None else card.to_dict(),
                "human_level_agi": False,
            }

        if kind == "preference" and verified:
            key = str(payload.get("key") or "pref")
            self.memory.set_preference(key, payload.get("value"))
            return {"ok": True, "learned": "preference", "key": key, "human_level_agi": False}

        if kind == "correction" and verified:
            # Store correction as preference + negative skill outcome if named
            self.memory.set_preference(
                f"correction:{payload.get('topic', 'general')}",
                payload.get("correct_value") or payload.get("text"),
            )
            skill = payload.get("skill")
            if skill:
                self.skills.record_outcome(str(skill), success=False)
            return {"ok": True, "learned": "correction", "human_level_agi": False}

        return {"ok": bool(verified), "learned": kind, "human_level_agi": False}

    def learn_from_examples(self, name: str, positives: list[LearningExample], **kwargs) -> dict[str, Any]:
        return self.skills.learn_from_examples(name, positives, **kwargs)

    def replay(self, *, n: int = 20) -> dict[str, Any]:
        """Experience replay — re-assert recent verified events (idempotent)."""
        batch = list(reversed(self.replay_buffer))[:n]
        reinforced = 0
        for item in batch:
            if not item.get("verified"):
                continue
            kind = item.get("kind")
            payload = item.get("payload") or {}
            if kind == "fact":
                self.observe(kind="fact", payload=payload, verified=True, source="replay")
                reinforced += 1
            elif kind == "preference":
                self.observe(kind="preference", payload=payload, verified=True, source="replay")
                reinforced += 1
        return {
            "replayed": len(batch),
            "reinforced": reinforced,
            "curriculum_level": self.curriculum_level,
            "human_level_agi": False,
        }

    def advance_curriculum(self, *, success_rate: float) -> int:
        if success_rate >= 0.8 and self.curriculum_level < 5:
            self.curriculum_level += 1
        elif success_rate < 0.4 and self.curriculum_level > 1:
            self.curriculum_level -= 1
        return self.curriculum_level

    def summary(self) -> dict[str, Any]:
        return {
            "events": len(self.events),
            "replay_buffer": len(self.replay_buffer),
            "curriculum_level": self.curriculum_level,
            "skills": self.skills.summary(),
            "human_level_agi": False,
            "note": "General learner is verified-experience only — not open-ended AGI learning.",
        }
