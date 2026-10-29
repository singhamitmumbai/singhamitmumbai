"""Persistent Goal / Objective engine for Mother AI."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now

GOAL_STATES = (
    "PROPOSED",
    "ACTIVE",
    "BLOCKED",
    "WAITING",
    "COMPLETED",
    "FAILED",
    "ABANDONED",
    "SUPERSEDED",
)


@dataclass
class Goal:
    description: str
    origin: str = "user"
    priority: float = 0.5
    status: str = "PROPOSED"
    success_conditions: list[str] = field(default_factory=list)
    failure_conditions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    parent_goal: Optional[str] = None
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    deadline: Optional[str] = None
    resources: list[str] = field(default_factory=list)
    reversible: bool = True
    goal_id: str = field(default_factory=lambda: new_id("goal"))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    result_summary: str = ""
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Goal":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class GoalManager:
    """Durable goals that survive restart."""

    def __init__(self, path: Path | str = "data/complete_ai/mother/goals.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.goals: dict[str, Goal] = {}
        self._load()

    def create(
        self,
        description: str,
        *,
        origin: str = "user",
        priority: float = 0.5,
        success_conditions: Optional[list[str]] = None,
        parent_goal: Optional[str] = None,
        activate: bool = True,
    ) -> Goal:
        g = Goal(
            description=description,
            origin=origin,
            priority=priority,
            status="ACTIVE" if activate else "PROPOSED",
            success_conditions=list(success_conditions or []),
            parent_goal=parent_goal,
        )
        with self._lock:
            self.goals[g.goal_id] = g
            self._save()
        return g

    def update(self, goal_id: str, *, status: Optional[str] = None, **fields: Any) -> Optional[Goal]:
        with self._lock:
            g = self.goals.get(goal_id)
            if not g:
                return None
            if status:
                if status not in GOAL_STATES:
                    raise ValueError(f"invalid_goal_status:{status}")
                g.status = status
            for k, v in fields.items():
                if hasattr(g, k) and k not in ("goal_id", "created_at"):
                    setattr(g, k, v)
            g.updated_at = utc_now()
            self._save()
            return g

    def complete(self, goal_id: str, summary: str = "") -> Optional[Goal]:
        return self.update(goal_id, status="COMPLETED", result_summary=summary, confidence=0.85)

    def fail(self, goal_id: str, summary: str = "") -> Optional[Goal]:
        return self.update(goal_id, status="FAILED", result_summary=summary, confidence=0.4)

    def active(self) -> list[Goal]:
        return [g for g in self.goals.values() if g.status == "ACTIVE"]

    def list_goals(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = sorted(self.goals.values(), key=lambda g: g.updated_at, reverse=True)
        return [g.to_dict() for g in rows[:limit]]

    def summary(self) -> dict[str, Any]:
        by: dict[str, int] = {}
        for g in self.goals.values():
            by[g.status] = by.get(g.status, 0) + 1
        return {"total": len(self.goals), "by_status": by, "active": len(self.active()), "human_level_agi": False}

    def _save(self) -> None:
        payload = {"goals": [g.to_dict() for g in self.goals.values()], "human_level_agi": False}
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for row in data.get("goals") or []:
                g = Goal.from_dict(row)
                self.goals[g.goal_id] = g
        except Exception:  # noqa: BLE001
            return
