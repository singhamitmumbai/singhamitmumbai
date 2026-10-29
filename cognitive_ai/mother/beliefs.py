"""Belief store — distinguish fact / hypothesis / observation / inference."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now

BELIEF_STATUS = ("active", "superseded", "retracted", "uncertain")


@dataclass
class Belief:
    proposition: str
    confidence: float = 0.5
    evidence_refs: list[str] = field(default_factory=list)
    source: str = "system"
    kind: str = "observation"  # fact|observation|hypothesis|prediction|assumption|user|inference
    scope: str = "session"
    status: str = "active"
    supersedes: Optional[str] = None
    belief_id: str = field(default_factory=lambda: new_id("bel"))
    timestamp: str = field(default_factory=utc_now)
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BeliefStore:
    def __init__(self, path: Path | str = "data/complete_ai/mother/beliefs.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.beliefs: dict[str, Belief] = {}
        self._load()

    def assert_belief(
        self,
        proposition: str,
        *,
        confidence: float = 0.6,
        source: str = "system",
        kind: str = "observation",
        evidence_refs: Optional[list[str]] = None,
        supersedes: Optional[str] = None,
    ) -> Belief:
        with self._lock:
            if supersedes and supersedes in self.beliefs:
                self.beliefs[supersedes].status = "superseded"
            # Soft contradiction: same proposition with opposite confidence markers
            for b in self.beliefs.values():
                if b.status == "active" and b.proposition.lower() == proposition.lower() and abs(b.confidence - confidence) > 0.5:
                    b.status = "uncertain"
            bel = Belief(
                proposition=proposition,
                confidence=max(0.0, min(1.0, confidence)),
                source=source,
                kind=kind,
                evidence_refs=list(evidence_refs or []),
                supersedes=supersedes,
            )
            self.beliefs[bel.belief_id] = bel
            self._save()
            return bel

    def active(self, *, limit: int = 40) -> list[dict[str, Any]]:
        rows = [b for b in self.beliefs.values() if b.status == "active"]
        rows.sort(key=lambda b: -b.confidence)
        return [b.to_dict() for b in rows[:limit]]

    def summary(self) -> dict[str, Any]:
        by: dict[str, int] = {}
        for b in self.beliefs.values():
            by[b.kind] = by.get(b.kind, 0) + 1
        return {"total": len(self.beliefs), "by_kind": by, "human_level_agi": False}

    def _save(self) -> None:
        payload = {"beliefs": [b.to_dict() for b in self.beliefs.values()], "human_level_agi": False}
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for row in data.get("beliefs") or []:
                b = Belief(**{k: v for k, v in row.items() if k in Belief.__dataclass_fields__})
                self.beliefs[b.belief_id] = b
        except Exception:  # noqa: BLE001
            return
