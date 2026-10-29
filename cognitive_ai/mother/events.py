"""Typed cognitive event bus for Mother AI observability."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Optional

from ..core.ids import new_id
from ..core.records import utc_now


@dataclass
class CognitiveEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: new_id("evt"))
    ts: str = field(default_factory=utc_now)
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    """Bounded in-memory event log + optional subscribers."""

    def __init__(self, *, capacity: int = 500) -> None:
        self.capacity = capacity
        self._events: Deque[CognitiveEvent] = deque(maxlen=capacity)
        self._subs: list[Callable[[CognitiveEvent], None]] = []
        self._lock = threading.Lock()

    def emit(self, kind: str, **payload: Any) -> CognitiveEvent:
        ev = CognitiveEvent(kind=kind, payload=dict(payload))
        with self._lock:
            self._events.append(ev)
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(ev)
            except Exception:  # noqa: BLE001
                pass
        return ev

    def subscribe(self, fn: Callable[[CognitiveEvent], None]) -> None:
        self._subs.append(fn)

    def recent(self, limit: int = 50, *, kind: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._events)
        if kind:
            rows = [e for e in rows if e.kind == kind]
        return [e.to_dict() for e in rows[-limit:]]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            kinds: dict[str, int] = {}
            for e in self._events:
                kinds[e.kind] = kinds.get(e.kind, 0) + 1
            return {"events": len(self._events), "by_kind": kinds, "human_level_agi": False}
