"""Persistent learning journal — append-only JSONL for 24/7 learners."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from ..core.ids import new_id
from ..core.records import utc_now


class LearningJournal:
    """Thread-safe durable log of observations / ticks / promotions."""

    def __init__(self, path: Path | str = "data/complete_ai/learning_journal.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, kind: str, payload: dict[str, Any], *, source: str = "system") -> dict[str, Any]:
        row = {
            "id": new_id("lj"),
            "kind": kind,
            "source": source,
            "payload": payload,
            "ts": utc_now(),
            "human_level_agi": False,
        }
        line = json.dumps(row, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return row

    def iter_recent(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return []
            lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def count(self) -> int:
        with self._lock:
            if not self.path.exists():
                return 0
            return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")
