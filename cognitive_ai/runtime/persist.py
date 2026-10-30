"""Lightweight JSON persistence for Complete AI session + user memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class StateStore:
    def __init__(self, path: str | Path = "data/complete_ai/state.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"user_memory": {}, "session": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"user_memory": {}, "session": []}

    def save(self, *, user_memory: dict[str, str], session: list[dict[str, Any]]) -> None:
        payload = {
            "user_memory": user_memory,
            "session": session[-200:],
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def default_store(path: Optional[str] = None) -> StateStore:
    return StateStore(path or "data/complete_ai/state.json")
