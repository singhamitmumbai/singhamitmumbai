"""Append-only multi-user audit trail for Finderbay sensitive actions."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from ...core.ids import new_id
from ...core.records import utc_now


class AuditLog:
    """Durable audit journal — login, payments, contracts, POs, exports."""

    version = "audit_v1"

    def __init__(self, path: Path | str = "data/complete_ai/finderbay_audit.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        self._memory: list[dict[str, Any]] = []

    def record(
        self,
        action: str,
        *,
        user_id: Optional[str] = None,
        roles: Optional[list[str]] = None,
        resource: str = "",
        detail: Optional[dict[str, Any]] = None,
        ok: bool = True,
        source: str = "finderbay",
    ) -> dict[str, Any]:
        row = {
            "id": new_id("aud"),
            "action": action,
            "user_id": user_id,
            "roles": list(roles or []),
            "resource": resource,
            "detail": dict(detail or {}),
            "ok": bool(ok),
            "source": source,
            "ts": utc_now(),
            "human_level_agi": False,
        }
        line = json.dumps(row, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._memory.append(row)
            if len(self._memory) > 2000:
                self._memory = self._memory[-2000:]
        return row

    def recent(
        self,
        *,
        limit: int = 50,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self._memory:
                rows = list(self._memory)
            elif self.path.exists():
                rows = []
                for line in self.path.read_text(encoding="utf-8").splitlines()[-500:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            else:
                rows = []
        if user_id:
            rows = [r for r in rows if r.get("user_id") == user_id]
        if action:
            al = action.lower()
            rows = [r for r in rows if al in str(r.get("action") or "").lower()]
        return rows[-limit:]

    def summary(self) -> dict[str, Any]:
        rows = self.recent(limit=500)
        by_action: dict[str, int] = {}
        for r in rows:
            a = str(r.get("action") or "unknown")
            by_action[a] = by_action.get(a, 0) + 1
        return {
            "version": self.version,
            "event_count": len(rows),
            "by_action": by_action,
            "path": str(self.path),
            "human_level_agi": False,
        }
