"""Durable checkpoints, restart recovery and OUTCOME_UNKNOWN reconciliation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..core.records import ActionResult, ActionStatus, utc_now
from .verifier import Verifier


class RecoveryManager:
    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else Path("data/finderbay/checkpoints.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoints: list[dict[str, Any]] = []
        self.load()

    def checkpoint(self, plan_id: str, step_id: str, result: ActionResult) -> None:
        self.checkpoints.append(
            {
                "plan_id": plan_id,
                "step_id": step_id,
                "result": result.to_dict(),
                "saved_at": utc_now(),
            }
        )
        self.save()

    def save(self) -> Path:
        self.path.write_text(
            json.dumps({"checkpoints": self.checkpoints[-500:]}, indent=2, default=str),
            encoding="utf-8",
        )
        return self.path

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.checkpoints = list(data.get("checkpoints") or [])

    def pending_unknowns(self) -> list[dict[str, Any]]:
        out = []
        for cp in self.checkpoints:
            status = (cp.get("result") or {}).get("status")
            if status == ActionStatus.OUTCOME_UNKNOWN.value:
                out.append(cp)
        return out

    def reconcile_all(self, verifier: Verifier) -> list[ActionResult]:
        resolved: list[ActionResult] = []
        for cp in self.pending_unknowns():
            rd = cp.get("result") or {}
            result = ActionResult(
                attempt_id=rd.get("attempt_id", ""),
                operator=rd.get("operator", ""),
                status=ActionStatus.OUTCOME_UNKNOWN,
                args=rd.get("args") or {},
                external_receipt=rd.get("external_receipt"),
                observed_effects=rd.get("observed_effects") or [],
                error=rd.get("error"),
                result_id=rd.get("result_id", ""),
            )
            updated = verifier.reconcile_unknown(result)
            if updated.status != ActionStatus.OUTCOME_UNKNOWN:
                rd["status"] = updated.status.value
                rd["external_receipt"] = updated.external_receipt
                cp["result"] = rd
                resolved.append(updated)
        if resolved:
            self.save()
        return resolved
