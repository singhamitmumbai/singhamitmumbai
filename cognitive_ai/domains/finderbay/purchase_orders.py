"""Purchase orders + vendor register for Finderbay."""

from __future__ import annotations

from typing import Any, Optional

from ...core.ids import new_id
from ...core.records import Claim, EvidenceStatus, utc_now
from ...memory.store import MemorySystem
from .auth import AuthService


class PurchaseOrderService:
    def __init__(self, memory: MemorySystem, auth: AuthService) -> None:
        self.memory = memory
        self.auth = auth

    def list_vendors(self) -> list[dict]:
        return self.memory.get_records("vendor")

    def list_pos(
        self,
        *,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        vendor_id: Optional[str] = None,
    ) -> list[dict]:
        rows = self.memory.get_records("purchase_order")
        if project_id:
            rows = [r for r in rows if r.get("project_id") == project_id]
        if status:
            rows = [r for r in rows if r.get("status") == status]
        if vendor_id:
            rows = [r for r in rows if r.get("vendor_id") == vendor_id]
        return rows

    def create_po(
        self,
        *,
        token: Optional[str],
        project_id: str,
        vendor_id: str,
        item: str,
        qty: float,
        unit_price_inr: float,
        notes: str = "",
    ) -> dict[str, Any]:
        self.auth.require(token, "po_write")
        vendors = self.memory.get_records("vendor", id=vendor_id)
        if not vendors:
            raise ValueError(f"vendor_not_found:{vendor_id}")
        projects = self.memory.get_records("project", id=project_id)
        if not projects:
            raise ValueError(f"project_not_found:{project_id}")
        pid = new_id("po")
        total = float(qty) * float(unit_price_inr)
        row = {
            "id": pid,
            "project_id": project_id,
            "vendor_id": vendor_id,
            "vendor_name": vendors[0].get("name"),
            "item": item,
            "qty": float(qty),
            "unit_price_inr": float(unit_price_inr),
            "total_inr": total,
            "status": "open",
            "notes": notes,
            "created_at": utc_now(),
        }
        self.memory.upsert_record("purchase_order", pid, row)
        self.memory.remember_claim(
            Claim(
                subject=pid,
                relationship="status",
                value="open",
                source="purchase_order_service",
                evidence_status=EvidenceStatus.OBSERVED,
                project_id=project_id,
            )
        )
        return {"purchase_order": row}

    def summarize(self, project_id: Optional[str] = None) -> dict[str, Any]:
        rows = self.list_pos(project_id=project_id)
        open_rows = [r for r in rows if r.get("status") == "open"]
        return {
            "count": len(rows),
            "open": len(open_rows),
            "open_value_inr": sum(float(r.get("total_inr") or 0) for r in open_rows),
            "vendors": len(self.list_vendors()),
        }
