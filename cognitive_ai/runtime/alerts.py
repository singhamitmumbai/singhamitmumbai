"""Proactive monitoring: overdue invoices, low stock, expiring contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

from ..memory.store import MemorySystem


@dataclass
class Alert:
    kind: str
    severity: str  # info|warn|critical
    entity_id: str
    message: str
    project_id: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "entity_id": self.entity_id,
            "message": self.message,
            "project_id": self.project_id,
            "meta": dict(self.meta),
        }


class ProactiveMonitor:
    """Scan registers and emit explicit, auditable alerts (non-LLM)."""

    def __init__(
        self,
        memory: MemorySystem,
        *,
        as_of: Optional[date] = None,
        low_stock_threshold: float = 50.0,
        contract_horizon_days: int = 90,
    ) -> None:
        self.memory = memory
        self.as_of = as_of or date.today()
        self.low_stock_threshold = low_stock_threshold
        self.contract_horizon_days = contract_horizon_days

    def scan(self, *, project_id: Optional[str] = None) -> list[Alert]:
        alerts: list[Alert] = []
        alerts.extend(self._overdue_invoices(project_id))
        alerts.extend(self._low_stock())
        alerts.extend(self._expiring_contracts(project_id))
        alerts.extend(self._pending_deliveries(project_id))
        return alerts

    def summary(self, *, project_id: Optional[str] = None) -> dict[str, Any]:
        alerts = self.scan(project_id=project_id)
        by_sev = {"critical": 0, "warn": 0, "info": 0}
        for a in alerts:
            by_sev[a.severity] = by_sev.get(a.severity, 0) + 1
        return {
            "as_of": self.as_of.isoformat(),
            "count": len(alerts),
            "by_severity": by_sev,
            "alerts": [a.to_dict() for a in alerts],
        }

    def _overdue_invoices(self, project_id: Optional[str]) -> list[Alert]:
        out: list[Alert] = []
        for inv in self.memory.get_records("invoice"):
            if project_id and inv.get("project_id") != project_id:
                continue
            if inv.get("payment_status") == "paid":
                continue
            due = _parse_date(inv.get("due_date"))
            if due is None or due >= self.as_of:
                continue
            days = (self.as_of - due).days
            sev = "critical" if days > 90 else "warn"
            out.append(
                Alert(
                    kind="overdue_invoice",
                    severity=sev,
                    entity_id=inv["id"],
                    project_id=inv.get("project_id"),
                    message=(
                        f"Invoice {inv['id']} unpaid and overdue by {days} day(s) "
                        f"(due {due.isoformat()}, ₹{inv.get('amount_inr')})."
                    ),
                    meta={"due_date": due.isoformat(), "days_overdue": days, "amount_inr": inv.get("amount_inr")},
                )
            )
        return out

    def _low_stock(self) -> list[Alert]:
        out: list[Alert] = []
        for mat in self.memory.get_records("material"):
            available = float(mat.get("available") or 0)
            reserved = float(mat.get("reserved") or 0)
            free = available - reserved
            if free > self.low_stock_threshold:
                continue
            sev = "critical" if free <= 0 else "warn"
            out.append(
                Alert(
                    kind="low_stock",
                    severity=sev,
                    entity_id=mat["id"],
                    message=(
                        f"Material {mat.get('name') or mat['id']} free stock {free:g} "
                        f"(available {available:g}, reserved {reserved:g}) "
                        f"at/below threshold {self.low_stock_threshold:g}."
                    ),
                    meta={"available": available, "reserved": reserved, "free": free},
                )
            )
        return out

    def _expiring_contracts(self, project_id: Optional[str]) -> list[Alert]:
        out: list[Alert] = []
        horizon = self.as_of + timedelta(days=self.contract_horizon_days)
        for ctr in self.memory.get_records("contract"):
            if project_id and ctr.get("project_id") != project_id:
                continue
            if ctr.get("status") not in (None, "active"):
                continue
            end = _parse_date(ctr.get("end_date"))
            if end is None:
                continue
            if end < self.as_of:
                out.append(
                    Alert(
                        kind="expired_contract",
                        severity="critical",
                        entity_id=ctr["id"],
                        project_id=ctr.get("project_id"),
                        message=f"Contract {ctr['id']} ({ctr.get('title')}) expired on {end.isoformat()}.",
                        meta={"end_date": end.isoformat()},
                    )
                )
            elif end <= horizon:
                days = (end - self.as_of).days
                out.append(
                    Alert(
                        kind="expiring_contract",
                        severity="warn",
                        entity_id=ctr["id"],
                        project_id=ctr.get("project_id"),
                        message=(
                            f"Contract {ctr['id']} ({ctr.get('title')}) expires in {days} day(s) "
                            f"({end.isoformat()})."
                        ),
                        meta={"end_date": end.isoformat(), "days_remaining": days},
                    )
                )
        return out

    def _pending_deliveries(self, project_id: Optional[str]) -> list[Alert]:
        out: list[Alert] = []
        for d in self.memory.get_records("delivery"):
            if project_id and d.get("project_id") != project_id:
                continue
            if d.get("receipt_status") == "missing":
                out.append(
                    Alert(
                        kind="missing_receipt",
                        severity="info",
                        entity_id=d["id"],
                        project_id=d.get("project_id"),
                        message=f"Delivery {d['id']} ({d.get('item')}) has unresolved receipt status.",
                        meta={"item": d.get("item")},
                    )
                )
                continue
            expected = d.get("expected_qty")
            received = d.get("received_qty")
            if expected is None or received is None:
                continue
            if float(received) < float(expected):
                out.append(
                    Alert(
                        kind="pending_delivery",
                        severity="warn",
                        entity_id=d["id"],
                        project_id=d.get("project_id"),
                        message=(
                            f"Delivery {d['id']} ({d.get('item')}) pending: "
                            f"received {received}/{expected}."
                        ),
                        meta={"expected_qty": expected, "received_qty": received},
                    )
                )
        return out


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
