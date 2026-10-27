"""CSV / structured report export for Finderbay registers (non-LLM)."""

from __future__ import annotations

import csv
import io
from typing import Any, Optional

from ..memory.store import MemorySystem
from ..runtime.alerts import ProactiveMonitor


REPORT_KINDS = ("invoices", "payments", "contracts", "purchase_orders", "alerts", "materials", "deliveries")


class ReportExporter:
    def __init__(self, memory: MemorySystem, monitor: Optional[ProactiveMonitor] = None) -> None:
        self.memory = memory
        self.monitor = monitor or ProactiveMonitor(memory)

    def export(
        self,
        kind: str,
        *,
        project_id: Optional[str] = None,
        fmt: str = "csv",
    ) -> dict[str, Any]:
        kind = kind.replace(" ", "_").lower()
        if kind in ("pos", "po", "purchase_order"):
            kind = "purchase_orders"
        if kind not in REPORT_KINDS:
            raise ValueError(f"unsupported_report:{kind}")

        if kind == "alerts":
            rows = [a.to_dict() for a in self.monitor.scan(project_id=project_id)]
        else:
            record_kind = {
                "invoices": "invoice",
                "payments": "payment",
                "contracts": "contract",
                "purchase_orders": "purchase_order",
                "materials": "material",
                "deliveries": "delivery",
            }[kind]
            rows = self.memory.get_records(record_kind)
            if project_id and record_kind != "material":
                rows = [r for r in rows if r.get("project_id") == project_id]

        if fmt == "json":
            return {"kind": kind, "count": len(rows), "rows": rows, "format": "json"}

        columns = _columns_for(rows)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in columns})
        return {
            "kind": kind,
            "count": len(rows),
            "columns": columns,
            "format": "csv",
            "csv": buf.getvalue(),
        }


def _columns_for(rows: list[dict]) -> list[str]:
    if not rows:
        return ["id"]
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen and k != "tables":
                seen.add(k)
                keys.append(k)
    return keys


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return str(value)
    return str(value)
