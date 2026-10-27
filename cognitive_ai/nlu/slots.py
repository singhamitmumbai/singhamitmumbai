"""Entity / slot extraction and compositional operators (negation, comparison)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .normalize import extract_dates_relative, extract_money


@dataclass
class Condition:
    field: str
    operator: str  # eq|ne|gt|gte|lt|lte|in|not_in
    value: Any
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "operator": "ne" if self.negated and self.operator == "eq" else self.operator,
            "value": self.value,
        }


@dataclass
class SlotBundle:
    conditions: list[Condition] = field(default_factory=list)
    project_id: Optional[str] = None
    amounts: list[dict] = field(default_factory=list)
    times: list[dict] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    prohibitions: list[str] = field(default_factory=list)
    output_hint: Optional[str] = None
    unknown_policies: dict[str, str] = field(default_factory=dict)


COMPARISON_PATTERNS = [
    (r"(?:above|over|greater than|more than|>\s*)\s*(.+)", "gt"),
    (r"(?:below|under|less than|<\s*)\s*(.+)", "lt"),
    (r"(?:at least|>=\s*)\s*(.+)", "gte"),
    (r"(?:at most|<=\s*)\s*(.+)", "lte"),
]


def _parse_amount_token(token: str) -> Optional[float]:
    money = extract_money(token)
    if money:
        return money[0]["amount_inr"]
    cleaned = token.replace(",", "").strip()
    m = re.match(r"^([\d.]+)", cleaned)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def extract_slots(text: str, project_id: Optional[str] = None) -> SlotBundle:
    lower = text.lower()
    bundle = SlotBundle(project_id=project_id)
    bundle.amounts = extract_money(text)
    bundle.times = extract_dates_relative(text)

    # Payment / dispute status
    if re.search(r"\bunpaid\b|\bnot\s+paid\b|\bbaaki\b|\babhi\s+pending\s+payment\b", lower):
        bundle.conditions.append(Condition("payment_status", "eq", "unpaid"))
    if re.search(r"\bpaid\b", lower) and "unpaid" not in lower and "not paid" not in lower:
        bundle.conditions.append(Condition("payment_status", "eq", "paid"))
    if re.search(r"\boverdue\b", lower):
        bundle.conditions.append(Condition("overdue", "eq", True))
    if re.search(r"\bpending\b", lower):
        bundle.conditions.append(Condition("delivery_status", "eq", "pending"))

    # Conjunction / alternatives markers retained as notes on conditions
    if re.search(r"\band\b|\bor\b", lower):
        bundle.unknown_policies["boolean_connective"] = "parsed_left_to_right"

    # Exclusions: "excluding disputed" / "except disputed" / "all except"
    excl = re.search(
        r"(?:excluding|except|without|excluding all)\s+([a-zA-Z0-9_ -]+?)(?:\s+ones|\s+invoices|$|,|\.)",
        lower,
    )
    if excl:
        phrase = excl.group(1).strip()
        bundle.exclusions.append(phrase)
        if "disputed" in phrase:
            bundle.conditions.append(Condition("dispute_status", "eq", "not_disputed"))
            bundle.unknown_policies["dispute_status"] = "report_separately"

    # Amount comparisons tied to money mentions
    for pat, op in COMPARISON_PATTERNS:
        for m in re.finditer(pat, lower):
            amt = _parse_amount_token(m.group(1))
            if amt is not None:
                bundle.conditions.append(Condition("amount_inr", op, amt))

    # Reserve quantity: "reserve 15 bags of cement" / "reserve 10 cement"
    m_qty = re.search(
        r"\breserve\s+(\d+(?:\.\d+)?)\s*(?:bags?\s+of\s+)?([a-zA-Z][a-zA-Z0-9_ -]*)",
        lower,
    )
    if m_qty:
        bundle.conditions.append(Condition("reserve_qty", "eq", float(m_qty.group(1))))
        bundle.conditions.append(Condition("material_name", "eq", m_qty.group(2).strip()))

    # Task title: create/add task "..."
    m_task = re.search(
        r"(?:create|add|naya)\s+task(?:\s+for\s+project\s+\w+)?\s*[:\-]?\s*(.+)$",
        text,
        flags=re.I,
    )
    if m_task:
        title = m_task.group(1).strip().rstrip(".")
        # strip trailing project clause if duplicated
        title = re.sub(r"\s+for\s+project\s+\w+$", "", title, flags=re.I).strip()
        if title:
            bundle.conditions.append(Condition("task_title", "eq", title))

    # Document field focus
    for field_name in ("amount_due", "subtotal", "tax", "invoice_number", "vendor"):
        if field_name.replace("_", " ") in lower or field_name in lower:
            bundle.conditions.append(Condition("document_field", "eq", field_name))

    # Payment / contract slots
    m_inv = re.search(r"\binvoice\s+(\d+)\b", lower)
    if m_inv:
        bundle.conditions.append(Condition("invoice_id", "eq", f"inv_{m_inv.group(1)}"))
    m_pay = re.search(
        r"(?:record|post|receive)\s+payment(?:\s+of)?\s*(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m_pay:
        try:
            bundle.conditions.append(
                Condition("payment_amount_inr", "eq", float(m_pay.group(1).replace(",", "")))
            )
        except ValueError:
            pass
    m_user = re.search(r"login\s+(\w+)\s+(?:with\s+)?(?:password\s+)?(\S+)", lower)
    if m_user:
        bundle.conditions.append(Condition("username", "eq", m_user.group(1)))
        bundle.conditions.append(Condition("password", "eq", m_user.group(2)))

    # Purchase order / vendor slots
    m_vendor = re.search(r"\b(?:vendor|supplier)\s+(vendor_[a-z0-9_]+|[a-z][a-z0-9_]*)\b", lower)
    if m_vendor:
        raw = m_vendor.group(1)
        vid = raw if raw.startswith("vendor_") else f"vendor_{raw}"
        bundle.conditions.append(Condition("vendor_id", "eq", vid))
    m_po = re.search(
        r"(?:create|raise|new)\s+(?:purchase\s+order|po)\s+(?:for\s+)?(.+?)\s+"
        r"(?:qty|quantity)\s+(\d+(?:\.\d+)?)\s+(?:@|at|rate|price)\s*(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m_po:
        bundle.conditions.append(Condition("item", "eq", m_po.group(1).strip()))
        bundle.conditions.append(Condition("qty", "eq", float(m_po.group(2))))
        bundle.conditions.append(
            Condition("unit_price_inr", "eq", float(m_po.group(3).replace(",", "")))
        )
    # Simpler: create po cement qty 10 @ 500 vendor buildco
    m_po2 = re.search(
        r"(?:create|raise)\s+(?:po|purchase\s+order)\s+(\w+)\s+qty\s+(\d+(?:\.\d+)?)\s+"
        r"(?:@|at)\s*([\d,]+(?:\.\d+)?)",
        lower,
    )
    if m_po2 and not m_po:
        bundle.conditions.append(Condition("item", "eq", m_po2.group(1)))
        bundle.conditions.append(Condition("qty", "eq", float(m_po2.group(2))))
        bundle.conditions.append(
            Condition("unit_price_inr", "eq", float(m_po2.group(3).replace(",", "")))
        )

    # Report kind
    for kind in ("invoices", "payments", "alerts", "purchase_orders", "contracts", "materials", "deliveries"):
        if re.search(rf"\b{kind}\b", lower) and re.search(r"\b(?:export|report|csv)\b", lower):
            bundle.conditions.append(Condition("report_kind", "eq", kind))
            break
    if re.search(r"\b(?:export|report).*\bpos?\b|\bpo\s+report\b", lower):
        bundle.conditions.append(Condition("report_kind", "eq", "purchase_orders"))

    # Prohibitions
    if re.search(r"do not send|don't send|mat bhejo|do not email", lower):
        bundle.prohibitions.append("send")
    if re.search(r"do not (?:delete|remove)", lower):
        bundle.prohibitions.append("delete")

    # Output hints
    if re.search(r"\btable\b|list|dikhao|batao", lower):
        bundle.output_hint = "table_with_sources"
    if re.search(r"follow-?up|draft", lower):
        bundle.output_hint = bundle.output_hint or "draft_with_table"

    # Time scope
    for t in bundle.times:
        if t.get("ambiguous"):
            bundle.unknown_policies["time"] = "clarify_kal"
        else:
            bundle.conditions.append(
                Condition("time_window", "eq", t["relative"])
            )

    return bundle
