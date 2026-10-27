"""Multi-turn slot filling — ask for missing parameters, then resume."""

from __future__ import annotations

import re
from typing import Any, Optional

from .intents import extract_project_id
from .pipeline import SemanticFrame
from .slots import extract_slots


# Operations that require specific slots before execution
REQUIRED_SLOTS: dict[str, list[str]] = {
    "query_invoices": [],  # project optional
    "query_deliveries": ["project_id"],
    "prepare_followup": ["project_id"],
    "reserve_stock": ["material", "quantity"],
    "record_payment": ["invoice_id", "payment_amount_inr"],
    "create_purchase_order": ["project_id", "vendor_id", "item", "qty", "unit_price_inr"],
    "create_po": ["project_id", "vendor_id", "item", "qty", "unit_price_inr"],
    "export_report": ["report_kind"],
    "understand_scene": ["image_path"],
    "detect_objects": ["image_path"],
    "detect_vehicles": ["image_path"],
    "count_people": ["image_path"],
    "detect_fire": ["image_path"],
    "scan_document_vision": ["image_path"],
    "ppe_compliance": ["image_path"],
    "estimate_queue": ["image_path"],
    # vision_status needs no image
}


QUESTIONS = {
    "project_id": "Which project should I use? (e.g. Project A)",
    "invoice_id": "Which invoice id should I use? (e.g. invoice 43)",
    "payment_amount_inr": "What payment amount in INR should I record?",
    "material": "Which material should I reserve?",
    "quantity": "What quantity should I reserve?",
    "vendor_id": "Which vendor id? (e.g. vendor_buildco)",
    "item": "What item should the purchase order cover?",
    "qty": "What quantity for the purchase order?",
    "unit_price_inr": "What unit price in INR?",
    "report_kind": "Which report? (invoices, payments, alerts, purchase_orders)",
    "image_path": "Which image/video path should I analyze? (e.g. ./photo.jpg)",
}


class ClarificationManager:
    """Detect missing slots and resume a pending proposal from short replies."""

    def missing_slots(self, frame: SemanticFrame) -> list[str]:
        needed = list(REQUIRED_SLOTS.get(frame.operation, []))
        missing: list[str] = []
        cond = {c.get("field"): c.get("value") for c in (frame.conditions or [])}
        for slot in needed:
            if slot == "project_id":
                if not frame.project_id:
                    missing.append(slot)
            elif slot == "material":
                if not any(k in cond for k in ("material", "material_name", "item")):
                    missing.append(slot)
            elif slot == "quantity":
                if not any(k in cond for k in ("quantity", "qty", "reserve_qty")):
                    missing.append(slot)
            elif slot == "report_kind":
                if "report_kind" not in cond:
                    missing.append(slot)
            elif slot not in cond or cond.get(slot) in (None, ""):
                missing.append(slot)
        return missing

    def questions_for(self, slots: list[str]) -> list[str]:
        return [QUESTIONS.get(s, f"Please provide {s}.") for s in slots]

    def try_resume(self, text: str, pending: dict[str, Any]) -> Optional[SemanticFrame]:
        """If dialogue has a pending proposal, fill slots from a short reply."""
        if not pending or pending.get("type") != "slot_fill":
            return None
        op = pending.get("operation")
        if not op:
            return None
        remaining = list(pending.get("missing_slots") or [])
        if not remaining:
            return None

        filled = dict(pending.get("filled") or {})
        conditions = list(pending.get("conditions") or [])
        project_id = pending.get("project_id")
        lower = text.strip().lower()

        # Project-only reply
        if "project_id" in remaining:
            pid = extract_project_id(text)
            if not pid and re.fullmatch(r"[a-zA-Z]", text.strip()):
                pid = f"project_{text.strip().lower()}"
            if pid:
                project_id = pid
                filled["project_id"] = pid
                remaining = [s for s in remaining if s != "project_id"]

        slots = extract_slots(text, project_id=project_id)
        cond_map = {c.field: c.value for c in slots.conditions}
        for slot in list(remaining):
            if slot in cond_map:
                filled[slot] = cond_map[slot]
                conditions.append({"field": slot, "operator": "eq", "value": cond_map[slot]})
                remaining.remove(slot)
            elif slot == "invoice_id":
                m = re.search(r"\b(?:invoice|inv)\s*#?\s*(\d+)\b", lower)
                if m:
                    val = f"inv_{m.group(1)}"
                    filled[slot] = val
                    conditions.append({"field": slot, "operator": "eq", "value": val})
                    remaining.remove(slot)
            elif slot == "image_path":
                m = re.search(
                    r"[\"']?((?:/|\./)?[^\s\"']+\.(?:jpg|jpeg|png|bmp|webp|mp4|avi))[\"']?",
                    text,
                    flags=re.I,
                )
                if m:
                    filled[slot] = m.group(1)
                    conditions.append({"field": slot, "operator": "eq", "value": m.group(1)})
                    remaining.remove(slot)
                elif text.strip() and not text.strip().endswith("?"):
                    # Accept bare path reply
                    filled[slot] = text.strip().strip("\"'")
                    conditions.append({"field": slot, "operator": "eq", "value": filled[slot]})
                    remaining.remove(slot)
            elif slot in ("payment_amount_inr", "unit_price_inr", "qty", "quantity"):
                if slots.amounts:
                    val = slots.amounts[0]["amount_inr"]
                    filled[slot] = val
                    conditions.append({"field": slot, "operator": "eq", "value": val})
                    remaining.remove(slot)
                else:
                    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
                    if m:
                        val = float(m.group(1))
                        filled[slot] = val
                        conditions.append({"field": slot, "operator": "eq", "value": val})
                        remaining.remove(slot)
            elif slot == "vendor_id":
                m = re.search(r"\b(vendor_[a-z0-9_]+|buildco|colorworks)\b", lower)
                if m:
                    raw = m.group(1)
                    val = raw if raw.startswith("vendor_") else f"vendor_{raw}"
                    filled[slot] = val
                    conditions.append({"field": slot, "operator": "eq", "value": val})
                    remaining.remove(slot)
            elif slot in ("item", "material"):
                # treat whole short reply as item name if alphabetic
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{1,40}", text.strip()):
                    filled[slot] = text.strip()
                    conditions.append({"field": slot, "operator": "eq", "value": text.strip()})
                    remaining.remove(slot)
            elif slot == "report_kind":
                for kind in ("invoices", "payments", "alerts", "purchase_orders", "contracts", "pos"):
                    if kind in lower or kind.replace("_", " ") in lower:
                        filled[slot] = "purchase_orders" if kind == "pos" else kind
                        conditions.append({"field": slot, "operator": "eq", "value": filled[slot]})
                        remaining.remove(slot)
                        break

        # Merge previous conditions (avoid dup fields)
        seen = {c["field"] for c in conditions}
        for c in pending.get("conditions") or []:
            if c.get("field") not in seen:
                conditions.append(c)
                seen.add(c.get("field"))

        needs = self.questions_for(remaining)
        frame = SemanticFrame(
            operation=op,
            project_id=project_id,
            conditions=conditions,
            original_text=text,
            normalized_text=text,
            needs_clarification=needs,
            confidence=0.85 if not remaining else 0.5,
            supported=True,
            intent_type="request",
        )
        # Stash resume meta for dialogue tracker
        frame.unknown_policies = {
            "slot_fill_resume": "1",
            "still_missing": ",".join(remaining),
            "filled": ",".join(sorted(filled.keys())),
        }
        return frame

    def make_pending(self, frame: SemanticFrame, missing: list[str]) -> dict[str, Any]:
        return {
            "type": "slot_fill",
            "operation": frame.operation,
            "project_id": frame.project_id,
            "conditions": list(frame.conditions or []),
            "missing_slots": list(missing),
            "filled": {},
            "questions": self.questions_for(missing),
        }
