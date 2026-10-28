"""SCAN-inspired compositional generalization tests for Finderbay NLU+actions.

Success on nearby paraphrases does NOT imply systematic composition.
This suite measures held-out concept combinations explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ScanCase:
    case_id: str
    split: str  # train_like | compositional_holdout
    text: str
    expect_op: str
    expect_slots: dict[str, Any]


# Train-like: familiar combinations
# Holdout: familiar concepts combined in unseen ways
CASES = [
    ScanCase("tl1", "train_like", "Show unpaid invoices for Project A", "query_invoices", {"project_id": "project_a"}),
    ScanCase("tl2", "train_like", "Reserve 10 bags of cement", "reserve_stock", {"reserve_qty": 10.0}),
    ScanCase(
        "tl3",
        "train_like",
        "Create task for Project A: Safety check",
        "manage_task",
        {"task_title": "Safety check"},
    ),
    # Compositional holdouts
    ScanCase(
        "ch1",
        "compositional_holdout",
        "Show unpaid invoices above ₹2 lakh for Project B, excluding disputed ones",
        "query_invoices",
        {"project_id": "project_b", "amount_inr": 200000.0},
    ),
    ScanCase(
        "ch2",
        "compositional_holdout",
        "From last week's delivery records find pending items for Project B and prepare a follow-up. Do not send it.",
        "prepare_followup",
        {"project_id": "project_b"},
    ),
    ScanCase(
        "ch3",
        "compositional_holdout",
        "Reserve 7 bags of cement for Project B",
        "reserve_stock",
        {"reserve_qty": 7.0},
    ),
    ScanCase(
        "ch4",
        "compositional_holdout",
        "Create task for Project B: Compose delivery checklist",
        "manage_task",
        {"project_id": "project_b"},
    ),
    ScanCase(
        "ch5",
        "compositional_holdout",
        "List tasks for Project B",
        "manage_task",
        {"project_id": "project_b"},
    ),
    # v1.22 expanded compositional coverage
    ScanCase(
        "ch6",
        "compositional_holdout",
        "Show payments for Project B",
        "query_payments",
        {"project_id": "project_b"},
    ),
    ScanCase(
        "ch7",
        "compositional_holdout",
        "Show contracts for Project A",
        "query_contracts",
        {"project_id": "project_a"},
    ),
    ScanCase(
        "tl4",
        "train_like",
        "Show payments for Project A",
        "query_payments",
        {"project_id": "project_a"},
    ),
    ScanCase(
        "tl5",
        "train_like",
        "Show contracts",
        "query_contracts",
        {},
    ),
    ScanCase(
        "ch8",
        "compositional_holdout",
        "Reserve 3 bags of steel for Project A",
        "reserve_stock",
        {"reserve_qty": 3.0, "project_id": "project_a"},
    ),
    # v1.26 stronger composition: ranges, under≠overdue, exclusions, stock qty
    ScanCase(
        "ch9",
        "compositional_holdout",
        "Find paid invoices for Project C between ₹1 lakh and ₹3 lakh",
        "query_invoices",
        {"project_id": "project_c", "amount_inr_gte": 100000.0, "amount_inr_lte": 300000.0},
    ),
    ScanCase(
        "ch10",
        "compositional_holdout",
        "Show overdue unpaid invoices for Project A under ₹50000 excluding cancelled",
        "query_invoices",
        {"project_id": "project_a", "amount_inr": 50000.0},
    ),
    ScanCase(
        "ch11",
        "compositional_holdout",
        "List materials below 20 units for Project A excluding scrap",
        "query_materials",
        {"project_id": "project_a", "stock_qty": 20.0},
    ),
    ScanCase(
        "tl6",
        "train_like",
        "Show audit log",
        "query_audit_log",
        {},
    ),
]


def run_scan_benchmark(parse_fn: Callable[[str], Any]) -> dict[str, Any]:
    """parse_fn(text) -> SemanticFrame-like with operation/conditions/project_id."""
    by_split = {"train_like": {"passed": 0, "failed": 0}, "compositional_holdout": {"passed": 0, "failed": 0}}
    details = []
    for case in CASES:
        frame = parse_fn(case.text)
        # support SemanticFrame or dict
        if hasattr(frame, "to_dict"):
            d = frame.to_dict()
        else:
            d = dict(frame)
        ok = d.get("operation") == case.expect_op
        conds = {c["field"]: c["value"] for c in d.get("conditions") or [] if "field" in c}
        if case.expect_slots.get("project_id"):
            ok = ok and d.get("project_id") == case.expect_slots["project_id"]
        for k, v in case.expect_slots.items():
            if k == "project_id":
                continue
            # Range helpers: amount_inr_gte / amount_inr_lte map to conditions with ops
            if k.endswith("_gte") or k.endswith("_lte") or k.endswith("_gt") or k.endswith("_lt"):
                field, _, op_suffix = k.rpartition("_")
                op_map = {"gte": "gte", "lte": "lte", "gt": "gt", "lt": "lt"}
                want_op = op_map.get(op_suffix)
                found = any(
                    c.get("field") == field and c.get("operator") == want_op and c.get("value") == v
                    for c in (d.get("conditions") or [])
                )
                if not found:
                    ok = False
                continue
            if k in conds and conds[k] != v:
                ok = False
            elif k not in conds and k in ("reserve_qty", "amount_inr", "task_title", "stock_qty"):
                # amount may be present; task_title may be in conditions
                if k == "task_title":
                    ok = ok and any(
                        c.get("field") == "task_title" and str(c.get("value")).lower().find(str(v).lower()) >= 0
                        for c in (d.get("conditions") or [])
                    )
                elif k == "stock_qty":
                    ok = ok and any(
                        c.get("field") == "stock_qty" and c.get("value") == v
                        for c in (d.get("conditions") or [])
                    )
                elif k == "amount_inr":
                    # under/above may use lt/gt — accept matching numeric value
                    ok = ok and any(
                        c.get("field") == "amount_inr" and c.get("value") == v
                        for c in (d.get("conditions") or [])
                    )
                else:
                    ok = False
        details.append({"case_id": case.case_id, "split": case.split, "passed": ok, "got": d.get("operation")})
        bucket = by_split[case.split]
        if ok:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    for split, b in by_split.items():
        total = b["passed"] + b["failed"]
        b["accuracy"] = b["passed"] / total if total else 0.0
    return {
        "by_split": by_split,
        "compositional_generalization": by_split["compositional_holdout"]["accuracy"],
        "details": details,
        "note": "High train_like accuracy with low holdout accuracy indicates weak systematic composition.",
    }
