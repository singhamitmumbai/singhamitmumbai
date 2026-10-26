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
            if k in conds and conds[k] != v:
                ok = False
            elif k not in conds and k in ("reserve_qty", "amount_inr", "task_title"):
                # amount may be present; task_title may be in conditions
                if k == "task_title":
                    ok = ok and any(
                        c.get("field") == "task_title" and str(c.get("value")).lower().find(str(v).lower()) >= 0
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
