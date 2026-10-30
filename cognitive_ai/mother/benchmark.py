"""FinderBench — lightweight general-intelligence flavored smoke suite.

Not AGI. Category scores are heuristic capability checks over MotherBrain.
"""

from __future__ import annotations

from typing import Any, Callable

from .brain import MotherBrain


CASES: list[dict[str, Any]] = [
    # Language / conversation
    {"id": "lu1", "cat": "Language Understanding", "ask": "hi", "expect": "hello|ready|mother|finder"},
    {"id": "lu2", "cat": "Language Understanding", "ask": "how are you", "expect": "function|ready|help|mother"},
    {"id": "lu3", "cat": "Language Understanding", "ask": "kaise ho", "expect": "function|ready|help|mother|hello"},
    {"id": "lu4", "cat": "Language Understanding", "ask": "who are you", "expect": "mother|finder|cognitive|ai"},
    {"id": "lu5", "cat": "Language Understanding", "ask": "tum kya kar sakte ho", "expect": "try:|invoice|mother|coding|help"},
    # Reasoning / Q
    {"id": "r1", "cat": "Reasoning", "ask": "Can penguins fly?", "expect": "cannot|no|flightless|not"},
    {"id": "r2", "cat": "Reasoning", "ask": "Calculate 7*(3+2)", "expect": "35"},
    # Memory / goals
    {"id": "m1", "cat": "Memory", "ask": "remember that warehouse is Central Store", "expect": "."},
    {"id": "m2", "cat": "Memory", "ask": "what can you do", "expect": "mother|try"},
    # Planning / tools / coding
    {"id": "p1", "cat": "Planning", "ask": "inspect this project and find what is broken", "expect": "diagnostic|mother|repository|gap|complete"},
    {"id": "t1", "cat": "Tool Use", "ask": "Show unpaid invoices for Project A", "expect": "invoice|project|unpaid|₹|inr|matched"},
    {"id": "c1", "cat": "Learning", "ask": "learning status", "expect": "learning|continuous|enabled|tick|human_level"},
    # Metacognition marker
    {"id": "mc1", "cat": "Metacognition", "ask": "hi", "check": "cycle"},
    # Generalization-ish paraphrase
    {"id": "g1", "cat": "Generalization", "ask": "hii", "expect": "hello|ready|mother|finder|symbolic"},
    {"id": "g2", "cat": "Generalization", "ask": "hey there", "expect": "hello|ready|mother|finder|symbolic"},
]


def _match(text: str, pattern: str) -> bool:
    low = (text or "").lower()
    return any(p.strip() and p.strip().lower() in low for p in pattern.split("|"))


def run_finderbench(mother: MotherBrain | None = None) -> dict[str, Any]:
    mb = mother or MotherBrain(seed_demo=True, enable_llm=False)
    by_cat: dict[str, dict[str, int]] = {}
    details = []
    for case in CASES:
        cat = case["cat"]
        by_cat.setdefault(cat, {"passed": 0, "failed": 0})
        try:
            out = mb.handle(case["ask"])
            text = out.get("text") or ""
            ok = True
            if case.get("check") == "cycle":
                ok = bool(out.get("mother_cycle") and out["mother_cycle"].get("stages"))
            elif case.get("expect"):
                ok = _match(text, case["expect"])
            if case["ask"].lower() in ("hii", "hi", "how are you", "kaise ho"):
                # Critical: must NOT be unknown command
                ok = ok and "unknown command" not in text.lower()
        except Exception as exc:  # noqa: BLE001
            ok = False
            text = str(exc)
            out = {"error": str(exc)}
        details.append(
            {
                "id": case["id"],
                "cat": cat,
                "ask": case["ask"],
                "passed": ok,
                "text": (text or "")[:160],
                "route": (out or {}).get("route"),
            }
        )
        if ok:
            by_cat[cat]["passed"] += 1
        else:
            by_cat[cat]["failed"] += 1
    scores = {}
    for cat, b in by_cat.items():
        total = b["passed"] + b["failed"]
        scores[cat] = round(b["passed"] / total, 3) if total else 0.0
    passed = sum(1 for d in details if d["passed"])
    return {
        "suite": "FinderBench",
        "version": mb.product_version,
        "ok": passed == len(details),
        "passed": passed,
        "total": len(details),
        "category_scores": scores,
        "details": details,
        "human_level_agi": False,
        "note": "Smoke capability checks — not a claim of AGI or human-level intelligence.",
    }
