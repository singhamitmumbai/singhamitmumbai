"""Evaluation suite for the common-sense KB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reasoner import CommonSenseReasoner


@dataclass
class CSCase:
    case_id: str
    kind: str  # can|is_a|used_for|analogy|exception
    args: tuple
    expect_success: bool
    expect_contains: str = ""


CASES = [
    CSCase("bird_fly", "can", ("sparrow", "can", "fly"), True, "fly"),
    CSCase("fish_swim", "can", ("salmon", "can", "swim"), True, "swim"),
    CSCase("dog_mammal", "is_a", ("dog", "is_a", "mammal"), True, "mammal"),
    CSCase("penguin_no_fly", "exception", ("penguin", "can", "fly"), True, "cannot"),
    CSCase("ostrich_no_fly", "exception", ("ostrich", "can", "fly"), True, "cannot"),
    CSCase("knife_cut", "used_for", ("knife", "used_for", "cutting"), True, "cutting"),
    CSCase("umbrella_rain", "used_for", ("umbrella", "used_for", "rain_protection"), True, "rain"),
    CSCase("mumbai_india", "is_a", ("mumbai", "is_in", "india"), True, "india"),
    CSCase("doctor_patients", "used_for", ("doctor", "helps", "patients"), True, "patient"),
    CSCase("water_boil", "can", ("water", "boils_at_c", "100"), True, "100"),
    CSCase("analogy_knife", "analogy", ("knife", "cutting", "umbrella"), True, ""),
    CSCase("analogy_dog_cat", "analogy", ("dog", "bark", "cat"), True, "meow"),
    CSCase("unknown_entity", "can", ("unicorn_xyz", "can", "fly"), False, "lack"),
    CSCase("opposite_hot", "is_a", ("hot", "opposite_of", "cold"), True, "cold"),
    CSCase("eye_seeing", "used_for", ("eye", "used_for", "seeing"), True, "seeing"),
]


def run_commonsense_eval(reasoner: CommonSenseReasoner | None = None) -> dict[str, Any]:
    reasoner = reasoner or CommonSenseReasoner()
    passed = failed = 0
    results = []
    for case in CASES:
        if case.kind == "analogy":
            ans = reasoner.analogy(*case.args)
        else:
            ans = reasoner.reason(*case.args)
        ok = ans.success == case.expect_success
        if case.expect_contains:
            ok = ok and case.expect_contains.lower() in ans.text.lower()
        results.append(
            {
                "case_id": case.case_id,
                "passed": ok,
                "text": ans.text,
                "method": ans.method,
            }
        )
        if ok:
            passed += 1
        else:
            failed += 1
    return {
        "passed": passed,
        "failed": failed,
        "accuracy": passed / max(1, passed + failed),
        "kb": reasoner.cs.summary(),
        "results": results,
    }
