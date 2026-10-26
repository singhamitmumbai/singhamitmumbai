"""Evaluate software correctness, component quality and end-to-end task capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class EvalCase:
    case_id: str
    task: str
    input_text: str
    expected: dict[str, Any]
    modality: str = "text"
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    passed: int
    failed: int
    results: list[dict[str, Any]]
    oracle_comparisons: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        total = self.passed + self.failed
        return self.passed / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "results": self.results,
            "oracle_comparisons": self.oracle_comparisons,
        }


class EvaluationHarness:
    def __init__(self, run_fn: Callable[[str], dict[str, Any]]) -> None:
        self.run_fn = run_fn

    def run_suite(self, cases: list[EvalCase]) -> EvalReport:
        results = []
        passed = failed = 0
        for case in cases:
            out = self.run_fn(case.input_text)
            ok, detail = self._check(case, out)
            results.append(
                {
                    "case_id": case.case_id,
                    "task": case.task,
                    "passed": ok,
                    "detail": detail,
                    "output_text": out.get("text"),
                }
            )
            if ok:
                passed += 1
            else:
                failed += 1
        return EvalReport(passed=passed, failed=failed, results=results)

    def locate_failure(
        self,
        case: EvalCase,
        full_pipeline_out: dict[str, Any],
        oracle_out: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare full pipeline vs oracle-input run to locate failure source."""
        full_ok, _ = self._check(case, full_pipeline_out)
        oracle_ok, _ = self._check(case, oracle_out)
        if not full_ok and oracle_ok:
            source = "perception_or_extraction"
        elif not full_ok and not oracle_ok:
            source = "interpretation_knowledge_planning_or_execution"
        else:
            source = "none"
        return {
            "case_id": case.case_id,
            "full_ok": full_ok,
            "oracle_ok": oracle_ok,
            "failure_source": source,
        }

    def _check(self, case: EvalCase, out: dict[str, Any]) -> tuple[bool, str]:
        structured = out.get("structured") or {}
        expected = case.expected
        for key, val in expected.items():
            if key == "contains":
                if str(val).lower() not in (out.get("text") or "").lower():
                    return False, f"missing text: {val}"
                continue
            if key == "not_contains":
                if str(val).lower() in (out.get("text") or "").lower():
                    return False, f"forbidden text present: {val}"
                continue
            if key == "sent":
                if structured.get("sent") is not False and structured.get("sent") is not None:
                    if structured.get("sent") != val:
                        return False, f"sent={structured.get('sent')} expected {val}"
                elif val is False and structured.get("draft_prepared") and structured.get("sent") is True:
                    return False, "draft was sent but must not be"
                continue
            actual = structured.get(key, out.get(key))
            if actual != val:
                # allow nested table meta
                table = out.get("table") or {}
                meta = table.get("meta") or {}
                if meta.get(key) == val or table.get(key) == val:
                    continue
                return False, f"{key}: got {actual!r} expected {val!r}"
        return True, "ok"
