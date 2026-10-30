"""Improvement process: collect → review → candidate → evaluate → release (blueprint §2, §13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now


@dataclass
class FailureRecord:
    failure_id: str
    operation: str
    input_text: str
    reason: str
    stage: str = "unknown"
    created_at: str = field(default_factory=utc_now)
    reviewed: bool = False
    reviewer: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "operation": self.operation,
            "input_text": self.input_text,
            "reason": self.reason,
            "stage": self.stage,
            "created_at": self.created_at,
            "reviewed": self.reviewed,
            "reviewer": self.reviewer,
        }


@dataclass
class ImprovementCandidate:
    candidate_id: str
    kind: str  # alias|rule|procedure|dataset
    payload: dict[str, Any]
    status: str = "proposed"  # proposed|accepted|rejected|released
    evidence: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "status": self.status,
            "evidence": list(self.evidence),
            "created_at": self.created_at,
        }


class ImprovementLoop:
    """Bounded non-LLM improvement process — never self-labels from unverified predictions."""

    def __init__(self) -> None:
        self.failures: list[FailureRecord] = []
        self.candidates: list[ImprovementCandidate] = []
        self.releases: list[dict[str, Any]] = []

    def record_failure(
        self,
        *,
        operation: str,
        input_text: str,
        reason: str,
        stage: str = "unknown",
    ) -> FailureRecord:
        rec = FailureRecord(
            failure_id=new_id("fail"),
            operation=operation,
            input_text=input_text[:500],
            reason=reason,
            stage=stage,
        )
        self.failures.append(rec)
        return rec

    def propose_from_failure(
        self,
        failure: FailureRecord,
        *,
        kind: str = "procedure",
        payload: Optional[dict] = None,
    ) -> ImprovementCandidate:
        cand = ImprovementCandidate(
            candidate_id=new_id("impr"),
            kind=kind,
            payload=payload
            or {
                "from_failure": failure.failure_id,
                "operation": failure.operation,
                "hint": failure.reason,
            },
            evidence=[failure.failure_id],
        )
        self.candidates.append(cand)
        return cand

    def review_candidate(
        self,
        candidate_id: str,
        *,
        accept: bool,
        reviewer: str,
    ) -> ImprovementCandidate:
        cand = self._get(candidate_id)
        if reviewer in ("model", "self", "unverified"):
            raise ValueError("Unverified model reviewers cannot accept improvement candidates.")
        cand.status = "accepted" if accept else "rejected"
        cand.payload["reviewer"] = reviewer
        return cand

    def release(self, candidate_id: str) -> dict[str, Any]:
        cand = self._get(candidate_id)
        if cand.status != "accepted":
            raise ValueError("Only accepted candidates can be released.")
        cand.status = "released"
        release = {
            "release_id": new_id("rel"),
            "candidate_id": cand.candidate_id,
            "kind": cand.kind,
            "payload": cand.payload,
            "released_at": utc_now(),
        }
        self.releases.append(release)
        return release

    def summary(self) -> dict[str, Any]:
        return {
            "failures": len(self.failures),
            "candidates": len(self.candidates),
            "by_status": {
                s: sum(1 for c in self.candidates if c.status == s)
                for s in ("proposed", "accepted", "rejected", "released")
            },
            "releases": len(self.releases),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "recent_failures": [f.to_dict() for f in self.failures[-10:]],
            "candidates": [c.to_dict() for c in self.candidates[-10:]],
            "releases": list(self.releases[-10:]),
        }

    def _get(self, candidate_id: str) -> ImprovementCandidate:
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        raise KeyError(candidate_id)
