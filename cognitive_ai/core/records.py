"""Common structured record types shared across the entire system.

Three distinctions that must survive everywhere:
- Unknown ≠ false
- Predicted ≠ observed
- Attempted ≠ confirmed
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .ids import new_id


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceStatus(str, Enum):
    OBSERVED = "observed"
    ASSERTED = "asserted"
    INFERRED = "inferred"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"


class ActionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class UncertaintyKind(str, Enum):
    RECOGNITION = "recognition"
    MISSING_INFORMATION = "missing_information"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    STALE_INFORMATION = "stale_information"
    UNSUPPORTED_TASK = "unsupported_task"
    FORECAST = "forecast"


@dataclass
class Observation:
    """Raw or interpreted sensory input with provenance and uncertainty."""

    source: str
    modality: str  # text | audio | image | document | sensor | structured
    raw_evidence: Any
    extracted_value: Any = None
    recognition_version: str = "v0"
    uncertainty: float = 0.0
    observation_id: str = field(default_factory=lambda: new_id("obs"))
    time: str = field(default_factory=utc_now)
    context: dict[str, Any] = field(default_factory=dict)
    alternatives: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Entity:
    """Stable identity for a thing the system understands."""

    entity_type: str
    names: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    entity_id: str = field(default_factory=lambda: new_id("ent"))
    attributes: dict[str, Any] = field(default_factory=dict)
    identity_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def all_labels(self) -> list[str]:
        return list(dict.fromkeys([*self.names, *self.aliases]))


@dataclass
class Claim:
    """A proposition about the world with evidence status and validity."""

    subject: str
    relationship: str
    value: Any
    units: Optional[str] = None
    source: str = "unknown"
    validity_start: Optional[str] = None
    validity_end: Optional[str] = None
    evidence_status: EvidenceStatus = EvidenceStatus.ASSERTED
    claim_id: str = field(default_factory=lambda: new_id("claim"))
    confidence: float = 1.0
    project_id: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    notes: str = ""
    conflicts_with: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence_status"] = self.evidence_status.value
        return d

    @property
    def is_observed(self) -> bool:
        return self.evidence_status == EvidenceStatus.OBSERVED

    @property
    def is_unknown(self) -> bool:
        return self.evidence_status == EvidenceStatus.UNKNOWN


@dataclass
class GoalRecord:
    """Desired conditions with budget, scope and completion criteria."""

    description: str
    desired_conditions: list[dict[str, Any]] = field(default_factory=list)
    requester: str = "user"
    scope: dict[str, Any] = field(default_factory=dict)
    deadline: Optional[str] = None
    priority: float = 1.0
    budget: dict[str, Any] = field(default_factory=lambda: {"time_ms": 5000, "steps": 20})
    permitted_operations: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    cancellation_conditions: list[str] = field(default_factory=list)
    goal_id: str = field(default_factory=lambda: new_id("goal"))
    status: str = "open"  # open|active|satisfied|failed|abandoned|infeasible|missing_info
    created_at: str = field(default_factory=utc_now)
    parent_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanStepRecord:
    operator: str
    args: dict[str, Any] = field(default_factory=dict)
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    expected_effects: list[dict[str, Any]] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    step_id: str = field(default_factory=lambda: new_id("step"))
    status: str = "pending"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanRecord:
    goal_id: str
    steps: list[PlanStepRecord] = field(default_factory=list)
    outcome: str = "unknown"  # already_satisfied|feasible|optimal|no_solution|infeasible|missing_info|unsupported
    explanation: str = ""
    plan_id: str = field(default_factory=lambda: new_id("plan"))
    created_at: str = field(default_factory=utc_now)
    cost: float = 0.0

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "outcome": self.outcome,
            "explanation": self.explanation,
            "cost": self.cost,
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class ActionResult:
    """Outcome of an attempted operation — may be OUTCOME_UNKNOWN."""

    attempt_id: str
    operator: str
    status: ActionStatus
    args: dict[str, Any] = field(default_factory=dict)
    external_receipt: Optional[str] = None
    observed_effects: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    started_at: str = field(default_factory=utc_now)
    finished_at: Optional[str] = None
    result_id: str = field(default_factory=lambda: new_id("act"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class LearningExample:
    """Reviewed training / evaluation sample with provenance."""

    input_data: Any
    verified_target: Any
    task: str
    reviewer: str = "system"
    source_group: str = "default"
    dataset_split: str = "train"  # train|dev|calibration|test
    example_id: str = field(default_factory=lambda: new_id("ex"))
    label_schema_version: str = "v1"
    language: str = "en"
    content_hash: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
