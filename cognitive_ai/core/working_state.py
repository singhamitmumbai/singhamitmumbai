"""Typed working state: current task, evidence, goals and dialogue focus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .ids import new_id
from .records import (
    ActionResult,
    Claim,
    Entity,
    GoalRecord,
    Observation,
    PlanRecord,
    UncertaintyKind,
    utc_now,
)


@dataclass
class DialogueState:
    active_topic: Optional[str] = None
    selected_entities: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    current_proposal: Optional[dict[str, Any]] = None
    confirmed_parameters: dict[str, Any] = field(default_factory=dict)
    last_verified_action: Optional[str] = None
    corrections: list[dict[str, Any]] = field(default_factory=list)
    cancellations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "active_topic": self.active_topic,
            "selected_entities": list(self.selected_entities),
            "unresolved_questions": list(self.unresolved_questions),
            "current_proposal": self.current_proposal,
            "confirmed_parameters": dict(self.confirmed_parameters),
            "last_verified_action": self.last_verified_action,
            "corrections": list(self.corrections),
            "cancellations": list(self.cancellations),
        }


@dataclass
class UncertaintyNote:
    kind: UncertaintyKind
    message: str
    related_ids: list[str] = field(default_factory=list)
    note_id: str = field(default_factory=lambda: new_id("unc"))
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "kind": self.kind.value,
            "message": self.message,
            "related_ids": list(self.related_ids),
            "created_at": self.created_at,
        }


class WorkingState:
    """Shared structured situation model used by all specialist components."""

    def __init__(self) -> None:
        self.session_id: str = new_id("sess")
        self.observations: list[Observation] = []
        self.entities: dict[str, Entity] = {}
        self.claims: dict[str, Claim] = {}
        self.goals: dict[str, GoalRecord] = {}
        self.active_goal_id: Optional[str] = None
        self.plans: dict[str, PlanRecord] = {}
        self.action_results: list[ActionResult] = []
        self.dialogue = DialogueState()
        self.uncertainties: list[UncertaintyNote] = []
        self.semantic_frame: Optional[dict[str, Any]] = None
        self.last_answer: Optional[dict[str, Any]] = None
        self.resource_budget: dict[str, Any] = {
            "time_ms": 10000,
            "steps": 50,
            "external_calls": 20,
        }
        self.updated_at: str = utc_now()

    # --- mutations ------------------------------------------------------------

    def add_observation(self, obs: Observation) -> Observation:
        self.observations.append(obs)
        self.touch()
        return obs

    def upsert_entity(self, entity: Entity) -> Entity:
        self.entities[entity.entity_id] = entity
        self.touch()
        return entity

    def add_claim(self, claim: Claim) -> Claim:
        # Detect simple conflicts: same subject+relationship, different value
        for existing in self.claims.values():
            if (
                existing.subject == claim.subject
                and existing.relationship == claim.relationship
                and existing.value != claim.value
                and existing.evidence_status.value != "unknown"
            ):
                claim.conflicts_with.append(existing.claim_id)
                existing.conflicts_with.append(claim.claim_id)
                self.note_uncertainty(
                    UncertaintyKind.CONFLICTING_EVIDENCE,
                    f"Conflict on {claim.subject}.{claim.relationship}: "
                    f"{existing.value!r} vs {claim.value!r}",
                    [existing.claim_id, claim.claim_id],
                )
        self.claims[claim.claim_id] = claim
        self.touch()
        return claim

    def set_goal(self, goal: GoalRecord) -> GoalRecord:
        self.goals[goal.goal_id] = goal
        self.active_goal_id = goal.goal_id
        self.touch()
        return goal

    def set_plan(self, plan: PlanRecord) -> PlanRecord:
        self.plans[plan.plan_id] = plan
        self.touch()
        return plan

    def record_action(self, result: ActionResult) -> ActionResult:
        self.action_results.append(result)
        if result.status.value == "CONFIRMED":
            self.dialogue.last_verified_action = result.result_id
        self.touch()
        return result

    def set_semantic_frame(self, frame: dict[str, Any]) -> None:
        self.semantic_frame = frame
        self.dialogue.active_topic = frame.get("operation") or frame.get("intent")
        self.touch()

    def note_uncertainty(
        self,
        kind: UncertaintyKind,
        message: str,
        related_ids: Optional[list[str]] = None,
    ) -> UncertaintyNote:
        note = UncertaintyNote(kind=kind, message=message, related_ids=related_ids or [])
        self.uncertainties.append(note)
        self.dialogue.unresolved_questions.append(message)
        self.touch()
        return note

    def apply_correction(self, field: str, old: Any, new: Any) -> None:
        self.dialogue.corrections.append(
            {"field": field, "old": old, "new": new, "at": utc_now()}
        )
        self.dialogue.confirmed_parameters[field] = new
        # Invalidate plans that may depend on corrected parameters
        for plan in self.plans.values():
            if plan.outcome in ("feasible", "optimal"):
                plan.outcome = "stale"
                plan.explanation = f"Invalidated by correction of {field}"
                for step in plan.steps:
                    if step.status == "pending":
                        step.status = "invalidated"
        self.touch()

    def touch(self) -> None:
        self.updated_at = utc_now()

    # --- queries --------------------------------------------------------------

    def claims_for(self, subject: str, relationship: Optional[str] = None) -> list[Claim]:
        out = []
        for c in self.claims.values():
            if c.subject != subject:
                continue
            if relationship and c.relationship != relationship:
                continue
            out.append(c)
        return out

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        key = name.strip().lower()
        for e in self.entities.values():
            labels = [x.lower() for x in e.all_labels()]
            if key in labels:
                return e
        return None

    def active_goal(self) -> Optional[GoalRecord]:
        if not self.active_goal_id:
            return None
        return self.goals.get(self.active_goal_id)

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "observations": len(self.observations),
            "entities": len(self.entities),
            "claims": len(self.claims),
            "goals": len(self.goals),
            "active_goal": self.active_goal_id,
            "plans": len(self.plans),
            "actions": len(self.action_results),
            "uncertainties": len(self.uncertainties),
            "dialogue": self.dialogue.to_dict(),
            "semantic_frame": self.semantic_frame,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "claims_detail": [c.to_dict() for c in self.claims.values()],
            "entities_detail": [e.to_dict() for e in self.entities.values()],
            "last_answer": self.last_answer,
        }
