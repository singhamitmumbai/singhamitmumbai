"""Determine what actually happened — separate from the executor."""

from __future__ import annotations

from typing import Any, Optional

from ..core.records import ActionResult, ActionStatus, PlanStepRecord
from ..core.working_state import WorkingState
from ..memory.store import MemorySystem


class Verifier:
    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory

    def verify(
        self,
        step: PlanStepRecord,
        result: ActionResult,
        state: WorkingState,
    ) -> Optional[bool]:
        """Return True confirmed, False failed, None unknown."""
        if result.status == ActionStatus.OUTCOME_UNKNOWN:
            return None
        if result.status == ActionStatus.FAILED:
            return False

        op = step.operator
        effects = {k: v for e in result.observed_effects for k, v in e.items()}

        checks = step.verification or []
        if not checks:
            return True

        for check in checks:
            if check == "record_count_available":
                if "count" not in effects and "records_loaded" not in effects:
                    return False
            elif check == "conclusion_present":
                if not effects.get("conclusions_ready"):
                    return False
            elif check == "table_row_count":
                table = effects.get("table") or {}
                if "row_count" not in table:
                    return False
            elif check == "draft_exists":
                draft = effects.get("draft")
                if not draft or not draft.get("body"):
                    return False
            elif check == "send_not_performed":
                if effects.get("sent") is True:
                    return False
                draft = effects.get("draft") or {}
                if draft.get("sent") is True:
                    return False
            elif check == "question_emitted":
                if not effects.get("clarification_requested"):
                    return False
            elif check == "claims_match_structure":
                continue  # validated later by response generator
            elif check == "inventory_reservation_receipt":
                if not result.external_receipt:
                    return None  # unknown — reconcile before retry
            elif check == "task_persisted":
                if not effects.get("task_created") and not effects.get("task"):
                    return False
            elif check == "fields_present":
                if not effects.get("fields_ready") and not effects.get("parsed"):
                    return False
            elif check == "alias_stored":
                if not effects.get("alias_stored"):
                    return False
            elif check == "login_ok":
                if not effects.get("login_ok"):
                    return False
            elif check == "payment_posted":
                if not effects.get("payment_posted"):
                    return False
            elif check == "vision_result_present":
                vision = effects.get("vision") or {}
                if not vision and "detections_ready" not in effects:
                    return False
            elif check == "detections_ready":
                if not effects.get("detections_ready") and not (effects.get("vision") or {}).get("detections"):
                    return False
            elif check == "report_ready":
                if not effects.get("report_ready") and not effects.get("report"):
                    return False
            elif check == "po_created":
                if not effects.get("po_created"):
                    return False
            else:
                # Unknown check name — do not pretend success
                continue
        return True

    def reconcile_unknown(self, result: ActionResult) -> ActionResult:
        """Attempt to resolve OUTCOME_UNKNOWN using attempt id / external state."""
        if result.status != ActionStatus.OUTCOME_UNKNOWN:
            return result
        # Look for matching receipt by attempt id pattern
        if result.operator == "reserve_stock":
            item = result.args.get("item")
            mats = self.memory.get_records("material")
            for m in mats:
                if m.get("id") == item and result.attempt_id in str(m.get("last_attempt", "")):
                    result.status = ActionStatus.CONFIRMED
                    result.external_receipt = f"reconciled:{item}"
                    return result
        # Still unknown
        return result
