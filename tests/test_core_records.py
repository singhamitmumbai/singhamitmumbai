from cognitive_ai.core.records import Claim, EvidenceStatus, ActionStatus, ActionResult
from cognitive_ai.core.working_state import WorkingState


def test_conflict_detection():
    state = WorkingState()
    state.add_claim(
        Claim("inv_1", "payment_status", "unpaid", evidence_status=EvidenceStatus.OBSERVED)
    )
    state.add_claim(
        Claim("inv_1", "payment_status", "paid", evidence_status=EvidenceStatus.ASSERTED)
    )
    assert state.uncertainties
    assert state.uncertainties[0].kind.value == "conflicting_evidence"


def test_distinctions_encoded():
    assert EvidenceStatus.UNKNOWN != EvidenceStatus.ASSERTED
    assert EvidenceStatus.PREDICTED != EvidenceStatus.OBSERVED
    assert ActionStatus.OUTCOME_UNKNOWN != ActionStatus.CONFIRMED
    r = ActionResult(attempt_id="a1", operator="reserve_stock", status=ActionStatus.OUTCOME_UNKNOWN)
    assert r.status == ActionStatus.OUTCOME_UNKNOWN
