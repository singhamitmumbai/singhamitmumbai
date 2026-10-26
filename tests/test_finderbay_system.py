"""End-to-end tests for the complete Finderbay non-LLM AI."""

from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.core.records import EvidenceStatus


def test_invoice_query_end_to_end():
    ai = CompleteNonLLMAI(seed_demo=True)
    result = ai.handle(
        "Show unpaid invoices above ₹1 lakh for Project A, excluding disputed ones."
    )
    assert result["llm_dependency"] is False
    assert result["frame"]["operation"] == "query_invoices"
    assert result["structured"]["count"] == 2
    assert result["structured"]["unresolved_count"] == 1
    assert result["structured"]["total_inr"] == 400_000  # 150k + 250k
    assert result["faithful"] is True
    assert result["table"]["row_count"] == 2


def test_pending_followup_not_sent():
    ai = CompleteNonLLMAI(seed_demo=True)
    result = ai.handle(
        "From this week's delivery records, find the items still pending for "
        "Project A and prepare a follow-up. Do not send it."
    )
    assert result["structured"]["pending_count"] == 1
    assert result["structured"]["uncertain_count"] == 1
    assert result["structured"]["draft_prepared"] is True
    assert result["structured"]["sent"] is False
    assert "not sent" in result["text"].lower()
    # Unknown receipts must produce UNKNOWN claims, not false pending
    unknown_claims = [
        c for c in ai.state.claims.values() if c.evidence_status == EvidenceStatus.UNKNOWN
    ]
    assert unknown_claims


def test_unknown_not_false():
    ai = CompleteNonLLMAI(seed_demo=True)
    ai.handle(
        "From this week's delivery records, find the items still pending for Project A."
    )
    # Sand has missing receipt — must not appear in confirmed pending table
    rows = (ai.executor.last_table or {}).get("rows") or []
    items = [r.get("item") for r in rows]
    assert "Sand (cu.m)" not in items
    assert "Cement bags" in items


def test_evaluation_suite():
    ai = CompleteNonLLMAI(seed_demo=True)
    report = ai.evaluate()
    assert report.failed == 0, report.to_dict()
    assert report.passed >= 4


def test_alias_learning():
    ai = CompleteNonLLMAI(seed_demo=True)
    claim = ai.learner.learn_fact(
        "central_store", "alias", "Warehouse B", source="user", reviewer="tester"
    )
    assert claim.evidence_status == EvidenceStatus.ASSERTED
    assert ai.memory.ontology.resolve("warehouse b") == "central_store"


def test_correction_invalidates_plan():
    ai = CompleteNonLLMAI(seed_demo=True)
    ai.handle("Show unpaid invoices for Project A")
    assert ai.state.plans
    ai.handle("make that fifteen")
    assert ai.state.dialogue.corrections
    assert any(p.outcome == "stale" for p in ai.state.plans.values())
