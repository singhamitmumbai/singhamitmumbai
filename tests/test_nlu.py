"""Tests for non-LLM language understanding."""

from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.nlu.normalize import extract_money, detect_language
from cognitive_ai.nlu.slots import extract_slots


def test_rupee_lakh_parses_to_100000():
    money = extract_money("above ₹1 lakh for Project A")
    assert money
    assert money[0]["amount_inr"] == 100_000


def test_invoice_query_frame():
    nlu = LanguageUnderstanding()
    frame = nlu.parse(
        "Show unpaid invoices above ₹1 lakh for Project A, excluding disputed ones."
    )
    assert frame.operation == "query_invoices"
    assert frame.project_id == "project_a"
    assert frame.supported
    fields = {c["field"]: c for c in frame.conditions}
    assert fields["payment_status"]["value"] == "unpaid"
    assert fields["amount_inr"]["operator"] == "gt"
    assert fields["amount_inr"]["value"] == 100_000
    assert fields["dispute_status"]["value"] == "not_disputed"
    assert frame.unknown_policies.get("dispute_status") == "report_separately"


def test_followup_prohibition():
    nlu = LanguageUnderstanding()
    frame = nlu.parse(
        "From this week's delivery records, find the items still pending for "
        "Project A and prepare a follow-up. Do not send it."
    )
    assert frame.operation == "prepare_followup"
    assert "send" in frame.prohibitions
    assert any(c["value"] == "this_week" for c in frame.conditions)


def test_kal_ambiguity():
    nlu = LanguageUnderstanding()
    frame = nlu.parse("Dikhao pending deliveries for Project A kal")
    assert frame.needs_clarification
    assert any("kal" in q.lower() for q in frame.needs_clarification)


def test_hinglish_detection():
    assert detect_language("kal dikhao unpaid invoice") == "hinglish"
