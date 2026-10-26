"""v0.6: open-domain chat, common-sense, production perception, transfer."""

from pathlib import Path

from cognitive_ai.general.agent import GeneralNonLLMAI
from cognitive_ai.commonsense.kb import CommonSenseKB
from cognitive_ai.commonsense.reasoner import CommonSenseReasoner
from cognitive_ai.perception.ocr_production import ProductionOCR
from cognitive_ai.perception.speech_production import ProductionSpeech
from cognitive_ai.memory.store import Procedure


def test_commonsense_penguin_cannot_fly():
    r = CommonSenseReasoner()
    ans = r.reason("penguin", "can", "fly")
    # exception path: cannot fly
    assert ans.success
    assert "cannot" in ans.text.lower() or "no" in ans.text.lower()


def test_commonsense_inheritance_dog_mammal():
    r = CommonSenseReasoner()
    # seed has dog is_a mammal directly
    ans = r.reason("dog", "is_a", "mammal")
    assert ans.success


def test_analogy_knife_umbrella():
    r = CommonSenseReasoner()
    ans = r.analogy("knife", "cutting", "umbrella")
    assert ans.success
    assert "rain" in ans.text.lower() or "protection" in ans.text.lower()


def test_open_domain_and_finderbay_routing():
    ai = GeneralNonLLMAI(seed_demo=True)
    od = ai.handle("Can penguins fly?")
    assert od["route"] == "open_domain"
    assert "penguin" in od["text"].lower() or "cannot" in od["text"].lower()

    task = ai.handle(
        "Show unpaid invoices above ₹1 lakh for Project A, excluding disputed ones."
    )
    assert task["route"] == "finderbay"
    assert task["structured"]["count"] == 2

    meta = ai.handle("Are you ChatGPT?")
    assert "llm" in meta["text"].lower() or "not" in meta["text"].lower()


def test_production_ocr_available_and_fields():
    ocr = ProductionOCR()
    assert ocr.available is True
    out = ocr.recognize(
        {
            "text": "Invoice No: 77\nAmount Due: 125000",
            "fields": {"invoice_number": "77", "amount_due": "125000"},
        }
    )
    assert out["fields"]["amount_due"] == 125000.0


def test_production_speech_vosk_available():
    sp = ProductionSpeech()
    assert sp.available is True
    # oracle path
    out = sp.recognize({"transcript": "show unpaid invoices for project a"})
    assert "unpaid" in out["transcript"]


def test_skill_transfer():
    ai = GeneralNonLLMAI(seed_demo=True)
    ai.task_ai.memory.add_procedure(
        Procedure(
            name="followup_pending",
            steps=["retrieve project_a deliveries", "draft qty=10 followup", "do_not_send"],
            applicability=["project:project_a"],
        )
    )
    result = ai.transfer_skill("followup_pending", "project_a", "project_b", scale=2.0)
    assert result.success
    assert any("project_b" in s for s in result.mapped_steps)
    assert any("qty=20" in s for s in result.mapped_steps)


def test_general_status_honesty():
    st = GeneralNonLLMAI(seed_demo=True).status()
    assert st["human_level_agi"] is False
    assert st["open_domain_chat"] is True
    assert st["llm_at_runtime"] is False
    assert st["speech"]["available"] is True
    assert st["ocr"]["available"] is True
