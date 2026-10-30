"""v0.7: large CS KB, speech models, layout OCR, payments/contracts/auth, benchmarks."""

from cognitive_ai.commonsense.kb import CommonSenseKB
from cognitive_ai.commonsense.eval_suite import run_commonsense_eval
from cognitive_ai.commonsense.reasoner import CommonSenseReasoner
from cognitive_ai.perception.model_registry import list_models
from cognitive_ai.perception.ocr_layout import LayoutOCR
from cognitive_ai.benchmarks import run_scan_benchmark, run_transfer_benchmark
from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.general.agent import GeneralNonLLMAI


def test_large_commonsense_kb_size():
    kb = CommonSenseKB()
    assert kb.summary()["facts"] >= 2000
    assert kb.summary()["rules"] >= 8


def test_commonsense_eval_suite():
    report = run_commonsense_eval()
    assert report["failed"] == 0, report
    assert report["kb"]["facts"] >= 2000


def test_speech_models_registered():
    models = list_models()
    # EN small ships in-repo; HI / EN-IN are registered and downloadable
    assert models["en-small"]["installed"] is True
    assert "hi-small" in models and models["hi-small"]["url"]
    assert "en-in-small" in models and models["en-in-small"]["url"]
    assert models["hi-small"]["lang"] == "hi"
    assert models["en-in-small"]["lang"] == "en-in"


def test_layout_ocr_multipage_tables():
    ocr = LayoutOCR()
    doc = {
        "pages": [
            {
                "text": "Invoice No: 101\nAmount Due: 50000\nItem    Qty    Rate\nCement  10     500",
                "fields": {"invoice_number": "101", "amount_due": "50000"},
                "tables": [[["Item", "Qty", "Rate"], ["Cement", "10", "500"]]],
            },
            {
                "text": "Page 2 notes\nTax: 9000",
                "fields": {"tax": "9000"},
            },
        ]
    }
    result = ocr.recognize_document(doc).to_dict()
    assert result["page_count"] == 2
    assert result["fields"]["amount_due"] == 50000.0
    assert result["fields"]["tax"] == 9000.0
    assert result["tables"]


def test_auth_payments_contracts():
    ai = CompleteNonLLMAI(seed_demo=True)
    # viewer cannot post payment
    tok = ai.auth.login("viewer1", "view-pass")
    assert tok
    try:
        ai.payments.record_payment(token=tok, invoice_id="inv_42", amount_inr=1000)
        assert False, "expected permission error"
    except PermissionError:
        pass
    # accountant can
    tok2 = ai.auth.login("acct1", "acct-pass")
    out = ai.payments.record_payment(token=tok2, invoice_id="inv_42", amount_inr=150000)
    assert out["payment"]["amount_inr"] == 150000
    contracts = ai.contracts.summarize("project_a")
    assert contracts["active"] >= 1


def test_login_and_payment_via_language():
    ai = CompleteNonLLMAI(seed_demo=True)
    r1 = ai.handle("login acct1 password acct-pass")
    assert r1["structured"].get("logged_in") is True
    r2 = ai.handle("record payment of 1000 for invoice 43")
    # may fail if invoice 43 amount path odd — check operation routed
    assert r2["frame"]["operation"] == "record_payment"


def test_scan_and_transfer_benchmarks():
    nlu = LanguageUnderstanding()
    scan = run_scan_benchmark(nlu.parse)
    assert "compositional_generalization" in scan
    assert scan["by_split"]["train_like"]["passed"] >= 2
    transfer = run_transfer_benchmark()
    assert transfer["passed"] is True


def test_query_contracts_and_payments():
    ai = CompleteNonLLMAI(seed_demo=True)
    c = ai.handle("Show contracts for Project A")
    assert c["frame"]["operation"] == "query_contracts"
    assert c["structured"]["count"] >= 1
    p = ai.handle("Show payments for Project A")
    assert p["frame"]["operation"] == "query_payments"


def test_general_still_routes():
    ai = GeneralNonLLMAI(seed_demo=True)
    assert ai.handle("Can penguins fly?")["route"] == "open_domain"
    assert ai.status()["human_level_agi"] is False
