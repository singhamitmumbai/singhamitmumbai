"""v1.22 full-code upgrade — every pillar + full suite runner."""

from __future__ import annotations

from cognitive_ai.agi_gap.deep_transfer import DOMAIN_SCHEMAS, DeepTransferEngine
from cognitive_ai.benchmarks.scan_like import run_scan_benchmark
from cognitive_ai.benchmarks.transfer_bench import run_transfer_benchmark
from cognitive_ai.commonsense.kb import CommonSenseKB
from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.domains.finderbay.auth import AuthService
from cognitive_ai.memory.store import MemorySystem, Procedure
from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.perception.model_registry import list_models
from cognitive_ai.perception.ocr_layout import LayoutOCR
from cognitive_ai.perception.speech_production import hindi_speech, english_speech
from cognitive_ai.runtime.full_suite import run_full_suite
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.version import __version__


def test_version_full_upgrade():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 22)


def test_kb_expanded_rules_and_tech():
    kb = CommonSenseKB()
    s = kb.summary()
    assert s["facts"] >= 2000
    assert s["rules"] >= 15
    assert kb.ask_fact("python", "is_a", "software")
    assert kb.ask_fact("invoice", "is_a", "business_document")
    assert kb.ask_fact("complete_ai", "does_not_claim", "human_level_agi")


def test_auth_multi_user_directory():
    auth = AuthService()
    users = auth.list_users()
    assert len(users) >= 5
    token = auth.login("mgr1", "mgr-pass")
    assert token
    info = auth.session_info(token)
    assert info and "payment_write" in info["permissions"]
    assert auth.active_session_count() >= 1
    assert auth.logout(token)


def test_payments_and_contracts_via_system():
    ai = CompleteNonLLMAI(seed_demo=True)
    ai.handle("login acct1 password acct-pass")
    pays = ai.handle("Show payments for Project A")
    assert (pays.get("frame") or {}).get("operation") in ("query_payments", "query_invoices")
    assert pays.get("thinking")
    contracts = ai.handle("Show contracts for Project A")
    assert (contracts.get("frame") or {}).get("operation") == "query_contracts"


def test_ocr_layout_analyze():
    ocr = LayoutOCR()
    doc = ocr.recognize_document(
        {
            "pages": [
                {
                    "text": "Item    Qty\nCement  5",
                    "fields": {"invoice_id": "X1"},
                    "tables": [[["Item", "Qty"], ["Cement", "5"]]],
                },
                {"text": "Totals", "fields": {"total": 1}},
            ]
        }
    )
    layout = ocr.analyze_layout(doc)
    assert layout["page_count"] == 2
    assert layout["has_multi_page"] is True
    assert layout["table_count"] >= 1


def test_speech_factories_and_registry():
    models = list_models()
    assert "hi-small" in models and "en-small" in models
    hi = hindi_speech()
    en = english_speech()
    assert hi.model_name == "hi-small"
    assert en.model_name == "en-small"


def test_scan_and_transfer_expanded():
    scan = run_scan_benchmark(LanguageUnderstanding().parse)
    assert scan["by_split"]["train_like"]["passed"] >= 1
    # at least some holdout coverage exists
    assert scan["by_split"]["compositional_holdout"]["passed"] + scan["by_split"]["compositional_holdout"]["failed"] >= 5
    xfer = run_transfer_benchmark()
    assert xfer["passed"] is True


def test_deep_transfer_new_domains():
    assert "retail" in DOMAIN_SCHEMAS
    assert "farm" in DOMAIN_SCHEMAS
    assert "construction" in DOMAIN_SCHEMAS
    mem = MemorySystem()
    mem.add_procedure(
        Procedure(
            name="retail_checkout",
            steps=["cashier scans sku", "checkout at store", "complete sale"],
            applicability=["retail"],
            version="1",
        )
    )
    eng = DeepTransferEngine(mem)
    res = eng.transfer_domain("retail_checkout", from_domain="retail", to_domain="farm")
    assert res.success


def test_complete_ai_full_test_method():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    # diagnostics must stay green
    diag = ai.diagnostics()
    assert diag["ok"], diag
    # thinking + kb checks present
    names = {c["name"] for c in diag["checks"]}
    assert "thinking_system" in names
    assert "commonsense_kb" in names


def test_run_full_suite():
    report = run_full_suite(allow_live_search=False)
    assert report["version"] == __version__
    assert report["human_level_agi"] is False
    assert report["total"] >= 20
    # Allow soft fails on optional engines but core must pass
    critical = {
        "completeness",
        "thinking_math",
        "thinking_explain",
        "open_domain_penguins",
        "commonsense_kb_size",
        "transfer_benchmark",
        "alignment_blocks_offense",
        "ui_static_assets",
        "assistant_thinking",
        "jarvis_ready",
    }
    by_name = {r["name"]: r for r in report["results"]}
    missing = [n for n in critical if n not in by_name]
    assert not missing, missing
    failed_critical = [n for n in critical if not by_name[n].get("ok")]
    assert not failed_critical, failed_critical
    # Overall should pass; if not, surface failures for debugging
    assert report["ok"], report.get("failed")
