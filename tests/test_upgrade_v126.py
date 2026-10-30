"""v1.26 system upgrade — audit, lifelong persist, NLU composition, OCR export."""

from __future__ import annotations

from pathlib import Path

from cognitive_ai.domains.finderbay.audit import AuditLog
from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.perception.ocr_layout import DocumentResult, LayoutOCR, PageResult
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.version import __version__


def test_version_v126():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 26)


def test_audit_log_login_and_query(tmp_path: Path):
    ai = CompleteNonLLMAI(seed_demo=True)
    ai.audit = AuditLog(tmp_path / "audit.jsonl")
    ai.auth.bind_audit(ai.audit)
    ai.executor.audit = ai.audit
    out = ai.handle("login acct1 password acct-pass")
    assert out.get("structured", {}).get("logged_in") or "success" in (out.get("text") or "").lower()
    rows = ai.audit.recent(limit=20)
    assert any(r.get("action") == "login" and r.get("ok") for r in rows)
    audit = ai.handle("Show audit log")
    assert (audit.get("frame") or {}).get("operation") == "query_audit_log"
    assert "audit" in (audit.get("text") or "").lower()


def test_nlu_between_and_under_not_overdue_bug():
    lu = LanguageUnderstanding()
    between = lu.parse("Find paid invoices for Project C between ₹1 lakh and ₹3 lakh")
    d = between.to_dict()
    amts = [c for c in d["conditions"] if c["field"] == "amount_inr"]
    assert d["operation"] == "query_invoices"
    assert d["project_id"] == "project_c"
    assert {c["operator"] for c in amts} >= {"gte", "lte"}

    under = lu.parse(
        "Show overdue unpaid invoices for Project A under ₹50000 excluding cancelled"
    ).to_dict()
    amount_ops = [c for c in under["conditions"] if c["field"] == "amount_inr"]
    assert len(amount_ops) == 1
    assert amount_ops[0]["operator"] == "lt"
    assert amount_ops[0]["value"] == 50000.0
    assert any(c["field"] == "status" and c["operator"] == "ne" for c in under["conditions"])


def test_nlu_stock_qty_exclusion():
    lu = LanguageUnderstanding()
    d = lu.parse("List materials below 20 units for Project A excluding scrap").to_dict()
    assert d["operation"] == "query_materials"
    assert any(c["field"] == "stock_qty" and c["value"] == 20.0 for c in d["conditions"])
    assert any(c["field"] == "material_tag" and c["operator"] == "ne" for c in d["conditions"])


def test_lifelong_auto_persist(tmp_path: Path):
    from cognitive_ai.agi_gap.stack import ProtoAGIStack

    path = tmp_path / "life.json"
    s1 = ProtoAGIStack(persist_path=path, auto_persist=True)
    s1.lifelong.remember_interaction("hello upgrade", actions=[{"op": "x"}], outcome="success")
    s1.persist_lifelong()
    assert path.exists()
    s2 = ProtoAGIStack(persist_path=path, auto_persist=True)
    assert s2.status()["auto_persist"] is True
    assert s2.lifelong.summary().get("episodes", 0) >= 1


def test_ocr_invoice_export():
    doc = DocumentResult(
        pages=[
            PageResult(
                0,
                "Invoice",
                {"invoice_number": "INV-1", "vendor": "V", "amount_due": "2500.50"},
                tables=[[["Item", "Amt"], ["A", "10"]]],
            )
        ],
        fields={"invoice_number": "INV-1", "vendor": "V", "amount_due": "2500.50"},
        tables=[[["Item", "Amt"], ["A", "10"]]],
        text="Invoice",
        page_count=1,
    )
    out = LayoutOCR().export_invoice_fields(doc)
    assert out["ok"] is True
    assert out["invoice_number"] == "INV-1"
    assert out["amount_due"] == 2500.50
    assert len(out["line_items"]) == 1
    assert out["human_level_agi"] is False


def test_scan_holdout_includes_v126_cases():
    from cognitive_ai.benchmarks.scan_like import CASES, run_scan_benchmark
    from cognitive_ai.nlu.pipeline import LanguageUnderstanding

    ids = {c.case_id for c in CASES}
    assert {"ch9", "ch10", "ch11", "tl6"} <= ids
    report = run_scan_benchmark(LanguageUnderstanding().parse)
    assert report["by_split"]["train_like"]["accuracy"] >= 0.8
    # Holdout should improve vs weak composition; require majority pass
    assert report["compositional_generalization"] >= 0.5
