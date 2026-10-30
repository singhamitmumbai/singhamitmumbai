"""v0.8: alerts, POs/vendors, clarification, causal CS, API/reports."""

import json
import threading
from urllib.request import Request, urlopen

from cognitive_ai.commonsense.reasoner import CommonSenseReasoner
from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.reporting.exporter import ReportExporter
from cognitive_ai.runtime.alerts import ProactiveMonitor
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.api.server import serve_api
from cognitive_ai.general.agent import GeneralNonLLMAI


def test_proactive_alerts_overdue_and_pending():
    ai = CompleteNonLLMAI(seed_demo=True)
    summary = ai.alerts.summary(project_id="project_a")
    assert summary["count"] >= 1
    kinds = {a["kind"] for a in summary["alerts"]}
    assert "overdue_invoice" in kinds or "pending_delivery" in kinds or "missing_receipt" in kinds
    r = ai.handle("Show alerts for Project A")
    assert r["frame"]["operation"] == "query_alerts"
    assert r["structured"]["count"] >= 1


def test_vendors_and_purchase_orders():
    ai = CompleteNonLLMAI(seed_demo=True)
    v = ai.handle("Show vendors")
    assert v["frame"]["operation"] == "query_vendors"
    assert v["structured"]["count"] >= 2
    po = ai.handle("Show purchase orders for Project A")
    assert po["frame"]["operation"] == "query_purchase_orders"
    assert po["structured"]["count"] >= 1


def test_create_po_requires_auth_and_works():
    ai = CompleteNonLLMAI(seed_demo=True)
    ai.handle("login acct1 password acct-pass")
    out = ai.handle("create po cement qty 10 @ 500 vendor buildco for Project A")
    assert out["frame"]["operation"] == "create_purchase_order"
    assert out["structured"].get("po_created") is True
    assert out["structured"]["purchase_order"]["total_inr"] == 5000.0


def test_slot_fill_clarification_for_followup():
    ai = CompleteNonLLMAI(seed_demo=True)
    r1 = ai.handle("prepare a follow-up. Do not send it.")
    assert r1["frame"]["needs_clarification"]
    assert "project" in r1["text"].lower()
    r2 = ai.handle("Project A")
    assert r2["frame"]["operation"] == "prepare_followup"
    assert not r2["frame"]["needs_clarification"]
    assert r2["structured"].get("draft") or "follow-up" in (r2["text"] or "").lower() or r2.get("faithful")


def test_export_report_csv():
    ai = CompleteNonLLMAI(seed_demo=True)
    r = ai.handle("export report invoices for Project A")
    assert r["frame"]["operation"] == "export_report"
    assert r["structured"]["report_ready"] is True
    assert r["structured"]["format"] == "csv"
    assert "invoice" in (r["structured"].get("csv") or "").lower() or r["structured"]["count"] >= 1
    # direct exporter
    exp = ReportExporter(ai.memory, ai.alerts)
    payload = exp.export("alerts", fmt="json")
    assert payload["count"] >= 1


def test_causal_why_reasoning():
    r = CommonSenseReasoner()
    ans = r.why("fire")
    assert ans.success
    assert "cause" in ans.text.lower() or "because" in ans.text.lower() or "needs" in ans.text.lower()
    g = GeneralNonLLMAI(seed_demo=True)
    out = g.handle("Why fire?")
    assert out["route"] == "open_domain"
    assert "fire" in out["text"].lower()


def test_rest_api_health_and_ask():
    httpd = serve_api("127.0.0.1", 8799, mode="general")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen("http://127.0.0.1:8799/health", timeout=5) as resp:
            health = json.loads(resp.read().decode())
        assert health["ok"] is True
        assert health["human_level_agi"] is False
        req = Request(
            "http://127.0.0.1:8799/ask",
            data=json.dumps({"text": "Can penguins fly?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
        assert "cannot" in body["text"].lower() or "no" in body["text"].lower()
        with urlopen("http://127.0.0.1:8799/alerts", timeout=5) as resp:
            alerts = json.loads(resp.read().decode())
        assert alerts["count"] >= 1
    finally:
        httpd.shutdown()


def test_human_level_agi_still_false():
    assert GeneralNonLLMAI(seed_demo=True).status()["human_level_agi"] is False


def test_intent_alerts_and_po():
    nlu = LanguageUnderstanding()
    assert nlu.parse("what needs attention").operation == "query_alerts"
    assert nlu.parse("show purchase orders").operation == "query_purchase_orders"
