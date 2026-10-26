"""v0.5 completion: datasets, calibration, recovery, adapters, Hindi, alias, HTN."""

import json
from pathlib import Path

from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.config import SystemConfig
from cognitive_ai.datasets import DatasetRegistry
from cognitive_ai.domains.finderbay.adapters import load_csv_records, export_json
from cognitive_ai.goals.htn import HTNLibrary
from cognitive_ai.uncertainty.calibration import brier_score, calibrate_report
from cognitive_ai.uncertainty.policies import respond_to_uncertainty
from cognitive_ai.core.records import UncertaintyKind
from cognitive_ai.response.hindi import realize_invoice_hi, prefer_hindi


def test_dataset_split_lock_prevents_leakage(tmp_path: Path):
    reg = DatasetRegistry(tmp_path / "examples")
    reg.make(
        "query_invoices",
        "show unpaid",
        {"count": 1},
        split="train",
        source_group="speaker_a",
    )
    try:
        reg.make(
            "query_invoices",
            "show unpaid again",
            {"count": 1},
            split="test",
            source_group="speaker_a",
        )
        assert False, "expected leakage guard"
    except ValueError as e:
        assert "locked" in str(e).lower()


def test_csv_adapter_and_export(tmp_path: Path):
    ai = CompleteNonLLMAI(seed_demo=True)
    n = load_csv_records(ai.memory, "data/finderbay/sample_invoices.csv", "invoice")
    assert n == 2
    out = export_json(ai.memory, "invoice", tmp_path / "invoices.json")
    payload = json.loads(out.read_text())
    assert payload["kind"] == "invoice"
    assert len(payload["records"]) >= 2


def test_learn_alias_end_to_end():
    ai = CompleteNonLLMAI(seed_demo=True)
    r = ai.handle("Warehouse B is now called Central Store.")
    assert r["frame"]["operation"] == "learn_alias"
    assert r["structured"]["learned"] is True
    assert ai.memory.ontology.resolve("central store")


def test_list_tasks():
    ai = CompleteNonLLMAI(seed_demo=True)
    r = ai.handle("List tasks for Project A")
    assert r["structured"].get("listed") is True
    assert "Site inspection" in r["text"]


def test_hindi_realization():
    assert prefer_hindi("hinglish")
    text = realize_invoice_hi({"count": 2, "total_inr": 400000, "unresolved_count": 1})
    assert "इनवॉइस" in text
    assert "लाख" in text


def test_htn_decompose():
    htn = HTNLibrary()
    steps = htn.decompose("prepare_followup")
    assert "draft_followup" in steps
    assert htn.select("reserve_stock") is not None


def test_uncertainty_policy_and_calibration():
    msg = respond_to_uncertainty(UncertaintyKind.MISSING_INFORMATION)
    assert "missing" in msg.lower() or "request" in msg.lower()
    report = calibrate_report([0.1, 0.9], [0, 1])
    assert report.brier == brier_score([0.1, 0.9], [0, 1])
    assert report.n == 2


def test_config_and_status():
    cfg = SystemConfig(allow_send=False, resource_time_ms=1234)
    ai = CompleteNonLLMAI(config=cfg, seed_demo=True)
    st = ai.status()
    assert st["version"] == "0.5.0"
    assert st["config"]["resource_time_ms"] == 1234
    assert st["datasets"]["total"] >= 1
    assert "query_invoices" in st["htn_tasks"]


def test_full_eval_v05():
    ai = CompleteNonLLMAI(seed_demo=True)
    report = ai.evaluate()
    assert report.failed == 0, report.to_dict()
    assert report.passed >= 11


def test_calibrate_intent_scores():
    ai = CompleteNonLLMAI(seed_demo=True)
    cal = ai.calibrate_intent_scores()
    assert cal["n"] >= 8
    assert "brier" in cal
