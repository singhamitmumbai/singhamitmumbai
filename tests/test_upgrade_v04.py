"""Upgrade coverage: BM25, tools, tasks, reserve, documents, runtime."""

from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.memory.retrieval import BM25Index
from cognitive_ai.execution.tools import ToolRegistry
from cognitive_ai.nlu.entities import EntityLinker


def test_bm25_finds_invoice():
    ai = CompleteNonLLMAI(seed_demo=True)
    hits = ai.memory.search("unpaid invoice project_a")
    assert hits
    assert any(h.get("kind") == "invoice" for h in hits)


def test_entity_linker_project_and_material():
    ai = CompleteNonLLMAI(seed_demo=True)
    linker = EntityLinker(ai.memory)
    linked = linker.link("reserve cement for Project A from Central Store")
    types = {e.entity_type for e in linked}
    assert "project" in types
    assert "material" in types


def test_tool_registry_validates_args():
    reg = ToolRegistry()
    tool = reg.get("reserve_stock")
    assert tool is not None
    assert tool.validate_args({"item": "x"})  # missing quantity
    assert tool.validate_args({"item": "x", "quantity": 1}) == []


def test_reserve_and_task_and_document():
    ai = CompleteNonLLMAI(seed_demo=True)
    r1 = ai.handle("Reserve 15 bags of cement")
    assert r1["structured"]["reserved"] is True
    assert r1["structured"]["qty_reserved"] == 15.0
    assert r1["structured"]["available"] == 185

    r2 = ai.handle("Create task for Project A: Order safety helmets")
    assert r2["structured"]["created"] is True
    assert "Order safety helmets" in r2["text"]

    r3 = ai.handle("What is the amount due on the invoice scan document?")
    assert r3["frame"]["operation"] == "document_qa"
    assert r3["structured"]["value"] == 118000.0


def test_full_eval_suite_v04():
    ai = CompleteNonLLMAI(seed_demo=True)
    report = ai.evaluate()
    assert report.failed == 0, report.to_dict()
    assert report.passed >= 8


def test_dependency_invalidation():
    ai = CompleteNonLLMAI(seed_demo=True)
    from cognitive_ai.core.records import Claim, EvidenceStatus

    src = ai.memory.remember_claim(
        Claim("inv_42", "payment_status", "unpaid", evidence_status=EvidenceStatus.OBSERVED)
    )
    derived = ai.memory.remember_claim(
        Claim("inv_42", "overdue", True, evidence_status=EvidenceStatus.INFERRED),
        depends_on=[src.claim_id],
    )
    invalidated = ai.memory.invalidate_dependents(src.claim_id)
    assert derived.claim_id in invalidated
    assert ai.memory.claims[derived.claim_id].evidence_status == EvidenceStatus.UNKNOWN


def test_runtime_monitor_records():
    ai = CompleteNonLLMAI(seed_demo=True)
    ai.handle("Show material stock and inventory")
    summary = ai.monitor.summary()
    assert summary["requests"] >= 1
    assert summary["p50_latency_ms"] is not None


def test_capability_model_speech_abstain():
    ai = CompleteNonLLMAI(seed_demo=True)
    out = ai.handle(b"\x00\x01", modality="audio")
    assert "acoustic" in out["text"].lower() or "speech" in out["text"].lower()
