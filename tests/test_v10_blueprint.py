"""v1.0: blueprint-aligned operating cycle, conflicts, cases, improvement, audit."""

from cognitive_ai.blueprint.audit import run_blueprint_audit
from cognitive_ai.core.records import LearningExample
from cognitive_ai.learning_sys.improvement import ImprovementLoop
from cognitive_ai.learning_sys.induction import ProgramInducer
from cognitive_ai.reasoning.cases import CaseBasedReasoner
from cognitive_ai.reasoning.conflicts import ConflictDetector
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.general.agent import GeneralNonLLMAI
from cognitive_ai.memory.store import Episode


def test_operating_cycle_trace_present():
    ai = CompleteNonLLMAI(seed_demo=True)
    r = ai.handle("Show unpaid invoices for Project A")
    assert r["cycle"] is not None
    stages = r["cycle"]["stage_names"]
    assert "perceive" in stages
    assert "understand" in stages
    assert "attend" in stages
    assert "plan" in stages
    assert "execute" in stages
    assert "respond" in stages
    assert r["cycle"]["llm_at_runtime"] is False
    assert r["human_level_agi"] is False
    assert r["distinctions"]["unknown_vs_false"] is True


def test_conflict_detector_invoice_vs_payments():
    ai = CompleteNonLLMAI(seed_demo=True)
    # inv_51 is paid with payment — force a conflict by marking unpaid while payment exists
    inv = ai.memory.get_records("invoice", id="inv_51")[0]
    ai.memory.upsert_record("invoice", "inv_51", {**inv, "payment_status": "unpaid"})
    reports = ConflictDetector().scan_memory_claims(ai.memory)
    assert any(r.subject == "inv_51" for r in reports)
    r = ai.handle("Show unpaid invoices for Project A")
    assert r.get("conflicts") is not None


def test_case_based_retrieval_after_episode():
    ai = CompleteNonLLMAI(seed_demo=True)
    ai.handle("Show unpaid invoices for Project A")
    # episode should exist
    assert ai.memory.episodes
    matches = CaseBasedReasoner().retrieve(ai.memory, "unpaid invoices Project A", operation="query_invoices")
    assert matches
    assert matches[0].score > 0


def test_improvement_loop_records_unsupported():
    loop = ImprovementLoop()
    fail = loop.record_failure(operation="fly_to_moon", input_text="fly me to the moon", reason="unsupported")
    cand = loop.propose_from_failure(fail)
    assert cand.status == "proposed"
    try:
        loop.review_candidate(cand.candidate_id, accept=True, reviewer="model")
        assert False, "should reject unverified reviewer"
    except ValueError:
        pass
    loop.review_candidate(cand.candidate_id, accept=True, reviewer="human")
    rel = loop.release(cand.candidate_id)
    assert rel["candidate_id"] == cand.candidate_id


def test_program_induction_from_examples():
    inducer = ProgramInducer()
    pos = [
        LearningExample(
            task="query_invoices",
            input_data={"operation": "query_invoices"},
            verified_target={"operation": "query_invoices", "steps": ["retrieve_records", "reason_over_records", "respond"]},
            reviewer="suite",
        ),
        LearningExample(
            task="query_invoices",
            input_data={"operation": "query_invoices"},
            verified_target={"operation": "query_invoices", "steps": ["retrieve_records", "reason_over_records", "respond"]},
            reviewer="suite",
        ),
    ]
    prog = inducer.induce("invoice_query_skill", pos)
    assert prog.status == "candidate"
    assert "retrieve_records" in prog.steps


def test_blueprint_audit_coverage():
    report = run_blueprint_audit()
    assert report["human_level_agi"] is False
    assert report["llm_at_runtime"] is False
    assert report["summary"]["present"] >= 30
    assert report["summary"]["coverage_ratio"] >= 0.8


def test_status_exposes_improvement_and_world():
    ai = CompleteNonLLMAI(seed_demo=True)
    st = ai.status()
    assert st["human_level_agi"] is False
    assert "improvement" in st
    assert "reserve_stock" in st["world_transitions"]
    audit = ai.blueprint_audit()
    assert audit["summary"]["total"] >= 30


def test_general_still_honest():
    g = GeneralNonLLMAI(seed_demo=True)
    assert g.status()["human_level_agi"] is False
    r = g.handle("How many unpaid invoices for Project A?")
    assert r["route"] == "finderbay"
    assert r.get("cycle") or r.get("frame")
