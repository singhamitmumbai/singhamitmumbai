"""Demo transactional data for Finderbay evaluation."""

from __future__ import annotations

from cognitive_ai.memory.store import MemorySystem
from cognitive_ai.memory.ontology import finderbay_ontology


def seed_memory(memory: MemorySystem | None = None) -> MemorySystem:
    memory = memory or MemorySystem(ontology=finderbay_ontology())

    # Projects
    memory.upsert_record("project", "project_a", {"name": "Project A", "status": "active"})
    memory.upsert_record("project", "project_b", {"name": "Project B", "status": "active"})

    # Invoices
    invoices = [
        {
            "id": "inv_42",
            "project_id": "project_a",
            "amount_inr": 150000,
            "payment_status": "unpaid",
            "dispute_status": "not_disputed",
            "due_date": "2024-01-01",
            "source": "register_v3",
            "status_as_of": "2026-03-01T00:00:00+00:00",
        },
        {
            "id": "inv_43",
            "project_id": "project_a",
            "amount_inr": 250000,
            "payment_status": "unpaid",
            "dispute_status": "not_disputed",
            "due_date": "2026-12-31",
            "source": "register_v3",
            "status_as_of": "2026-03-01T00:00:00+00:00",
        },
        {
            "id": "inv_44",
            "project_id": "project_a",
            "amount_inr": 500000,
            "payment_status": "unpaid",
            "dispute_status": "disputed",
            "due_date": "2025-06-01",
            "source": "register_v3",
            "status_as_of": "2026-03-01T00:00:00+00:00",
        },
        {
            "id": "inv_45",
            "project_id": "project_a",
            "amount_inr": 800000,
            "payment_status": "unpaid",
            "dispute_status": None,  # unknown — must not be treated as not_disputed
            "due_date": "2025-01-01",
            "source": "register_v3",
            "status_as_of": "2026-03-01T00:00:00+00:00",
        },
        {
            "id": "inv_50",
            "project_id": "project_b",
            "amount_inr": 1200000,
            "payment_status": "unpaid",
            "dispute_status": "not_disputed",
            "due_date": "2025-01-01",
            "source": "register_v3",
            "status_as_of": "2026-03-01T00:00:00+00:00",
        },
        {
            "id": "inv_51",
            "project_id": "project_a",
            "amount_inr": 90000,
            "payment_status": "paid",
            "dispute_status": "not_disputed",
            "due_date": "2025-01-01",
            "source": "register_v3",
            "status_as_of": "2026-03-01T00:00:00+00:00",
        },
    ]
    for inv in invoices:
        memory.upsert_record("invoice", inv["id"], inv)

    # Deliveries for this week — Project A
    deliveries = [
        {
            "id": "del_1",
            "project_id": "project_a",
            "item": "Cement bags",
            "expected_qty": 100,
            "received_qty": 60,
            "receipt_status": "known",
            "time_window": "this_week",
            "week": "this_week",
            "source": "delivery_register",
        },
        {
            "id": "del_2",
            "project_id": "project_a",
            "item": "Steel rods",
            "expected_qty": 50,
            "received_qty": 50,
            "receipt_status": "known",
            "time_window": "this_week",
            "week": "this_week",
            "source": "delivery_register",
        },
        {
            "id": "del_3",
            "project_id": "project_a",
            "item": "Sand (cu.m)",
            "expected_qty": 20,
            "received_qty": None,
            "receipt_status": "missing",
            "time_window": "this_week",
            "week": "this_week",
            "source": "delivery_register",
        },
        {
            "id": "del_4",
            "project_id": "project_b",
            "item": "Paint",
            "expected_qty": 10,
            "received_qty": 0,
            "receipt_status": "known",
            "time_window": "this_week",
            "week": "this_week",
            "source": "delivery_register",
        },
    ]
    for d in deliveries:
        memory.upsert_record("delivery", d["id"], d)

    # Materials
    memory.upsert_record(
        "material",
        "mat_cement",
        {"name": "Cement", "available": 200, "reserved": 40, "unit": "bags", "warehouse": "central_store"},
    )
    memory.learn_alias("Central Store", "central_store", source="seed")
    memory.learn_alias("Warehouse B", "central_store", source="seed")

    # Tasks
    memory.upsert_record(
        "task",
        "task_1",
        {"project_id": "project_a", "title": "Site inspection", "status": "open"},
    )

    # Sample invoice scan / document for OCR QA
    memory.upsert_record(
        "document",
        "doc_inv_scan_1",
        {
            "project_id": "project_a",
            "page_id": "page_1",
            "text": "INVOICE 99\nVendor: BuildCo\nSubtotal 100000\nTax 18000\nAmount Due 118000",
            "fields": {
                "invoice_number": "99",
                "vendor": "BuildCo",
                "subtotal": "100000",
                "tax": "18000",
                "amount_due": "118000",
            },
            "tables": [],
            "regions": [{"id": "r1", "text": "Amount Due 118000"}],
            "source": "scan_register",
        },
    )
    # Ambiguous handwritten-style amount for capability messaging tests
    memory.upsert_record(
        "document",
        "doc_inv_scan_2",
        {
            "project_id": "project_a",
            "page_id": "page_2",
            "text": "Amount Due twenty-five?",
            "fields": {"amount_due": "twenty-five?"},
            "tables": [],
            "regions": [],
            "source": "handwritten_note",
        },
    )
    return memory


def load_finderbay_demo() -> MemorySystem:
    return seed_memory()
