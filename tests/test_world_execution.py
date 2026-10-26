from cognitive_ai.domains.finderbay.seed import seed_memory
from cognitive_ai.world.model import WorldModel
from cognitive_ai.execution.executor import Executor
from cognitive_ai.execution.verifier import Verifier
from cognitive_ai.reasoning.engine import ReasoningEngine
from cognitive_ai.core.records import PlanStepRecord, ActionStatus
from cognitive_ai.core.working_state import WorkingState


def test_reserve_stock_transition():
    world = WorldModel()
    state = {"available": 100, "reserved": 10, "authorized": True}
    out = world.predict(state, "reserve_stock", {"quantity": 15})
    assert out["state"]["available"] == 85
    assert out["state"]["reserved"] == 25


def test_reserve_execution_and_verify():
    memory = seed_memory()
    ex = Executor(memory, ReasoningEngine(memory), WorldModel(), Verifier(memory))
    state = WorkingState()
    step = PlanStepRecord(
        operator="reserve_stock",
        args={"item": "mat_cement", "quantity": 10},
        verification=["inventory_reservation_receipt"],
    )
    result = ex.execute_step(step, state, client_attempt_id="attempt_1")
    assert result.status == ActionStatus.CONFIRMED
    mats = memory.get_records("material")
    cement = next(m for m in mats if m["id"] == "mat_cement")
    assert cement["available"] == 190
    assert cement["reserved"] == 50
    # Duplicate attempt suppressed
    again = ex.execute_step(step, state, client_attempt_id="attempt_1")
    assert again.attempt_id == "attempt_1"
    cement2 = next(m for m in memory.get_records("material") if m["id"] == "mat_cement")
    assert cement2["available"] == 190
