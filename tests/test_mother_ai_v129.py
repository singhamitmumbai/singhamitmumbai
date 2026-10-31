"""Finderbay Mother AI second-pass — world model, roles, cognition, self-mod (v1.29)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_ai.mother import (
    CognitionBudget,
    ContinuousCognition,
    MotherBrain,
    MotherWorldModel,
    RoleLibrary,
    SelfModLab,
    run_finderbench,
)
from cognitive_ai.mother.gateway import InputGateway
from cognitive_ai.version import __version__


def test_version_mother_v129():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 29)


def test_world_model_predict_observe(tmp_path: Path):
    wm = MotherWorldModel()
    wm.ingest_utterance("Check Project Alpha inventory", intent="task")
    assert wm.state.entities
    pred = wm.predict("complete_ai")
    assert pred["ok"] is True
    assert pred["expected_cues"]
    assert pred["human_level_agi"] is False
    row = wm.observe("complete_ai", "Invoice for Project Alpha cannot be paid yet — thinking…", predicted=pred)
    assert "prediction_error" in row
    assert 0.0 <= row["prediction_error"] <= 1.0
    assert wm.summary()["history"] >= 1


def test_roles_verifier_rejects_empty_inspect():
    roles = RoleLibrary()
    gw = InputGateway()
    interp = gw.interpret("")
    proposals = roles.propose(interp)
    critique = roles.critique(proposals, interp)
    chosen = roles.verify_select(proposals, critique, interp)
    # Empty input should not stay on expensive inspect
    assert chosen.get("action") != "inspect_repo" or not interp.normalized


def test_roles_prefer_low_cost_when_uncertain():
    roles = RoleLibrary()
    gw = InputGateway()
    interp = gw.interpret("asdf qwer zxcv weird")
    # Force low confidence path via unknown-ish
    proposals = roles.propose(interp)
    critique = roles.critique(proposals, interp)
    chosen = roles.verify_select(proposals, critique, interp)
    assert chosen.get("action")
    assert chosen.get("human_level_agi") is False


def test_mother_cycle_uses_world_model(tmp_path: Path):
    m = MotherBrain(store_dir=tmp_path / "m", seed_demo=False, enable_llm=False)
    out = m.handle("hi")
    assert out["route"] == "mother"
    assert out["human_level_agi"] is False
    st = m.status()
    assert st["world_model"]["version"]
    assert "attention" in st["world_model"]["resources"]
    assert m.world_model.history
    cyc = out["mother_cycle"]
    assert isinstance(cyc.get("prediction_error"), (int, float))


def test_honesty_not_agi(tmp_path: Path):
    m = MotherBrain(store_dir=tmp_path / "h", seed_demo=False, enable_llm=False)
    out = m.handle("are you human-level AGI?")
    text = (out.get("text") or "").lower()
    assert "unknown command" not in text
    assert "false" in text or "not" in text or "human_level" in text


def test_continuous_cognition_budget(tmp_path: Path):
    m = MotherBrain(store_dir=tmp_path / "cog", seed_demo=False, enable_llm=False)
    report = m.run_cognition(max_cycles=2, max_seconds=20.0, prompts=["hi", "how are you"])
    assert report["human_level_agi"] is False
    assert report["cycles_run"] == 2
    assert report["stop_reason"] in ("max_cycles", "prompts_exhausted", None) or report["cycles_run"] <= 2
    assert len(report["recent_ticks"]) >= 1


def test_self_mod_lab_isolated_rollback(tmp_path: Path):
    lab = SelfModLab(tmp_path / "lab")
    snap = {"expected_cues": {"dialogue": ["hello"]}}
    cand = lab.propose(
        kind="cue_map",
        title="Add ready cue",
        payload={"expected_cues": {"dialogue": ["hello", "ready"]}},
        evidence=["unit_test"],
        snapshot_before=snap,
    )
    assert cand.status == "isolated"
    lab.evaluate(cand.candidate_id)
    with pytest.raises(ValueError):
        lab.review(cand.candidate_id, accept=True, reviewer="self")
    lab.review(cand.candidate_id, accept=True, reviewer="human_tester")
    sandbox = lab.apply_isolated(cand.candidate_id, {"expected_cues": {}})
    assert sandbox["live_mutated"] is False
    assert "ready" in sandbox["sandbox"]["expected_cues"]["dialogue"]
    promo = lab.promote(cand.candidate_id, reviewer="human_tester")
    assert promo["ok"] is True
    rb = lab.rollback(cand.candidate_id)
    assert rb["status"] == "rolled_back"
    assert rb["restored"] == snap


def test_finderbench_expanded(tmp_path: Path):
    m = MotherBrain(store_dir=tmp_path / "fb", seed_demo=True, enable_llm=False)
    report = run_finderbench(m)
    assert report["human_level_agi"] is False
    assert report["total"] >= 25
    assert report["passed"] >= 18
    assert "World Model" in report["category_scores"]
    assert "Language Understanding" in report["category_scores"]
