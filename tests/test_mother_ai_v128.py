"""Finderbay Mother AI — NL REPL fix + cognitive loop (v1.28)."""

from __future__ import annotations

from pathlib import Path

from cognitive_ai.cli import handle_line
from cognitive_ai.brain import Brain
from cognitive_ai.mother import MotherBrain, run_finderbench
from cognitive_ai.mother.gateway import InputGateway
from cognitive_ai.version import __version__


def test_version_mother_v128():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 28)


def test_gateway_greetings_and_hinglish():
    gw = InputGateway()
    for q in ("hii", "hi", "how are you", "kaise ho", "tum kya kar sakte ho"):
        i = gw.interpret(q)
        assert i.intent in ("conversation", "meta")
        assert "unknown command" not in i.intent


def test_repl_routes_natural_language_to_dialogue(tmp_path: Path, capsys):
    brain = Brain()
    mother = MotherBrain(store_dir=tmp_path / "m", seed_demo=False, enable_llm=False)
    assert handle_line(brain, "hii", mother=mother) is True
    out = capsys.readouterr().out.lower()
    assert "unknown command" not in out
    assert "hello" in out or "ready" in out or "mother" in out

    assert handle_line(brain, "how are you", mother=mother) is True
    out2 = capsys.readouterr().out.lower()
    assert "unknown command" not in out2
    assert "function" in out2 or "ready" in out2 or "help" in out2


def test_mother_cycle_has_stages(tmp_path: Path):
    m = MotherBrain(store_dir=tmp_path / "m2", seed_demo=False, enable_llm=False)
    out = m.handle("hi")
    assert out["route"] == "mother"
    assert out["human_level_agi"] is False
    cyc = out["mother_cycle"]
    stages = [s["stage"] for s in cyc["stages"]]
    assert "understand" in stages
    assert "execute" in stages
    assert "learn" in stages


def test_goals_persist(tmp_path: Path):
    d = tmp_path / "goals"
    m1 = MotherBrain(store_dir=d, seed_demo=False, enable_llm=False)
    m1.handle("inspect this repository and tell me what is broken")
    assert m1.goals.summary()["total"] >= 1
    m2 = MotherBrain(store_dir=d, seed_demo=False, enable_llm=False)
    assert m2.goals.summary()["total"] >= 1


def test_symbolic_commands_still_work(tmp_path: Path, capsys):
    brain = Brain()
    brain.load_domain("animals")
    mother = MotherBrain(store_dir=tmp_path / "m3", seed_demo=False, enable_llm=False)
    assert handle_line(brain, "help", mother=mother) is True
    out = capsys.readouterr().out.lower()
    assert "commands" in out or "assert" in out


def test_finderbench_smoke(tmp_path: Path):
    m = MotherBrain(store_dir=tmp_path / "fb", seed_demo=True, enable_llm=False)
    report = run_finderbench(m)
    assert report["human_level_agi"] is False
    assert report["passed"] >= 10
    assert "Language Understanding" in report["category_scores"]
