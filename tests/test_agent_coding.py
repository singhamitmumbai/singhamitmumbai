"""SelfBrain autonomous agent + non-LLM CodingSystem."""

from pathlib import Path

from cognitive_ai.agent import FinderbayAgentSystem, SelfBrainAgent
from cognitive_ai.coding import CodingSystem
from cognitive_ai.coding.ast_tools import analyze_source, rename_symbol_in_source
from cognitive_ai.coding.generator import generate_function, generate_class, parse_generate_request
from cognitive_ai.system import CompleteNonLLMAI


def test_coding_analyze_and_generate(tmp_path: Path):
    coding = CodingSystem(root=tmp_path)
    out = coding.handle("generate function add(a, b)", metadata={"path": "mathy.py"})
    assert out["ok"] is True
    assert (tmp_path / "mathy.py").exists()
    src = (tmp_path / "mathy.py").read_text()
    assert "def add" in src
    analysis = analyze_source(src)
    assert analysis.ok
    assert any(s.name == "add" for s in analysis.symbols)


def test_coding_rename_and_search(tmp_path: Path):
    coding = CodingSystem(root=tmp_path)
    coding.handle("generate function foo()", metadata={"path": "m.py"})
    renamed = coding.handle("rename symbol foo to bar in m.py")
    assert renamed.get("ok") is True
    src = (tmp_path / "m.py").read_text()
    assert "def bar" in src
    hits = coding.handle("find symbol bar")
    assert hits["ok"]
    assert hits["hits"]


def test_generate_class_and_parse_request():
    code = generate_class("Widget", methods=[{"name": "run", "args": ["self"], "body": "    return 1\n", "returns": "int"}])
    assert "class Widget" in code
    assert "def run" in code
    plan = parse_generate_request("generate function total(x, y)")
    assert plan and plan["name"] == "total"


def test_ast_rename_unparse():
    src = "def old():\n    return old\n"
    out, meta = rename_symbol_in_source(src, "old", "new")
    assert meta["ok"]
    assert "def new" in out


def test_self_brain_executes_finderbay_queue():
    ai = CompleteNonLLMAI(seed_demo=True)
    agent = SelfBrainAgent(ai=ai)
    agent.submit("Show unpaid invoices for Project A", priority=2)
    agent.submit("Show alerts for Project A", priority=1)
    report = agent.run()
    assert report.succeeded >= 1
    assert report.human_level_agi is False
    assert report.llm_at_runtime is False


def test_finder_bridge_auto_batch(tmp_path: Path):
    system = FinderbayAgentSystem(seed_demo=True, workspace=str(tmp_path))
    report = system.auto(
        [
            "Show unpaid invoices for Project A",
            "generate function multiply(x, y)",
        ]
    )
    assert report["succeeded"] >= 1
    # coding job should have written a file under generated/
    gen = list(tmp_path.rglob("*.py"))
    assert gen, "expected generated python file"


def test_coding_system_status_honest():
    st = CodingSystem().status()
    assert st["human_level_agi"] is False
    assert st["llm_at_runtime"] is False
    assert "ast_analyze" in st["capabilities"]


def test_agent_system_status_honest():
    st = FinderbayAgentSystem(seed_demo=True).status()
    assert st["human_level_agi"] is False
    assert st["llm_at_runtime"] is False
