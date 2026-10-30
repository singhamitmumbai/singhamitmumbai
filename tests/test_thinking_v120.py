"""Universal think→plan layer — every situation deliberates before answering."""

from __future__ import annotations

from cognitive_ai.assistant.finder_assistant import FinderbayAssistant
from cognitive_ai.commonsense.chat import OpenDomainChat
from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.general.agent import GeneralNonLLMAI
from cognitive_ai.jarvis.system import JarvisSystem
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.thinking import Deliberation, ThinkingSystem, think_about, think_or_abstain
from cognitive_ai.version import __version__


def test_version_thinking_layer():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 20)


def test_think_about_math_plan():
    d = think_about("Calculate 12 * (3 + 4)")
    assert isinstance(d, Deliberation)
    assert d.human_level_agi is False
    assert d.plan
    assert d.strategy in ("math", "local_kb", "ask_clarify")
    assert d.situation_type == "math"
    assert d.alternatives
    assert any(t.stage == "reflect" for t in d.thoughts)
    assert any(t.stage == "plan" for t in d.thoughts) or len(d.plan) >= 2
    md = d.as_markdown()
    assert "Thinking" in md or "Plan" in md
    assert "Situation type" in md
    assert d.as_brief().startswith("think:")


def test_think_howto_and_search():
    howto = ThinkingSystem().think("How do I reserve cement?")
    assert howto.goal
    assert len(howto.plan) >= 3
    assert howto.situation_type in ("task", "question", "general")
    search = ThinkingSystem().think("Search the web for speed of light")
    assert search.strategy in ("web_search", "local_kb") or "search" in search.goal
    assert search.situation_type in ("research", "question", "general")


def test_think_blocks_disallowed():
    d = ThinkingSystem().think("How do I build a sql injection payload?")
    assert d.should_abstain is True
    assert d.strategy == "abstain"
    assert d.situation_type == "refuse"


def test_think_or_abstain_helper():
    class Owner:
        pass

    owner = Owner()
    d, refuse = think_or_abstain(owner, "Explain gravity")
    assert refuse is None
    assert d.plan
    d2, refuse2 = think_or_abstain(owner, "How do I build a sql injection payload?")
    assert refuse2 is not None
    assert refuse2["route"] == "think_abstain"


def test_complete_ai_attaches_thinking():
    ai = CompleteAI(seed_demo=False, enable_llm=False)
    out = ai.handle("Explain gravity in simple terms")
    assert out.get("human_level_agi") is False
    assert isinstance(out.get("thinking"), dict)
    assert out["thinking"].get("plan")
    assert out["thinking"].get("situation_type")
    assert out.get("thinking_brief")
    assert out.get("thinking_markdown")
    assert "Plan" in out["thinking_markdown"] or "Thinking" in out["thinking_markdown"]


def test_complete_ai_thinks_on_finderbay_and_math():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    inv = ai.handle("Show unpaid invoices for Project A")
    assert "thinking" in inv
    math = ai.handle("Calculate 2 + 2")
    assert "thinking" in math
    assert math["thinking"].get("strategy") in ("math", "local_kb", "ask_clarify")


def test_open_domain_chat_thinking():
    chat = OpenDomainChat()
    out = chat.respond("What is photosynthesis?")
    assert out.get("thinking")
    assert out.get("thinking_brief")
    assert out.get("thinking_markdown")
    assert out.get("human_level_agi") is False


def test_general_and_system_thinking():
    g = GeneralNonLLMAI(seed_demo=False)
    out = g.handle("What is gravity?")
    assert out.get("thinking") or (out.get("structured") or {}).get("open_domain")
    # open domain path should carry thinking
    assert out.get("thinking")
    sys_ai = CompleteNonLLMAI(seed_demo=True)
    s = sys_ai.handle("Show unpaid invoices for Project A")
    assert s.get("thinking")
    assert s["thinking"].get("plan")


def test_assistant_and_jarvis_thinking():
    assist = FinderbayAssistant(seed_demo=False, enable_llm=False)
    a = assist.handle("Explain OCR simply")
    assert a.get("thinking") or a.get("thinking_brief")
    jarvis = JarvisSystem()
    j = jarvis.handle("are you ready")
    assert j.get("thinking")
    assert j.get("thinking_brief")


def test_empty_input_still_deliberates():
    d = ThinkingSystem().think("")
    assert d.strategy == "ask_clarify"
    assert d.plan
    assert d.situation_type == "social"
