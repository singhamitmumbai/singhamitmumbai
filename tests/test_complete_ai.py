"""Complete AI facade + optional LLM bridge (disabled by default)."""

from __future__ import annotations

from cognitive_ai.chat.llm_bridge import LLMChatBridge
from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.commonsense.chat import OpenDomainChat


def test_complete_ai_identity_and_status():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    st = ai.status()
    assert st["name"] == "Complete AI"
    assert st["human_level_agi"] is False
    assert st["llm_at_runtime"] is False
    assert st["version"].startswith("1.3")
    assert "vision" in st["modes"]
    g = ai.greeting()
    assert "Complete AI" in g


def test_complete_ai_routes_tasks_and_chat():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    inv = ai.handle("Show unpaid invoices for Project A")
    assert inv["route"] in ("finderbay", "general")
    assert inv["human_level_agi"] is False
    assert inv["llm_at_runtime"] is False

    who = ai.handle("Who are you?")
    assert "Complete AI" in who["text"] or "complete" in who["text"].lower()

    peng = ai.handle("Can penguins fly?")
    assert peng["llm_at_runtime"] is False
    assert "penguin" in peng["text"].lower() or "fly" in peng["text"].lower()


def test_complete_ai_vision_and_coding_hooks():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    vis = ai.handle("Count people in image ./crowd_people.jpg")
    assert vis["route"] in ("vision", "finderbay", "general")
    assert "person" in vis["text"].lower() or "Counted" in vis["text"]

    code = ai.handle("generate function add(a, b)")
    assert code["route"] == "coding"
    assert code.get("ok") or "def add" in str(code.get("code") or code.get("text") or "")


def test_llm_bridge_disabled_by_default():
    bridge = LLMChatBridge(enabled=False, api_key=None)
    assert bridge.available is False
    out = bridge.chat("Hello")
    assert out["ok"] is False
    assert out["llm_at_runtime"] is False
    assert out["human_level_agi"] is False


def test_llm_bridge_forced_mode_without_key():
    ai = CompleteAI(seed_demo=False, enable_llm=True, llm=LLMChatBridge(enabled=True, api_key=None))
    out = ai.handle("Invent a poem about Mars", mode="llm")
    assert out["route"] == "llm_forced"
    assert out["llm_at_runtime"] is False  # no key → not actually calling LLM
    assert "API key" in out["text"] or "not enabled" in out["text"].lower() or "LLM" in out["text"]


def test_open_domain_explain_compare_name():
    chat = OpenDomainChat()
    r = chat.respond("My name is Alex")
    assert "Alex" in r["text"]
    r2 = chat.respond("hello")
    assert "Alex" in r2["text"]
    r3 = chat.respond("explain penguin")
    assert r3["method"] in ("explain", "explain_miss", "describe", "define") or r3["supported"]
