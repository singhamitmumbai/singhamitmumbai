"""Finderbay Assistant — voice, memory, vision, planning, computer, learning."""

from __future__ import annotations

from cognitive_ai.assistant import FinderbayAssistant
from cognitive_ai.assistant.computer import ComputerControl
from cognitive_ai.assistant.voice import AssistantVoice


def test_assistant_status_and_greeting():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False, allow_computer_control=False)
    st = a.status()
    assert st["name"] == "Finderbay Assistant"
    assert st["human_level_agi"] is False
    assert "voice" in st and "memory" in st and "vision" in st
    assert "planning" in st and "computer_control" in st and "learning" in st
    assert "Finderbay Assistant" in a.greeting()


def test_memory_remember_recall():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    out = a.handle("remember that my warehouse is Central Store")
    assert out["route"] == "memory"
    out2 = a.handle("recall warehouse")
    assert out2["route"] == "memory"
    assert "Central Store" in out2["text"] or "warehouse" in out2["text"].lower()


def test_planning_preview():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    out = a.handle("plan Show unpaid invoices for Project A")
    assert out["route"] == "planning"
    assert out["structured"]["steps"]
    assert a.last_plan is not None


def test_vision_via_assistant():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    out = a.handle("Count people in image ./crowd_people.jpg")
    assert "person" in out["text"].lower() or "Counted" in out["text"]
    assert "vision" in out.get("capabilities_used", []) or out.get("route") in ("vision", "finderbay", "general")


def test_computer_control_gated():
    cc = ComputerControl(allow=False)
    r = cc.run("open_url", url="https://example.com")
    assert r["ok"] is False
    assert r["error"] == "computer_control_disabled"
    cc.enable()
    r2 = cc.run("status")
    assert r2["ok"] is True
    r3 = cc.run("notify", message="hi")
    assert r3["ok"] is True


def test_voice_speak_stub():
    v = AssistantVoice()
    out = v.speak("Hello Finderbay")
    assert out["ok"] is True
    assert out["text"] == "Hello Finderbay"
    heard = v.listen({"transcript": "show unpaid invoices for project a", "uncertainty": 0.1})
    assert "invoice" in heard["transcript"]


def test_learning_correction_and_alias():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    out = a.handle("correction: show bills => invoices wrong => unpaid invoices")
    assert out["route"] == "learning"
    out2 = a.handle("Central Store is also called Warehouse B")
    assert out2["route"] == "learning"


def test_assistant_tools_registered_on_complete():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    assert "voice" in a.complete.tools.handlers
    assert "computer" in a.complete.tools.handlers
    assert "plan" in a.complete.tools.handlers
    res = a.complete.tools.run("plan", {"text": "Show alerts for Project A"})
    assert res.get("route") == "planning" or res.get("structured")
