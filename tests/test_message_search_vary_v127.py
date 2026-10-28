"""Message search + message variation (v1.27)."""

from __future__ import annotations

from cognitive_ai.chat.messages import (
    MessageSearch,
    MessageVariation,
    parse_search_command,
    parse_vary_command,
)
from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.version import __version__


def test_version_v127():
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 27)


def test_parse_commands():
    assert parse_search_command("search messages for invoice") == "invoice"
    assert parse_search_command("/search gravity") == "gravity"
    assert parse_vary_command("vary message: Hello world") == "Hello world"
    assert parse_vary_command("paraphrase Show unpaid invoices") == "Show unpaid invoices"
    assert parse_search_command("hello there") is None


def test_message_search_with_variations():
    rows = [
        {"role": "user", "text": "Explain gravity", "route": "auto", "ts": 1},
        {"role": "assistant", "text": "Gravity pulls masses together.", "route": "open_domain", "ts": 2},
        {"role": "user", "text": "Show unpaid invoices for Project A", "route": "auto", "ts": 3},
        {"role": "assistant", "text": "2 invoices matched.", "route": "finderbay", "ts": 4},
    ]
    out = MessageSearch().search(rows, "bill")  # synonym → invoice
    assert out["ok"] is True
    assert out["total_hits"] >= 1
    assert any("invoice" in (h["text"] or "").lower() for h in out["hits"])
    assert len(out["query_variations"]) >= 2
    assert out["human_level_agi"] is False


def test_message_variation_styles():
    out = MessageVariation().vary("Show unpaid invoices for Project A", n=8)
    assert out["ok"] is True
    styles = {v["style"] for v in out["variations"]}
    assert "formal" in styles or "polite" in styles
    assert "hinglish" in styles or "question" in styles
    assert all(v["text"].strip() for v in out["variations"])
    assert out["human_level_agi"] is False


def test_complete_ai_search_and_vary():
    ai = CompleteAI(seed_demo=False, enable_llm=False, autosave=False)
    ai.handle("Explain gravity in simple terms")
    ai.handle("Show unpaid invoices for Project A")
    found = ai.search_messages("gravity")
    assert found["ok"] and found["total_hits"] >= 1
    cmd = ai.handle("search messages for invoice")
    assert cmd["route"] == "message_search"
    assert "invoice" in (cmd.get("text") or "").lower() or (cmd.get("structured") or {}).get("total_hits", 0) >= 1
    varied = ai.handle("vary message: Can you show unpaid bills?")
    assert varied["route"] == "message_variation"
    assert (varied.get("structured") or {}).get("count", 0) >= 1


def test_vary_last_message():
    ai = CompleteAI(seed_demo=False, enable_llm=False, autosave=False)
    ai.handle("Explain gravity")
    out = ai.handle("vary last message")
    assert out["route"] == "message_variation"
    assert (out.get("structured") or {}).get("ok") is True


def test_tool_router_message_tools():
    from cognitive_ai.chat.tools import ToolRouter

    ai = CompleteAI(seed_demo=False, enable_llm=False, autosave=False)
    ai.handle("Show unpaid invoices for Project A")
    tr = ToolRouter(ai)
    s = tr.run("search_messages", {"query": "invoice"})
    assert s.get("ok") is True
    v = tr.run("vary_message", {"text": "Find the bill please", "n": 3})
    assert v.get("ok") is True
    assert len(v.get("variations") or []) >= 1
