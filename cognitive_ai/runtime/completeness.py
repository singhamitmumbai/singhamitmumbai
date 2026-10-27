"""Completeness audit — verify the full Complete AI code surface is wired."""

from __future__ import annotations

import importlib
from typing import Any

from ..version import __version__


REQUIRED_MODULES = [
    "cognitive_ai.complete_ai",
    "cognitive_ai.system",
    "cognitive_ai.general.agent",
    "cognitive_ai.chat.llm_bridge",
    "cognitive_ai.chat.tools",
    "cognitive_ai.vision.system",
    "cognitive_ai.coding.system",
    "cognitive_ai.agent.self_brain",
    "cognitive_ai.api.server",
    "cognitive_ai.ui.server",
    "cognitive_ai.nlu.pipeline",
    "cognitive_ai.commonsense.chat",
    "cognitive_ai.runtime.persist",
    "cognitive_ai.response.hindi",
]


REQUIRED_ATTRS = {
    "cognitive_ai.complete_ai": ["CompleteAI", "create_complete_ai"],
    "cognitive_ai.chat.tools": ["ToolRouter", "extract_tool_calls"],
    "cognitive_ai.chat.llm_bridge": ["LLMChatBridge", "llm_bridge_from_env"],
    "cognitive_ai.vision.system": ["VisionSystem"],
    "cognitive_ai.coding.system": ["CodingSystem"],
}


def run_completeness_check() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for mod_name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            missing = [
                a for a in REQUIRED_ATTRS.get(mod_name, []) if not hasattr(mod, a)
            ]
            results.append(
                {
                    "module": mod_name,
                    "ok": not missing,
                    "missing_attrs": missing,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"module": mod_name, "ok": False, "error": str(exc)})

    # Smoke-construct Complete AI
    try:
        from ..complete_ai import CompleteAI

        ai = CompleteAI(seed_demo=False, enable_llm=False)
        diag = ai.diagnostics()
        results.append(
            {
                "module": "CompleteAI.diagnostics",
                "ok": bool(diag.get("ok")),
                "detail": {"passed": diag.get("passed"), "total": diag.get("total")},
            }
        )
        export = ai.export_session(fmt="markdown")
        results.append(
            {
                "module": "CompleteAI.export_session",
                "ok": isinstance(export, str) and "Complete AI" in export,
            }
        )
    except Exception as exc:  # noqa: BLE001
        results.append({"module": "CompleteAI.smoke", "ok": False, "error": str(exc)})

    passed = sum(1 for r in results if r.get("ok"))
    return {
        "product": "Complete AI",
        "version": __version__,
        "ok": passed == len(results),
        "passed": passed,
        "total": len(results),
        "results": results,
        "human_level_agi": False,
        "complete_code": passed == len(results),
    }
