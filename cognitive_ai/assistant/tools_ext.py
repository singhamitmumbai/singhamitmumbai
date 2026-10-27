"""Register assistant tools on Complete AI tool router."""

from __future__ import annotations

from typing import Any


def attach_assistant_tools(complete_ai, assistant) -> None:
    """Extend CompleteAI.tools with voice/memory/computer/learning/plan."""

    def voice(args: dict[str, Any]) -> dict[str, Any]:
        action = (args.get("action") or "speak").lower()
        if action == "speak":
            return assistant.voice.speak(str(args.get("text") or ""))
        if action == "listen":
            return assistant.voice.listen(args.get("audio") or args.get("transcript") or "")
        return assistant.voice.status()

    def memory(args: dict[str, Any]) -> dict[str, Any]:
        action = (args.get("action") or "search").lower()
        if action == "remember":
            return assistant.memory.remember_preference(str(args.get("key")), str(args.get("value") or ""))
        if action == "recall":
            return {"ok": True, "value": assistant.memory.recall(str(args.get("key") or ""))}
        if action == "search":
            return {"ok": True, "hits": assistant.memory.search(str(args.get("query") or ""))}
        return {"ok": True, **assistant.memory.snapshot()}

    def computer(args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "status")
        payload = {k: v for k, v in args.items() if k != "action"}
        return assistant.computer.run(action, **payload)

    def learn(args: dict[str, Any]) -> dict[str, Any]:
        action = (args.get("action") or "status").lower()
        if action == "alias":
            return assistant.learning.learn_alias(str(args.get("former")), str(args.get("alias")))
        if action == "correct":
            return assistant.learning.record_correction(
                input_text=str(args.get("input") or ""),
                wrong=str(args.get("wrong") or ""),
                right=str(args.get("right") or ""),
                task=str(args.get("task") or "general"),
            )
        return assistant.learning.status()

    def plan(args: dict[str, Any]) -> dict[str, Any]:
        return assistant.plan(str(args.get("text") or args.get("goal") or ""))

    complete_ai.tools.handlers.update(
        {
            "voice": voice,
            "memory": memory,
            "computer": computer,
            "learn": learn,
            "plan": plan,
        }
    )
