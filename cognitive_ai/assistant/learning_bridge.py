"""Learning bridge — reviewed examples, corrections, and skill candidates."""

from __future__ import annotations

from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import LearningExample


class LearningBridge:
    def __init__(self, finderbay_ai) -> None:
        self.ai = finderbay_ai
        self.corrections: list[dict[str, Any]] = []

    def learn_alias(self, former: str, alias: str) -> dict[str, Any]:
        out = self.ai.handle(f"Remember that {former} is also called {alias}")
        claim = self.ai.learner.learn_fact(
            subject=former.lower().replace(" ", "_"),
            relationship="alias",
            value=alias,
            source="assistant",
            reviewer="user",
        )
        return {"ok": True, "claim": claim.to_dict(), "route_text": out.get("text")}

    def record_correction(
        self,
        *,
        input_text: str,
        wrong: str,
        right: str,
        task: str = "general",
    ) -> dict[str, Any]:
        ex = LearningExample(
            example_id=new_id("ex"),
            task=task,
            input_data={"text": input_text, "wrong": wrong},
            verified_target=right,
            reviewer="user",
            notes=f"assistant_correction wrong={wrong!r}",
            source_group="assistant_corrections",
        )
        path = self.ai.learner.add_example(ex)
        row = {"input": input_text, "wrong": wrong, "right": right, "path": str(path)}
        self.corrections.append(row)
        try:
            self.ai.improvement.record_failure(
                operation=task,
                input_text=input_text,
                reason=f"corrected:{wrong}->{right}",
                stage="assistant",
            )
        except Exception:
            pass
        return {"ok": True, **row}

    def propose_skill(
        self,
        name: str,
        steps: list[str],
        *,
        examples: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        pos = [
            LearningExample(
                example_id=new_id("ex"),
                task=name,
                input_data=e.get("input"),
                verified_target=e.get("target"),
                reviewer="user",
            )
            for e in (examples or [])
        ]
        return self.ai.learner.propose_procedure(name, steps, pos, [])

    def status(self) -> dict[str, Any]:
        return {
            "name": "LearningBridge",
            "corrections": len(self.corrections),
            "candidates": len(getattr(self.ai.learner, "candidates", []) or []),
            "releases": len(getattr(self.ai.learner, "releases", []) or []),
            "policy": "reviewed_examples_only",
        }
