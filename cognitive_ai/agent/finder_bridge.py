"""Bridge: Finderbay AI issues tasks → SelfBrain executes them autonomously."""

from __future__ import annotations

from typing import Any, Optional

from ..coding.system import CodingSystem
from ..system import CompleteNonLLMAI
from .scheduler import AgentScheduler
from .self_brain import SelfBrainAgent, AgentReport


class FinderbayAgentSystem:
    """Full stack: Finderbay AI + SelfBrain automation + CodingSystem."""

    def __init__(self, *, seed_demo: bool = True, workspace: Optional[str] = None) -> None:
        self.finderbay = CompleteNonLLMAI(seed_demo=seed_demo)
        self.coding = CodingSystem(root=workspace)
        self.brain = SelfBrainAgent(ai=self.finderbay, coding=self.coding, seed_demo=seed_demo)
        self.scheduler = AgentScheduler(self.brain)

    def ask_finderbay(self, text: str) -> dict[str, Any]:
        """Interactive Finderbay query (single-shot, not queued)."""
        return self.finderbay.handle(text)

    def assign(self, instruction: str, **kwargs) -> dict[str, Any]:
        """Finderbay AI assigns a task to SelfBrain."""
        job = self.brain.submit_from_finderbay(instruction, **kwargs)
        return job.to_dict()

    def assign_batch(self, instructions: list[str]) -> list[dict[str, Any]]:
        return [j.to_dict() for j in self.brain.submit_many(instructions, source="finderbay")]

    def execute_queue(self, **kwargs) -> dict[str, Any]:
        report = self.brain.run(**kwargs)
        return report.to_dict()

    def auto(self, instructions: list[str], **kwargs) -> dict[str, Any]:
        """Assign then immediately execute a batch of Finderbay tasks."""
        self.assign_batch(instructions)
        return self.execute_queue(**kwargs)

    def code(self, instruction: str, **metadata) -> dict[str, Any]:
        job = self.brain.submit(instruction, tags=["coding"], metadata=metadata, source="finderbay+coding")
        report = self.brain.run(max_jobs=1)
        return {"job": job.to_dict(), "report": report.to_dict()}

    def start_daemon(self) -> None:
        self.scheduler.start()

    def stop_daemon(self) -> None:
        self.scheduler.stop()

    def status(self) -> dict[str, Any]:
        return {
            "name": "FinderbayAgentSystem",
            "finderbay": self.finderbay.status(),
            "self_brain": self.brain.status(),
            "coding": self.coding.status(),
            "scheduler": self.scheduler.status(),
            "llm_at_runtime": False,
            "human_level_agi": False,
        }
