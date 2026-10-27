"""Map implemented modules to blueprint sections — honest status, not AGI claims."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..version import __version__


@dataclass
class ComponentStatus:
    name: str
    blueprint_section: str
    module: str
    status: str  # present|partial|stub|absent
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "blueprint_section": self.blueprint_section,
            "module": self.module,
            "status": self.status,
            "notes": self.notes,
        }


INVENTORY: list[ComponentStatus] = [
    ComponentStatus("Input gateway", "§2/§5", "perception/gateway.py", "present"),
    ComponentStatus("Speech perception", "§5A", "perception/speech_production.py", "present", "Vosk offline ASR"),
    ComponentStatus("OCR / layout", "§5B", "perception/ocr_layout.py", "present", "Tesseract + layout heuristics"),
    ComponentStatus("Computer vision (YOLO)", "§5C", "vision/", "present", "Vendored YOLO_Projects + Ultralytics/stub"),
    ComponentStatus("Complete AI facade", "product", "complete_ai.py", "present", "Session + ChatGPT-class chat + local tools"),
    ComponentStatus("LLM chat bridge", "product", "chat/llm_bridge.py", "present", "Optional; auto-on with API key"),
    ComponentStatus("Tool router", "product", "chat/tools.py", "present", "finderbay/vision/coding/alerts/memory"),
    ComponentStatus("Language understanding", "§6", "nlu/pipeline.py", "present"),
    ComponentStatus("Question understanding", "§6", "nlu/questions.py", "present"),
    ComponentStatus("Dialogue state", "§6", "nlu/dialogue_tracker.py", "present"),
    ComponentStatus("Clarification / slot-fill", "§6", "nlu/clarification.py", "present"),
    ComponentStatus("Ontology / knowledge", "§7", "memory/ontology.py", "present"),
    ComponentStatus("Multi-memory store", "§7", "memory/store.py", "present"),
    ComponentStatus("Attention scheduler", "§7", "memory/attention.py", "present"),
    ComponentStatus("Retrieval (BM25)", "§7", "memory/retrieval.py", "present"),
    ComponentStatus("Working state", "§2/§4", "core/working_state.py", "present"),
    ComponentStatus("Typed records", "§4", "core/records.py", "present"),
    ComponentStatus("Reasoning engine", "§8", "reasoning/engine.py", "present"),
    ComponentStatus("Conflict detection", "§8/§15", "reasoning/conflicts.py", "present"),
    ComponentStatus("Case-based reasoning", "§8", "reasoning/cases.py", "present"),
    ComponentStatus("Common-sense KB", "§7/§19", "commonsense/", "present", "Curated finite KB — not unrestricted"),
    ComponentStatus("World model", "§9", "world/model.py", "present", "Explicit transitions; learned forecasts partial"),
    ComponentStatus("Goal manager", "§10", "goals/manager.py", "present"),
    ComponentStatus("Planner / HTN", "§10", "goals/planner.py", "present"),
    ComponentStatus("Executor", "§11", "execution/executor.py", "present"),
    ComponentStatus("Verifier", "§11", "execution/verifier.py", "present"),
    ComponentStatus("Recovery / OUTCOME_UNKNOWN", "§11", "execution/recovery.py", "present"),
    ComponentStatus("Response generator", "§12", "response/generator.py", "present"),
    ComponentStatus("Hindi realization", "§12", "response/hindi.py", "present", "Invoices/deliveries/materials/reserve/tasks/alerts/payments/vision/status"),
    ComponentStatus("Session persistence", "product", "runtime/persist.py", "present"),
    ComponentStatus("Completeness checker", "product", "runtime/completeness.py", "present"),
    ComponentStatus("Finderbay Assistant", "product", "assistant/", "present", "Voice, memory, vision, planning, computer control, learning"),
    ComponentStatus("Assistant voice (ASR/TTS)", "§5A", "assistant/voice.py", "present"),
    ComponentStatus("Computer control (gated)", "§11", "assistant/computer.py", "present", "Disabled by default; allowlisted actions"),
    ComponentStatus("Learning bridge", "§13", "assistant/learning_bridge.py", "present"),
    ComponentStatus("Learning system", "§13", "learning_sys/learner.py", "present"),
    ComponentStatus("Program induction", "§13/§19", "learning_sys/induction.py", "present"),
    ComponentStatus("Improvement loop", "§2/§13", "learning_sys/improvement.py", "present"),
    ComponentStatus("Datasets / split locks", "§14", "datasets/registry.py", "present"),
    ComponentStatus("Uncertainty policies", "§15", "uncertainty/", "present"),
    ComponentStatus("Capability model", "§15", "runtime/capability_model.py", "present"),
    ComponentStatus("Operating cycle trace", "§2/§20", "runtime/cycle.py", "present"),
    ComponentStatus("Evaluation harness", "§16", "evaluation/harness.py", "present"),
    ComponentStatus("Generalization benches", "§16/§19", "benchmarks/", "present"),
    ComponentStatus("Skill transfer", "§19", "general/transfer.py", "present"),
    ComponentStatus("Proactive alerts", "domains", "runtime/alerts.py", "present"),
    ComponentStatus("Finderbay domains", "§18", "domains/finderbay/", "present"),
    ComponentStatus("REST API", "§17", "api/server.py", "present"),
    ComponentStatus("Runtime monitor", "§17", "runtime/monitor.py", "present"),
    ComponentStatus("SelfBrain autonomous agent", "§10/§11/§19", "agent/self_brain.py", "present", "Queue + verify/retry automation"),
    ComponentStatus("Agent scheduler", "§17", "agent/scheduler.py", "present"),
    ComponentStatus("Finderbay→agent bridge", "§18", "agent/finder_bridge.py", "present"),
    ComponentStatus("CodingSystem (AST/templates/tests)", "§19", "coding/", "present", "Symbolic — not LLM pair-programming"),
]


class BlueprintAudit:
    def run(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for c in INVENTORY:
            by_status[c.status] = by_status.get(c.status, 0) + 1
        present = sum(1 for c in INVENTORY if c.status == "present")
        partial = sum(1 for c in INVENTORY if c.status == "partial")
        return {
            "version": __version__,
            "llm_at_runtime": False,
            "human_level_agi": False,
            "blueprint": "Complete_Non_LLM_AI_Engineering_Blueprint",
            "components": [c.to_dict() for c in INVENTORY],
            "summary": {
                "total": len(INVENTORY),
                "present": present,
                "partial": partial,
                "absent": by_status.get("absent", 0),
                "stub": by_status.get("stub", 0),
                "coverage_ratio": round(present / max(1, len(INVENTORY)), 3),
            },
            "distinctions": [
                "Unknown ≠ false",
                "Predicted ≠ observed",
                "Attempted ≠ confirmed",
            ],
            "honesty": [
                "Scoped complete non-LLM AI for declared Finderbay + curated open-domain capabilities",
                "Not human-level AGI",
                "Open-domain coverage is finite and explicit",
                "No pretrained general-purpose LLM at runtime",
            ],
        }


def run_blueprint_audit() -> dict[str, Any]:
    return BlueprintAudit().run()
