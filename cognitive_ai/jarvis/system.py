"""Complete JARVIS system — full non-LLM controller facade (blueprint §§1–14)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..version import __version__
from .adapters import FinderbayAdapters
from .dialogue import DialogueManager
from .documents import DocumentToolkit
from .eval_suite import run_jarvis_eval
from .jobs import JobStore
from .learning import LearningPipeline
from .nlu import IntentClassifier
from .pipeline import JarvisPipeline, PipelineResult
from .policy import AuthContext, PolicyGuard
from .provenance import ProvenanceMemory
from .registry import JarvisToolRegistry
from .skills import SkillRegistry
from .sqlite_memory import SqliteMemory
from .voice import VoicePipeline


class JarvisSystem:
    """End-to-end JARVIS for Finderbay: perception → NLU → memory → plan → act → verify."""

    PRODUCT_NAME = "Finderbay JARVIS"
    HUMAN_LEVEL_AGI = False

    def __init__(
        self,
        *,
        finderbay: Any = None,
        memory_path: str | Path | None = None,
        report_dir: str | Path = "data/jarvis/reports",
        learning_dir: str | Path | None = "data/jarvis/learning",
        auth: Optional[AuthContext] = None,
    ) -> None:
        self.version = __version__
        self.finderbay = finderbay
        self.tools = JarvisToolRegistry()
        self.jobs = JobStore()
        self.memory = ProvenanceMemory()
        self.sqlite = SqliteMemory(memory_path or ":memory:")
        self.docs = DocumentToolkit(report_dir)
        self.policy = PolicyGuard(auth or AuthContext())
        self.adapters = FinderbayAdapters(finderbay=finderbay, docs=self.docs, policy=self.policy)
        self.adapters.bind_registry(self.tools)
        self.skills = SkillRegistry(self.tools)
        self.nlu = IntentClassifier()
        self.dialogue = DialogueManager()
        self.learning = LearningPipeline(learning_dir)
        self.voice = VoicePipeline()
        self.pipeline = JarvisPipeline(
            tools=self.tools,
            jobs=self.jobs,
            memory=self.memory,
            skills=self.skills,
        )
        # Seed procedural memory
        self.memory.register_procedure(
            "materials.shortage_report",
            [
                "resolve site/date",
                "read demand and stock",
                "normalize units",
                "calculate shortages",
                "generate report",
                "verify output",
            ],
        )
        self.sqlite.index_document(
            "blueprint_shortage",
            "Material shortage procedure",
            "Resolve site date demand stock calculate shortages generate verify report",
        )

    def ask(self, text: str, **kwargs: Any) -> str:
        return self.handle(text, **kwargs)["text"]

    def handle(
        self,
        text: str,
        *,
        source: str = "text",
        attempt_id: Optional[str] = None,
        demand: Optional[list] = None,
        stock: Optional[list] = None,
        audio: Any = None,
        speak: bool = False,
    ) -> dict[str, Any]:
        if audio is not None:
            heard = self.adapters.audio_transcribe(audio)
            text = heard.get("text") or text
            source = "voice"
            if not text:
                return self._tag(
                    {
                        "text": "I could not transcribe audio. Use push-to-talk with text fallback or install ASR.",
                        "structured": heard,
                    },
                    route="voice_failed",
                )

        # Follow-up resolution
        follow = self.dialogue.resolve_followup(text)
        # Prefer learning-active classifier for intent when available
        nlu = self.learning.active_classifier.predict(text)
        # Sync dialogue site into pipeline working memory
        if self.dialogue.state.active_project:
            self.pipeline.memory.set_working(active_project=self.dialogue.state.active_project)
            self.sqlite.set_working("active_project", self.dialogue.state.active_project)

        result: PipelineResult = self.pipeline.run(
            text,
            source=source,
            attempt_id=attempt_id,
            demand=demand,
            stock=stock,
        )
        # If classifier disagrees and pipeline unsupported, retry with classifier intent via slots
        if result.intent == "unsupported" and nlu.supported:
            # Force a supported path by rewriting lightly
            if nlu.intent == "system.health":
                result = self.pipeline.run("are you ready", source=source, attempt_id=attempt_id)

        self.dialogue.update_from_result(result.intent, result.slots, missing=result.missing)
        if result.slots.get("site"):
            self.sqlite.set_working("active_project", result.slots["site"])
            self.memory.set_working(active_project=result.slots["site"])
        # Persist episodic + semantic breadcrumbs
        self.sqlite.record_episode(
            f"{result.intent}:{result.verified}",
            {"job_id": result.job_id, "response": result.response[:200]},
        )
        if result.verified and result.slots.get("site"):
            from .contracts import FactRecord

            self.sqlite.record_fact(
                FactRecord(
                    subject=str(result.slots["site"]),
                    relation="last_verified_intent",
                    value=result.intent,
                    verification_status="observed",
                    source_id=result.job_id or "pipeline",
                    kind="episodic",
                )
            )

        out = self._tag(
            {
                "text": result.response,
                "structured": result.to_dict(),
                "nlu": nlu.to_dict(),
                "followup": follow,
                "job_id": result.job_id,
                "verified": result.verified,
                "intent": result.intent,
                "dialogue": self.dialogue.snapshot(),
            },
            route="jarvis",
        )
        if speak and out.get("text"):
            out["voice"] = self.voice.speak(str(out["text"])[:500])
        return out

    def correct_intent(self, text: str, wrong: str, right: str) -> dict[str, Any]:
        ex = self.learning.collect_correction(text, wrong, right, kind="intent")
        self.learning.train_offline()
        return {"ok": True, "example": ex.to_dict(), "learning": self.learning.status()}

    def eval(self) -> dict[str, Any]:
        report = run_jarvis_eval(self.pipeline)
        report["system"] = self.status()
        return report

    def status(self) -> dict[str, Any]:
        return {
            "name": self.PRODUCT_NAME,
            "version": self.version,
            "human_level_agi": self.HUMAN_LEVEL_AGI,
            "controller": "non_llm",
            "tools": len(self.tools.contracts),
            "tool_names": sorted(self.tools.contracts.keys()),
            "skills": self.skills.list_skills(),
            "jobs": self.jobs.summary(),
            "sqlite": self.sqlite.stats(),
            "provenance": self.memory.snapshot(),
            "dialogue": self.dialogue.snapshot(),
            "policy": self.policy.status(),
            "learning": self.learning.status(),
            "voice": self.voice.status(),
            "llm_required": False,
        }

    def greeting(self) -> str:
        return (
            f"{self.PRODUCT_NAME} v{self.version}\n"
            f"Non-LLM controller · {len(self.tools.contracts)} tools · "
            f"skills={len(self.skills.skills)} · human_level_agi=false\n"
            "Pipeline: capture→interpret→resolve→plan→execute→verify→respond→record\n"
            "Try: Prepare tomorrow's material shortage report for Site-7\n"
            "     kal subah 10 baje tender review yaad dilana\n"
            "     Stop this task\n"
        )

    def _tag(self, out: dict[str, Any], *, route: str) -> dict[str, Any]:
        result = dict(out)
        result.setdefault("text", "")
        result["route"] = route
        result["product"] = self.PRODUCT_NAME
        result["version"] = self.version
        result["human_level_agi"] = False
        result["llm_at_runtime"] = False
        result["controller"] = "non_llm"
        return result
