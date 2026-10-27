"""General non-LLM AI: Finderbay tasks + open-domain common-sense chat + perception."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..commonsense.chat import OpenDomainChat
from ..commonsense.kb import CommonSenseKB
from ..commonsense.reasoner import CommonSenseReasoner
from ..config import SystemConfig
from ..nlu.intents import rank_intents
from ..system import CompleteNonLLMAI
from ..version import __version__
from .transfer import SkillTransfer


FINDERBAY_HINTS = {
    "query_invoices",
    "query_deliveries",
    "query_materials",
    "reserve_stock",
    "manage_task",
    "document_qa",
    "prepare_followup",
    "learn_alias",
    "query_payments",
    "record_payment",
    "query_contracts",
    "login",
    "query_alerts",
    "query_purchase_orders",
    "create_purchase_order",
    "query_vendors",
    "export_report",
    "detect_objects",
    "detect_vehicles",
    "count_people",
    "detect_fire",
    "scan_document_vision",
    "vision_status",
    "ppe_compliance",
    "estimate_queue",
    "status",
    "cancel",
    "correct",
}


class GeneralNonLLMAI:
    """Broader assistant: routes between task AI and open-domain chat.

    Still not human-level AGI — but supports open conversation over curated
    common-sense knowledge plus full Finderbay task execution, with production
    speech/OCR engines when available.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        storage: Optional[Path | str] = None,
        seed_demo: bool = True,
    ) -> None:
        self.version = __version__
        self.task_ai = CompleteNonLLMAI(config=config, storage=storage, seed_demo=seed_demo)
        self.cs_kb = CommonSenseKB()
        self.cs_reasoner = CommonSenseReasoner(self.cs_kb)
        self.chat = OpenDomainChat(self.cs_reasoner)
        self.transfer = SkillTransfer(self.task_ai.memory)

    def handle(
        self,
        payload: Any,
        *,
        modality: str = "text",
        source: str = "user",
        context: Optional[dict] = None,
        force: Optional[str] = None,  # finderbay | open_domain | auto
    ) -> dict[str, Any]:
        force = force or "auto"
        text = payload if isinstance(payload, str) else ""
        if modality == "audio":
            # Always go through task AI gateway first for ASR, then route on transcript
            prelim = self.task_ai.gateway.ingest(payload, modality="audio", source=source, context=context)
            text = prelim.extracted_value if isinstance(prelim.extracted_value, str) else ""
            if not text:
                return {
                    "text": prelim.context.get("speech_meta", {}).get("error")
                    or "Speech engine produced no transcript.",
                    "route": "speech_failed",
                    "observation": prelim.to_dict(),
                    "llm_dependency": False,
                    "version": self.version,
                }
            modality = "text"
            payload = text

        if force == "finderbay" or (force == "auto" and self._looks_like_task(text)):
            result = self.task_ai.handle(payload, modality=modality, source=source, context=context)
            # If task AI abstains as unsupported, fall through to open-domain
            if (
                force == "auto"
                and (result.get("frame") or {}).get("operation") == "unsupported"
            ):
                od = self.chat.respond(text)
                return self._wrap_open(od, fallback_from="finderbay_unsupported")
            result = dict(result)
            result["route"] = "finderbay"
            result["general_ai"] = True
            return result

        od = self.chat.respond(text if text else str(payload))
        return self._wrap_open(od)

    def ask(self, text: str) -> str:
        return self.handle(text)["text"]

    def _looks_like_task(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        ranked = rank_intents(text, top_k=3)
        if not ranked:
            return False
        top = ranked[0]
        if top.intent in FINDERBAY_HINTS and top.score >= 1.5:
            return True
        # strong lexical hooks
        lower = text.lower()
        hooks = (
            "invoice",
            "delivery",
            "follow-up",
            "followup",
            "reserve",
            "project a",
            "project b",
            "amount due",
            "create task",
            "list tasks",
            "warehouse",
            "how many",
            "purchase order",
            "alert",
            "payment",
            "contract",
            "vendor",
            "detect objects",
            "detect vehicles",
            "count people",
            "detect fire",
            "vision status",
            "yolo",
            "ppe compliance",
            "computer vision",
        )
        return any(h in lower for h in hooks)

    def _wrap_open(self, od: dict[str, Any], fallback_from: str = "") -> dict[str, Any]:
        return {
            "text": od.get("text"),
            "structured": {
                "open_domain": True,
                "method": od.get("method"),
                "triples": od.get("triples"),
                "topic": od.get("topic"),
                "fallback_from": fallback_from or None,
            },
            "faithful": True,
            "frame": {
                "operation": "open_domain_chat",
                "confidence": od.get("confidence", 0.0),
                "supported": od.get("supported", True),
            },
            "route": "open_domain",
            "general_ai": True,
            "llm_dependency": False,
            "version": self.version,
            "commonsense": self.cs_kb.summary(),
        }

    def status(self) -> dict[str, Any]:
        from ..perception.ocr_production import ProductionOCR
        from ..perception.speech_production import ProductionSpeech

        return {
            "name": "GeneralNonLLMAI",
            "version": self.version,
            "llm_at_runtime": False,
            "human_level_agi": False,
            "open_domain_chat": True,
            "commonsense": self.cs_kb.summary(),
            "finderbay": self.task_ai.status(),
            "speech": {
                "engine": "vosk",
                "available": ProductionSpeech().available,
            },
            "ocr": {
                "engine": "tesseract",
                "available": ProductionOCR().available,
            },
            "vision": self.task_ai.vision.status(),
            "limits": [
                "Not human-level AGI",
                "Open-domain chat is curated common-sense + patterns, not unrestricted LLM chat",
                "Common-sense coverage is finite and explicit",
                "Vision uses YOLO/classical CV (optional Ultralytics) — not an LLM",
            ],
        }

    def transfer_skill(self, procedure_name: str, from_project: str, to_project: str, scale: float = 1.0):
        return self.transfer.transfer(
            procedure_name,
            from_project=from_project,
            to_project=to_project,
            quantity_scale=scale,
        )
