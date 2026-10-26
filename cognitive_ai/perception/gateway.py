"""Input gateway: receive inputs with identity, time and context."""

from __future__ import annotations

from typing import Any, Optional

from ..core.records import Observation
from ..core.working_state import WorkingState
from .ocr import DocumentPipeline
from .speech import SpeechPipeline


class InputGateway:
    def __init__(
        self,
        speech: Optional[SpeechPipeline] = None,
        documents: Optional[DocumentPipeline] = None,
    ) -> None:
        self.speech = speech or SpeechPipeline()
        self.documents = documents or DocumentPipeline()

    def ingest(
        self,
        payload: Any,
        *,
        modality: str = "text",
        source: str = "user",
        context: Optional[dict] = None,
        state: Optional[WorkingState] = None,
    ) -> Observation:
        context = context or {}
        extracted: Any = payload
        alternatives: list = []
        uncertainty = 0.0
        version = "passthrough_v1"

        if modality == "audio":
            result = self.speech.recognize(payload, context=context)
            extracted = result["transcript"]
            alternatives = result.get("alternatives") or []
            uncertainty = float(result.get("uncertainty") or 0.0)
            version = result.get("version", "speech_v1")
        elif modality in ("image", "document"):
            result = self.documents.recognize(payload, context=context)
            extracted = result.get("fields") or result.get("text")
            uncertainty = float(result.get("uncertainty") or 0.0)
            version = result.get("version", "ocr_v1")
            alternatives = result.get("alternatives") or []
        elif modality == "text":
            extracted = str(payload)
            version = "text_v1"

        obs = Observation(
            source=source,
            modality=modality,
            raw_evidence=payload if modality != "text" else None,
            extracted_value=extracted,
            recognition_version=version,
            uncertainty=uncertainty,
            context=context,
            alternatives=alternatives,
        )
        if state is not None:
            state.add_observation(obs)
        return obs
