"""Input gateway: receive inputs with identity, time and context."""

from __future__ import annotations

from typing import Any, Optional

from ..core.records import Observation
from ..core.working_state import WorkingState
from .ocr import DocumentPipeline
from .ocr_production import ProductionOCR
from .speech import SpeechPipeline
from .speech_production import ProductionSpeech


class InputGateway:
    def __init__(
        self,
        speech: Optional[Any] = None,
        documents: Optional[Any] = None,
        *,
        prefer_production: bool = True,
    ) -> None:
        if prefer_production:
            prod_speech = ProductionSpeech()
            prod_ocr = ProductionOCR()
            self.speech = speech or (prod_speech if prod_speech.available else SpeechPipeline())
            self.documents = documents or (prod_ocr if prod_ocr.available else DocumentPipeline())
        else:
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
            extracted = result.get("transcript", "")
            alternatives = result.get("alternatives") or []
            uncertainty = float(result.get("uncertainty") or 0.0)
            version = result.get("version", "speech_v1")
            context = {**context, "speech_meta": {k: result.get(k) for k in ("engine", "error", "model")}}
        elif modality in ("image", "document"):
            result = self.documents.recognize(payload, context=context)
            extracted = result.get("fields") or result.get("text")
            uncertainty = float(result.get("uncertainty") or 0.0)
            version = result.get("version", "ocr_v1")
            alternatives = result.get("alternatives") or []
            context = {**context, "ocr_meta": {k: result.get(k) for k in ("engine", "error", "lang", "issues")}}
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
