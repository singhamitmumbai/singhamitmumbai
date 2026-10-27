"""Always-on multimodal orchestration (speech / OCR / vision) — best-effort.

Does not claim human-level perception. Activates available engines, falls back
to stubs, and exposes a unified tick/status API for continuous sensing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from ..core.records import utc_now
from ..perception.gateway import InputGateway


@dataclass
class ModalityChannel:
    name: str
    enabled: bool = True
    available: bool = False
    engine: str = "stub"
    last_tick: Optional[str] = None
    last_error: Optional[str] = None
    events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlwaysOnMultimodalHub:
    """Register channels and ingest across modalities in one place."""

    def __init__(self, gateway: Optional[InputGateway] = None) -> None:
        self.gateway = gateway or InputGateway(prefer_production=True)
        self.channels: dict[str, ModalityChannel] = {
            "text": ModalityChannel("text", available=True, engine="passthrough"),
            "speech": ModalityChannel("speech"),
            "ocr": ModalityChannel("ocr"),
            "vision": ModalityChannel("vision"),
        }
        self.event_log: list[dict[str, Any]] = []
        self.session_buffer: list[dict[str, Any]] = []
        self._probe()

    def _probe(self) -> None:
        speech = getattr(self.gateway, "speech", None)
        if speech is not None:
            self.channels["speech"].available = bool(getattr(speech, "available", True))
            eng = getattr(speech, "engine", None) or getattr(speech, "backend", "speech")
            self.channels["speech"].engine = str(eng)

        docs = getattr(self.gateway, "documents", None)
        if docs is not None:
            self.channels["ocr"].available = bool(getattr(docs, "available", True))
            eng = getattr(docs, "engine", None) or "ocr"
            self.channels["ocr"].engine = str(eng)

        try:
            from ..vision.system import VisionSystem

            vs = VisionSystem()
            st = vs.status() if hasattr(vs, "status") else {}
            eng = (st or {}).get("engine") or "stub"
            self.channels["vision"].available = True  # stub counts as channel present
            self.channels["vision"].engine = str(eng)
            self.gateway.vision = vs
        except Exception as exc:  # noqa: BLE001
            self.channels["vision"].last_error = str(exc)

    def status(self) -> dict[str, Any]:
        return {
            "channels": {k: v.to_dict() for k, v in self.channels.items()},
            "always_on": True,
            "events": len(self.event_log),
            "human_level_agi": False,
            "note": "Best-effort multimodal hub; human-level perception not claimed.",
        }

    def enable(self, channel: str, on: bool = True) -> None:
        if channel in self.channels:
            self.channels[channel].enabled = on

    def ingest(
        self,
        payload: Any,
        *,
        modality: str = "text",
        source: str = "sensor",
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        ch = self.channels.get(modality) or self.channels["text"]
        if not ch.enabled:
            return {
                "ok": False,
                "error": f"channel_{modality}_disabled",
                "human_level_agi": False,
            }
        obs = self.gateway.ingest(
            payload, modality=modality, source=source, context=context
        )
        ch.events += 1
        ch.last_tick = utc_now()
        extracted = obs.extracted_value
        event = {
            "modality": modality,
            "extracted": extracted,
            "uncertainty": obs.uncertainty,
            "version": obs.recognition_version,
            "ts": ch.last_tick,
        }
        self.event_log.append(event)
        self.session_buffer.append(event)
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]
        if len(self.session_buffer) > 50:
            self.session_buffer = self.session_buffer[-50:]
        return {
            "ok": True,
            "observation": obs.to_dict() if hasattr(obs, "to_dict") else event,
            "channel": ch.to_dict(),
            "human_level_agi": False,
        }

    def arbitrate(self) -> dict[str, Any]:
        """Pick the most trustworthy recent modality event (lowest uncertainty)."""
        if not self.session_buffer:
            return {"ok": False, "error": "empty_session_buffer", "human_level_agi": False}
        ranked = sorted(
            self.session_buffer,
            key=lambda e: float(e.get("uncertainty") if e.get("uncertainty") is not None else 0.5),
        )
        best = ranked[0]
        return {
            "ok": True,
            "winner": best,
            "modality": best.get("modality"),
            "confidence": round(1.0 - float(best.get("uncertainty") or 0.5), 3),
            "candidates": len(ranked),
            "human_level_agi": False,
            "note": "Arbitration is heuristic — not human-level multimodal judgment.",
        }

    def clear_session(self) -> None:
        self.session_buffer.clear()

    def tick_all_health(self) -> dict[str, Any]:
        """Health tick for always-on monitoring (no heavy capture required)."""
        self._probe()
        return {
            "ts": utc_now(),
            "channels": {
                name: {
                    "enabled": ch.enabled,
                    "available": ch.available,
                    "engine": ch.engine,
                    "events": ch.events,
                }
                for name, ch in self.channels.items()
            },
            "human_level_agi": False,
        }

    def fuse(
        self,
        inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Confidence-weighted multimodal fusion of several ingest results/payloads.

        Each item: {modality, payload} or prior ingest result with observation.
        """
        parts: list[dict[str, Any]] = []
        for item in inputs:
            if "observation" in item or item.get("ok") is not None and "channel" in item:
                parts.append(item)
                continue
            modality = str(item.get("modality") or "text")
            payload = item.get("payload", item.get("text", ""))
            parts.append(self.ingest(payload, modality=modality, source=str(item.get("source") or "fuse")))
        # Weight by (1 - uncertainty)
        texts: list[str] = []
        weight_sum = 0.0
        conf = 0.0
        modalities: list[str] = []
        for p in parts:
            obs = p.get("observation") or {}
            if isinstance(obs, dict):
                unc = float(obs.get("uncertainty") or 0.3)
                val = obs.get("extracted_value") or obs.get("extracted") or ""
                mod = obs.get("modality") or (p.get("channel") or {}).get("name") or "text"
            else:
                unc, val, mod = 0.3, str(obs), "text"
            w = max(0.05, 1.0 - unc)
            texts.append(f"[{mod} w={w:.2f}] {val}")
            weight_sum += w
            conf += w * (1.0 - unc)
            modalities.append(str(mod))
        fused_conf = (conf / weight_sum) if weight_sum else 0.0
        return {
            "ok": bool(parts),
            "fused_text": " | ".join(str(t) for t in texts),
            "modalities": modalities,
            "confidence": round(fused_conf, 3),
            "parts": len(parts),
            "human_level_agi": False,
            "note": "Fusion is heuristic — not human-level multimodal understanding.",
        }
