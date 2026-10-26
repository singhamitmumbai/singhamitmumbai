"""Microphone / chunk streaming ASR interface (Vosk) — non-LLM.

Provides a push-audio API suitable for websockets or local mic readers.
Does not require a microphone at import time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .model_registry import ensure_model
from .speech_production import ProductionSpeech


class StreamingASR:
    """Incremental recognition over PCM16 mono chunks."""

    def __init__(self, model_name: str = "en-small", sample_rate: int = 16000) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self._rec = None
        self._model = None
        self.partial = ""
        self.final_parts: list[str] = []

    def start(self, *, download_model: bool = False) -> None:
        from vosk import KaldiRecognizer, Model

        path = ensure_model(self.model_name, download=download_model)
        self._model = Model(str(path))
        self._rec = KaldiRecognizer(self._model, self.sample_rate)
        self._rec.SetWords(True)
        self.partial = ""
        self.final_parts = []

    def accept_audio(self, pcm16_bytes: bytes) -> dict[str, Any]:
        if self._rec is None:
            raise RuntimeError("Call start() before accept_audio()")
        if self._rec.AcceptWaveform(pcm16_bytes):
            res = json.loads(self._rec.Result())
            text = (res.get("text") or "").strip()
            if text:
                self.final_parts.append(text)
            self.partial = ""
            return {"type": "final", "text": text, "utterance": self.transcript()}
        partial = json.loads(self._rec.PartialResult())
        self.partial = (partial.get("partial") or "").strip()
        return {"type": "partial", "text": self.partial, "utterance": self.transcript()}

    def finish(self) -> dict[str, Any]:
        if self._rec is None:
            return {"transcript": "", "parts": []}
        final = json.loads(self._rec.FinalResult())
        text = (final.get("text") or "").strip()
        if text:
            self.final_parts.append(text)
        self.partial = ""
        return {"transcript": self.transcript(), "parts": list(self.final_parts)}

    def transcript(self) -> str:
        bits = list(self.final_parts)
        if self.partial:
            bits.append(self.partial)
        return " ".join(bits).strip()


def recognize_file(path: str | Path, model_name: str = "en-small") -> dict[str, Any]:
    """Recognize a WAV file with a selected language model."""
    model_path = ensure_model(model_name, download=False) if False else None
    # Prefer installed path; ProductionSpeech accepts model_path
    from .model_registry import model_path as mp

    p = mp(model_name)
    speech = ProductionSpeech(model_path=p if p.exists() else None)
    return speech.recognize(str(path))


def mic_stream_loop(
    seconds: float = 5.0,
    model_name: str = "en-small",
    on_event: Optional[Callable[[dict], None]] = None,
) -> dict[str, Any]:
    """Optional live mic capture using sounddevice if installed.

    Returns transcript or a clear dependency error — never pretends to hear.
    """
    try:
        import sounddevice as sd
    except Exception as exc:
        return {
            "transcript": "",
            "error": "sounddevice_not_installed",
            "detail": str(exc),
            "hint": "pip install sounddevice and grant mic permissions",
        }
    streamer = StreamingASR(model_name=model_name, sample_rate=16000)
    try:
        streamer.start(download_model=False)
    except Exception as exc:
        return {"transcript": "", "error": "model_start_failed", "detail": str(exc)}

    block = 4000
    frames_needed = int(16000 * seconds)
    read = 0
    events = []
    with sd.RawInputStream(samplerate=16000, blocksize=block, dtype="int16", channels=1) as stream:
        while read < frames_needed:
            data, _ = stream.read(block)
            evt = streamer.accept_audio(bytes(data))
            events.append(evt)
            if on_event:
                on_event(evt)
            read += block
    final = streamer.finish()
    final["events"] = len(events)
    final["engine"] = "vosk_streaming"
    final["model"] = model_name
    return final
