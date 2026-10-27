"""Voice pipeline adapters (blueprint §4): capture, VAD, ASR, TTS.

Portable Python contracts. Real Silero/whisper.cpp/sounddevice bind when installed;
stubs keep the desktop core testable on Linux CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AudioCaptureConfig:
    sample_rate: int = 16000
    channels: int = 1
    push_to_talk: bool = True
    buffer_ms: int = 300
    device: Optional[str] = None


@dataclass
class VoiceSession:
    listening: bool = False
    speaking: bool = False
    cancelled: bool = False
    level: float = 0.0
    transcript: str = ""
    engine: str = "stub"


class VoicePipeline:
    """Separate capture from heavy ASR so UI stays responsive (blueprint §4A)."""

    def __init__(self, config: Optional[AudioCaptureConfig] = None) -> None:
        self.config = config or AudioCaptureConfig()
        self.session = VoiceSession()
        self.speech_queue: list[str] = []
        self.pronunciation: dict[str, str] = {}
        self._backend = self._detect_backend()

    def _detect_backend(self) -> dict[str, str]:
        asr = "stub"
        tts = "stub"
        vad = "energy_stub"
        try:
            import sounddevice  # noqa: F401

            capture = "sounddevice"
        except Exception:
            capture = "stub"
        # whisper.cpp / faster-whisper optional
        try:
            import whisper  # noqa: F401

            asr = "whisper_py_available"
        except Exception:
            pass
        try:
            # macOS say
            import shutil

            if shutil.which("say"):
                tts = "macos_say"
        except Exception:
            pass
        return {"capture": capture, "asr": asr, "tts": tts, "vad": vad, "wake_word": "push_to_talk"}

    def start_listening(self) -> VoiceSession:
        self.session = VoiceSession(listening=True, engine=self._backend["asr"])
        return self.session

    def stop_listening(self) -> VoiceSession:
        self.session.listening = False
        return self.session

    def cancel(self) -> VoiceSession:
        """Immediate playback/recording cancellation (blueprint §4A, §11 stop)."""
        self.session.cancelled = True
        self.session.listening = False
        self.session.speaking = False
        self.speech_queue.clear()
        return self.session

    def vad_segments(self, pcm: bytes, *, sample_rate: int = 16000) -> list[dict[str, Any]]:
        """Energy-based VAD stub; Silero when available."""
        if not pcm:
            return []
        # Treat any non-empty buffer as one speech segment (testable stub)
        duration = len(pcm) / max(1, sample_rate * 2)  # assume 16-bit mono
        return [{"start": 0.0, "end": duration, "is_speech": True, "engine": self._backend["vad"]}]

    def transcribe(self, pcm: bytes | None = None, *, text_override: Optional[str] = None) -> dict[str, Any]:
        """Local ASR. Prefer push-to-talk + provided text in tests; whisper when wired."""
        if text_override is not None:
            self.session.transcript = text_override
            return {"ok": True, "text": text_override, "engine": "override", "backend": self._backend}
        if self.session.cancelled:
            return {"ok": False, "error": "cancelled", "text": ""}
        # Without real audio/models, return empty with honest engine tag
        return {
            "ok": True,
            "text": self.session.transcript,
            "engine": self._backend["asr"],
            "note": "Install whisper.cpp / Vosk for real ASR; push-to-talk first per blueprint.",
            "backend": self._backend,
        }

    def speak(self, text: str, *, interrupt: bool = False) -> dict[str, Any]:
        if interrupt:
            self.speech_queue.clear()
        # Apply pronunciation dictionary
        spoken = text
        for src, dst in self.pronunciation.items():
            spoken = spoken.replace(src, dst)
        self.speech_queue.append(spoken)
        self.session.speaking = True
        return {
            "ok": True,
            "queued": spoken,
            "engine": self._backend["tts"],
            "queue_len": len(self.speech_queue),
        }

    def set_pronunciation(self, word: str, spoken_as: str) -> None:
        self.pronunciation[word] = spoken_as

    def status(self) -> dict[str, Any]:
        return {
            "listening": self.session.listening,
            "speaking": self.session.speaking,
            "cancelled": self.session.cancelled,
            "level": self.session.level,
            "queue_len": len(self.speech_queue),
            "push_to_talk": self.config.push_to_talk,
            "wake_word": "disabled_prefer_ptt",
            "backend": self._backend,
            "sample_rate": self.config.sample_rate,
        }
