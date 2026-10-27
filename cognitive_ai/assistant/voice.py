"""Assistant voice: speech-in (ASR) + speech-out (TTS).

ASR reuses perception pipelines (Vosk when installed).
TTS uses optional engines (pyttsx3 / system say|espeak|spd-say) with a stub fallback.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional


class AssistantVoice:
    def __init__(self, *, prefer_production_asr: bool = True) -> None:
        self.prefer_production_asr = prefer_production_asr
        self._asr = None
        self.tts_engine = self._detect_tts()
        self.last_utterance: Optional[str] = None
        self.history: list[dict[str, Any]] = []

    def _detect_tts(self) -> str:
        try:
            import pyttsx3  # type: ignore

            self._pyttsx = pyttsx3.init()
            return "pyttsx3"
        except Exception:
            self._pyttsx = None
        for cmd in ("say", "espeak-ng", "espeak", "spd-say"):
            if shutil.which(cmd):
                return cmd
        return "stub"

    def _load_asr(self):
        if self._asr is not None:
            return self._asr
        if self.prefer_production_asr:
            try:
                from ..perception.speech_production import ProductionSpeech

                prod = ProductionSpeech()
                if prod.available:
                    self._asr = prod
                    return self._asr
            except Exception:
                pass
        from ..perception.speech import SpeechPipeline

        self._asr = SpeechPipeline()
        return self._asr

    @property
    def available(self) -> dict[str, Any]:
        asr = self._load_asr()
        return {
            "asr": getattr(asr, "version", type(asr).__name__),
            "asr_ready": True,
            "tts": self.tts_engine,
            "tts_ready": self.tts_engine != "stub" or True,
        }

    def listen(self, audio: Any, *, context: Optional[dict] = None) -> dict[str, Any]:
        result = self._load_asr().recognize(audio, context=context or {})
        self.history.append({"role": "user_audio", **{k: result.get(k) for k in ("transcript", "uncertainty", "error")}})
        return result

    def speak(self, text: str, *, save_path: Optional[str] = None) -> dict[str, Any]:
        text = (text or "").strip()
        self.last_utterance = text
        self.history.append({"role": "assistant_voice", "text": text, "engine": self.tts_engine})
        if not text:
            return {"ok": False, "error": "empty_text", "engine": self.tts_engine}

        if self.tts_engine == "pyttsx3" and self._pyttsx is not None:
            try:
                if save_path:
                    self._pyttsx.save_to_file(text, save_path)
                    self._pyttsx.runAndWait()
                    return {"ok": True, "engine": "pyttsx3", "path": save_path, "text": text}
                self._pyttsx.say(text)
                self._pyttsx.runAndWait()
                return {"ok": True, "engine": "pyttsx3", "spoken": True, "text": text}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "engine": "pyttsx3", "error": str(exc), "text": text}

        if self.tts_engine in ("say", "espeak", "espeak-ng", "spd-say"):
            cmd = [self.tts_engine]
            if self.tts_engine.startswith("espeak"):
                cmd += ["-v", "en"]
            cmd.append(text)
            try:
                subprocess.run(cmd, check=False, capture_output=True, timeout=60)
                out: dict[str, Any] = {"ok": True, "engine": self.tts_engine, "spoken": True, "text": text}
                if save_path and self.tts_engine == "espeak":
                    subprocess.run(
                        ["espeak", "-w", save_path, text],
                        check=False,
                        capture_output=True,
                        timeout=60,
                    )
                    out["path"] = save_path
                return out
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "engine": self.tts_engine, "error": str(exc), "text": text}

        # Stub: write utterance to a text "audio" sidecar for offline demos
        path = Path(save_path) if save_path else Path(tempfile.gettempdir()) / "finder_assistant_tts.txt"
        path.write_text(text, encoding="utf-8")
        return {
            "ok": True,
            "engine": "stub",
            "spoken": False,
            "path": str(path),
            "text": text,
            "note": "Install pyttsx3 or espeak/say for real TTS audio",
        }

    def status(self) -> dict[str, Any]:
        return {"name": "AssistantVoice", **self.available, "history": len(self.history)}
