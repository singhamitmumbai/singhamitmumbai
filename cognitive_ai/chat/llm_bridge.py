"""ChatGPT-class general chat for Complete AI (OpenAI-compatible HTTP API).

Primary general-chat engine when an API key is present.

Enable (any of):
  OPENAI_API_KEY=sk-...                         # auto-enables general LLM chat
  COMPLETE_AI_ENABLE_LLM=1                      # force enable
  COMPLETE_AI_LLM_KEY=...                       # alt key name

Disable:
  COMPLETE_AI_DISABLE_LLM=1

Optional:
  COMPLETE_AI_LLM_MODEL=gpt-4o-mini
  COMPLETE_AI_LLM_BASE=https://api.openai.com/v1

Status always reports human_level_agi=false. When active, llm_at_runtime=true.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional


DEFAULT_SYSTEM = (
    "You are Complete AI — a full general assistant for conversation, writing, "
    "brainstorming, coding help, explanations, and everyday Q&A (ChatGPT-class). "
    "Be helpful, clear, and accurate. Match the user's language. "
    "Do not claim to be conscious or human-level AGI. "
    "If the user asks about Finderbay invoices, inventory, purchase orders, or "
    "on-device YOLO vision jobs, note that Complete AI also has dedicated task "
    "and vision engines for those."
)


def _env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


class LLMChatBridge:
    """OpenAI-compatible chat completions client (stdlib urllib)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        enabled: Optional[bool] = None,
        system_prompt: str = DEFAULT_SYSTEM,
        timeout_s: float = 90.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "COMPLETE_AI_LLM_KEY"
        )
        self.base_url = (
            base_url
            or os.environ.get("COMPLETE_AI_LLM_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("COMPLETE_AI_LLM_MODEL") or "gpt-4o-mini"
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.history: list[dict[str, str]] = []

        if enabled is not None:
            self.enabled = enabled
        else:
            disable = _env_flag("COMPLETE_AI_DISABLE_LLM")
            force_on = _env_flag("COMPLETE_AI_ENABLE_LLM")
            if disable is True:
                self.enabled = False
            elif force_on is True:
                self.enabled = True
            else:
                # Auto-on when a key is present → ChatGPT-class general chat by default
                self.enabled = bool(self.api_key)

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.api_key)

    def status(self) -> dict[str, Any]:
        return {
            "name": "LLMChatBridge",
            "role": "chatgpt_class_general_chat",
            "enabled": self.enabled,
            "available": self.available,
            "model": self.model if self.available else None,
            "base_url": self.base_url if self.enabled else None,
            "has_api_key": bool(self.api_key),
            "llm_at_runtime": self.available,
            "human_level_agi": False,
            "note": (
                "Primary general chat when API key is set. "
                "Set COMPLETE_AI_DISABLE_LLM=1 to force symbolic-only chat."
            ),
        }

    def chat(
        self,
        message: str,
        *,
        system: Optional[str] = None,
        reset: bool = False,
    ) -> dict[str, Any]:
        if not self.available:
            return {
                "ok": False,
                "text": (
                    "General LLM chat needs an API key. Set OPENAI_API_KEY "
                    "(or COMPLETE_AI_LLM_KEY), then ask again. "
                    "Until then Complete AI still answers with its symbolic engines."
                ),
                "error": "llm_disabled_or_no_key",
                "llm_at_runtime": False,
                "human_level_agi": False,
            }
        if reset:
            self.history.clear()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system or self.system_prompt}
        ]
        messages.extend(self.history[-40:])
        messages.append({"role": "user", "content": message})
        try:
            data = self._post_chat(messages)
            text = self._extract_text(data)
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": text})
            return {
                "ok": True,
                "text": text,
                "model": self.model,
                "llm_at_runtime": True,
                "human_level_agi": False,
                "method": "llm_bridge",
                "raw_usage": data.get("usage"),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "text": f"LLM chat failed: {exc}",
                "error": str(exc),
                "llm_at_runtime": True,
                "human_level_agi": False,
                "method": "llm_bridge_error",
            }

    def _post_chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "").strip()


def llm_bridge_from_env() -> LLMChatBridge:
    return LLMChatBridge()
