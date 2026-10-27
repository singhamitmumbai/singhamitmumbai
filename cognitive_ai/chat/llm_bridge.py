"""Optional ChatGPT-class chat bridge (OpenAI-compatible HTTP API).

This is the *only* path that uses a pretrained general-purpose LLM at runtime.
It is OFF by default. Enable with env:

  COMPLETE_AI_ENABLE_LLM=1
  OPENAI_API_KEY=...
  COMPLETE_AI_LLM_MODEL=gpt-4o-mini          (optional)
  COMPLETE_AI_LLM_BASE=https://api.openai.com/v1  (optional)

When disabled, Complete AI stays fully non-LLM.
When enabled, status reports llm_at_runtime=true for that turn — never claims AGI.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional


DEFAULT_SYSTEM = (
    "You are Complete AI's optional general-chat mode — a helpful assistant. "
    "Be clear and accurate. You are not human-level AGI; do not claim to be conscious. "
    "For Finderbay business tasks (invoices, stock, POs), prefer directing the user "
    "to the symbolic task engine when appropriate."
)


class LLMChatBridge:
    """Thin OpenAI-compatible chat completions client (stdlib urllib)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        enabled: Optional[bool] = None,
        system_prompt: str = DEFAULT_SYSTEM,
        timeout_s: float = 60.0,
    ) -> None:
        env_on = os.environ.get("COMPLETE_AI_ENABLE_LLM", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.enabled = env_on if enabled is None else enabled
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

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.api_key)

    def status(self) -> dict[str, Any]:
        return {
            "name": "LLMChatBridge",
            "enabled": self.enabled,
            "available": self.available,
            "model": self.model if self.available else None,
            "base_url": self.base_url if self.enabled else None,
            "has_api_key": bool(self.api_key),
            "llm_at_runtime": self.available,
            "human_level_agi": False,
            "note": (
                "ChatGPT-class general chat when enabled with API key; "
                "OFF by default so core Complete AI stays non-LLM."
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
                    "LLM chat is not enabled. Set COMPLETE_AI_ENABLE_LLM=1 and "
                    "OPENAI_API_KEY to use ChatGPT-class general chat, or keep "
                    "talking to the non-LLM Complete AI engine."
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
        messages.extend(self.history[-20:])
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
        except Exception as exc:  # noqa: BLE001 — surface as chat error
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
