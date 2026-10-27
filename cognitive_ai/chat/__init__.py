"""General chat stack — open-domain (non-LLM) + optional LLM bridge."""

from .llm_bridge import LLMChatBridge, llm_bridge_from_env

__all__ = ["LLMChatBridge", "llm_bridge_from_env"]
