"""Runtime configuration for the complete non-LLM AI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SystemConfig:
    domain: str = "finderbay"
    seed_demo: bool = True
    storage_path: str | None = None
    dataset_dir: str = "data/finderbay/examples"
    allow_send: bool = False
    allow_inventory_write: bool = True
    default_language: str = "en"  # en | hi | hinglish
    resource_time_ms: int = 10000
    resource_steps: int = 50
    resource_external_calls: int = 20
    enable_speech_oracle: bool = True  # accept {"transcript": ...}
    enable_ocr_oracle: bool = True
    log_episodes: bool = True
    # Complete AI v1.5+
    enable_llm_chat: bool | None = None  # None = auto when API key present
    max_tool_rounds: int = 3
    session_limit: int = 100
    extra: dict[str, Any] = field(default_factory=dict)

    def storage(self) -> Path | None:
        return Path(self.storage_path) if self.storage_path else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)
