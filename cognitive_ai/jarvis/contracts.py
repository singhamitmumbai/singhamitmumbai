"""Tool contracts, job states, and provenance fact records (JARVIS blueprint §7–§8, §11)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now


class JobState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    FAILED = "failed"
    PARTIAL = "partial"
    OUTCOME_UNKNOWN = "outcome_unknown"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


@dataclass
class ToolContract:
    """Every callable tool needs a contract — not just a name (blueprint §8)."""

    name: str
    version: str = "1.0"
    description: str = ""
    input_schema: dict[str, str] = field(default_factory=dict)  # field → type
    output_schema: dict[str, str] = field(default_factory=dict)
    effect: str = "read"  # read | local_write | external_write | side_effect
    scope: str = "local"
    timeout_seconds: float = 30.0
    retry_policy: str = "reconcile_before_retry"
    verification: str = "inspect_result"
    requires_auth: bool = False
    requires_confirm: bool = False
    family: str = "general"

    def validate_input(self, args: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for key, typ in self.input_schema.items():
            if key not in args or args[key] in (None, ""):
                # optional fields end with ?
                if key.endswith("?"):
                    continue
                clean = key.rstrip("?")
                if clean not in args or args.get(clean) in (None, ""):
                    if "?" in key:
                        continue
                    errors.append(f"missing:{key}")
                    continue
            val = args.get(key) if key in args else args.get(key.rstrip("?"))
            if typ == "str" and not isinstance(val, str):
                errors.append(f"type:{key}:str")
            elif typ == "number" and not isinstance(val, (int, float)):
                errors.append(f"type:{key}:number")
            elif typ == "bool" and not isinstance(val, bool):
                errors.append(f"type:{key}:bool")
            elif typ == "dict" and not isinstance(val, dict):
                errors.append(f"type:{key}:dict")
            elif typ == "list" and not isinstance(val, list):
                errors.append(f"type:{key}:list")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactRecord:
    """Provenance-aware semantic memory item (blueprint §7)."""

    subject: str
    relation: str
    value: Any
    units: Optional[str] = None
    source_id: str = ""
    source_version: str = "1"
    page_or_row: Optional[str] = None
    observed_at: str = field(default_factory=utc_now)
    recorded_at: str = field(default_factory=utc_now)
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    verification_status: str = "asserted"  # asserted|observed|calculated|user_said|document
    access_scope: str = "local"
    supersedes_fact_id: Optional[str] = None
    fact_id: str = field(default_factory=lambda: new_id("fact"))
    kind: str = "semantic"  # working|episodic|semantic|procedural|source

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifiedArtifactResult:
    ok: bool
    artifact_id: Optional[str] = None
    path: Optional[str] = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
