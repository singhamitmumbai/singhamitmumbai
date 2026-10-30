"""Authorization & policy — planner must not self-approve (blueprint §8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AuthContext:
    """Trusted application state — never taken from planner model output alone."""

    user_id: str = "local"
    grants: set[str] = field(default_factory=lambda: {"read", "local_write"})
    approved_folders: list[str] = field(default_factory=lambda: ["reports", "data", "."])
    approved_projects: list[str] = field(default_factory=list)
    computer_control: bool = False
    external_write: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "grants": sorted(self.grants),
            "approved_folders": list(self.approved_folders),
            "approved_projects": list(self.approved_projects),
            "computer_control": self.computer_control,
            "external_write": self.external_write,
        }


class PolicyGuard:
    """Derive authorization from trusted grants — ignore planner 'approved: true'."""

    EFFECT_GRANT = {
        "read": "read",
        "local_write": "local_write",
        "external_write": "external_write",
        "side_effect": "local_write",
    }

    def __init__(self, auth: Optional[AuthContext] = None) -> None:
        self.auth = auth or AuthContext()

    def authorize_tool(self, tool_name: str, effect: str, args: dict[str, Any]) -> dict[str, Any]:
        # Strip any self-approval from planner args
        cleaned = {k: v for k, v in args.items() if k not in ("approved", "authorize", "permission")}
        need = self.EFFECT_GRANT.get(effect, "read")
        if need == "external_write" and not self.auth.external_write:
            return {"ok": False, "error": "external_write_denied", "tool": tool_name}
        if need not in self.auth.grants:
            return {"ok": False, "error": "grant_missing", "need": need, "tool": tool_name}
        if effect == "local_write":
            path = str(cleaned.get("path") or cleaned.get("folder") or "reports")
            if not self._path_allowed(path):
                return {"ok": False, "error": "scope_denied", "path": path, "tool": tool_name}
        project = cleaned.get("site") or cleaned.get("project") or cleaned.get("project_id")
        if project and self.auth.approved_projects:
            if str(project) not in self.auth.approved_projects and str(project) != "selected_site":
                return {"ok": False, "error": "project_not_approved", "project": project}
        return {"ok": True, "tool": tool_name, "args": cleaned, "grant": need}

    def _path_allowed(self, path: str) -> bool:
        p = path.replace("\\", "/")
        for folder in self.auth.approved_folders:
            if p == folder or p.startswith(folder.rstrip("/") + "/") or folder == ".":
                return True
        return False

    def status(self) -> dict[str, Any]:
        return self.auth.to_dict()
