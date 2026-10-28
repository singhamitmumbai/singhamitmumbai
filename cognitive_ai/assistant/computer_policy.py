"""Sandbox + denylist policy for gated local computer control.

Controls the *user's own* machine only. Never enables keylogging or background
clipboard/surveillance capture.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional


# Apps that may be launched by name (mapped per OS later)
APP_ALLOWLIST: dict[str, dict[str, list[str]]] = {
    "calculator": {
        "linux": ["gnome-calculator", "kcalc", "galculator", "xcalc"],
        "darwin": ["open", "-a", "Calculator"],
        "win": ["calc.exe"],
    },
    "text_editor": {
        "linux": ["gedit", "kate", "mousepad", "xdg-open"],
        "darwin": ["open", "-a", "TextEdit"],
        "win": ["notepad.exe"],
    },
    "browser": {
        "linux": ["xdg-open"],
        "darwin": ["open"],
        "win": ["cmd", "/c", "start"],
    },
    "file_manager": {
        "linux": ["xdg-open", "nautilus", "dolphin", "thunar"],
        "darwin": ["open"],
        "win": ["explorer.exe"],
    },
    "terminal": {
        "linux": ["x-terminal-emulator", "gnome-terminal", "konsole", "xterm"],
        "darwin": ["open", "-a", "Terminal"],
        "win": ["cmd.exe"],
    },
}

# Shell substrings that are always refused
SHELL_DENY = [
    r"rm\s+-rf\s+/",
    r"mkfs\b",
    r"dd\s+if=",
    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",
    r"shutdown\b",
    r"reboot\b",
    r"poweroff\b",
    r"format\s+[a-z]:",
    r"del\s+/[sf]",
    r"curl\s+[^\n]*\|\s*sh",
    r"wget\s+[^\n]*\|\s*sh",
    r"chmod\s+-R\s+777\s+/",
    r">\s*/etc/",
    r">\s*/boot/",
]

# Paths we never write/delete even with confirm
FORBIDDEN_PREFIXES = (
    "/etc",
    "/boot",
    "/sys",
    "/proc",
    "/dev",
    "/root",
    "C:\\Windows",
    "C:\\Windows\\System32",
)


def platform_family() -> str:
    import sys

    if sys.platform == "darwin":
        return "darwin"
    if os.name == "nt":
        return "win"
    return "linux"


def default_sandbox_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("COMPLETE_AI_COMPUTER_ROOT", "").strip()
    if env:
        for part in env.split(os.pathsep):
            if part.strip():
                roots.append(Path(part).expanduser().resolve())
    cwd = Path.cwd().resolve()
    roots.append(cwd)
    roots.append((cwd / "data").resolve())
    home = Path.home()
    for sub in ("Documents", "Downloads", "Desktop", "tmp", "Temp"):
        p = home / sub
        if p.exists():
            roots.append(p.resolve())
    # de-dupe
    seen = set()
    out: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def resolve_in_sandbox(
    path: str | Path,
    *,
    roots: Optional[list[Path]] = None,
    must_exist: bool = False,
) -> tuple[Optional[Path], Optional[str]]:
    """Resolve path; return (path, error)."""
    roots = roots or default_sandbox_roots()
    try:
        p = Path(str(path)).expanduser()
        # Allow relative paths against cwd
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
    except Exception as exc:  # noqa: BLE001
        return None, f"bad_path:{exc}"

    sp = str(p)
    for bad in FORBIDDEN_PREFIXES:
        if sp.lower().startswith(bad.lower()):
            return None, f"forbidden_system_path:{bad}"

    if must_exist and not p.exists():
        return None, "path_not_found"

    # Parent must be under a sandbox root for writes; for reads same rule
    for root in roots:
        try:
            p.relative_to(root)
            return p, None
        except ValueError:
            continue
    return None, f"outside_sandbox:{sp}"


def shell_allowed(command: str | list[str]) -> tuple[bool, str]:
    text = command if isinstance(command, str) else " ".join(str(c) for c in command)
    lower = text.lower()
    for pat in SHELL_DENY:
        if re.search(pat, lower):
            return False, f"shell_denied_pattern:{pat}"
    return True, "ok"


def policy_snapshot() -> dict[str, Any]:
    return {
        "sandbox_roots": [str(p) for p in default_sandbox_roots()],
        "apps": sorted(APP_ALLOWLIST),
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "shell_deny_patterns": list(SHELL_DENY),
        "no_keylogging": True,
        "no_background_clipboard_monitor": True,
        "human_level_agi": False,
    }
