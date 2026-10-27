"""Gated local computer control for the Finderbay Assistant.

Safety:
  • Disabled by default (enable with allow=True or COMPLETE_AI_COMPUTER_CONTROL=1)
  • Allowlisted actions only (no keylogging, no silent clipboard siphoning)
  • Every action is audited
  • Destructive-ish actions require confirm=True

This is for controlling the *user's own* machine as an assistant — not remote
unauthorized access.
"""

from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional


ALLOWED = {
    "open_url",
    "open_path",
    "screenshot",
    "notify",
    "run_command",
    "type_text",
    "hotkey",
    "mouse_click",
    "status",
}


class ComputerControl:
    def __init__(self, *, allow: Optional[bool] = None, allow_shell: bool = False) -> None:
        env = os.environ.get("COMPLETE_AI_COMPUTER_CONTROL", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.allow = env if allow is None else allow
        self.allow_shell = allow_shell or os.environ.get(
            "COMPLETE_AI_COMPUTER_SHELL", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        self.audit: list[dict[str, Any]] = []

    def status(self) -> dict[str, Any]:
        return {
            "name": "ComputerControl",
            "enabled": self.allow,
            "allow_shell": self.allow_shell,
            "allowed_actions": sorted(ALLOWED),
            "audit_events": len(self.audit),
            "note": "Set COMPLETE_AI_COMPUTER_CONTROL=1 to enable local automation",
        }

    def enable(self) -> None:
        self.allow = True

    def disable(self) -> None:
        self.allow = False

    def run(self, action: str, **kwargs: Any) -> dict[str, Any]:
        action = (action or "").strip().lower()
        event = {"action": action, "kwargs": _safe_kwargs(kwargs), "ts": time.time()}
        if action not in ALLOWED:
            event["ok"] = False
            event["error"] = f"action_not_allowlisted:{action}"
            self.audit.append(event)
            return event
        if action == "status":
            event["ok"] = True
            event["result"] = self.status()
            self.audit.append(event)
            return event
        if not self.allow:
            event["ok"] = False
            event["error"] = "computer_control_disabled"
            self.audit.append(event)
            return event
        try:
            result = self._dispatch(action, **kwargs)
            event["ok"] = bool(result.get("ok", True))
            event["result"] = result
        except Exception as exc:  # noqa: BLE001
            event["ok"] = False
            event["error"] = str(exc)
        self.audit.append(event)
        return event

    def _dispatch(self, action: str, **kwargs: Any) -> dict[str, Any]:
        if action == "open_url":
            url = str(kwargs.get("url") or "")
            if not url.startswith(("http://", "https://", "file:")):
                return {"ok": False, "error": "url_must_be_http_https_or_file"}
            webbrowser.open(url)
            return {"ok": True, "opened": url}

        if action == "open_path":
            path = Path(str(kwargs.get("path") or "")).expanduser()
            if not path.exists():
                return {"ok": False, "error": "path_not_found", "path": str(path)}
            # Cross-platform open
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys_platform() == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
            return {"ok": True, "opened": str(path)}

        if action == "screenshot":
            out = Path(str(kwargs.get("path") or "data/complete_ai/screenshot.png"))
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                from PIL import ImageGrab  # type: ignore

                img = ImageGrab.grab()
                img.save(out)
                return {"ok": True, "path": str(out), "engine": "PIL.ImageGrab"}
            except Exception as exc:
                # Write a placeholder note instead of failing hard offline
                note = out.with_suffix(".txt")
                note.write_text(f"screenshot_unavailable:{exc}\n", encoding="utf-8")
                return {
                    "ok": False,
                    "error": f"screenshot_unavailable:{exc}",
                    "note_path": str(note),
                    "hint": "pip install Pillow (and a display) for screenshots",
                }

        if action == "notify":
            title = str(kwargs.get("title") or "Complete AI")
            message = str(kwargs.get("message") or "")
            # Best-effort desktop notify
            if shutil_which("notify-send"):
                subprocess.run(["notify-send", title, message], check=False)
                return {"ok": True, "engine": "notify-send", "title": title}
            return {"ok": True, "engine": "log_only", "title": title, "message": message}

        if action == "run_command":
            if not self.allow_shell:
                return {
                    "ok": False,
                    "error": "shell_disabled",
                    "hint": "Set COMPLETE_AI_COMPUTER_SHELL=1 and pass confirm=True",
                }
            if not kwargs.get("confirm"):
                return {"ok": False, "error": "confirm_required_for_shell"}
            cmd = kwargs.get("command")
            if not isinstance(cmd, (str, list)):
                return {"ok": False, "error": "command_required"}
            completed = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                capture_output=True,
                text=True,
                timeout=float(kwargs.get("timeout") or 30),
            )
            return {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": (completed.stdout or "")[:4000],
                "stderr": (completed.stderr or "")[:2000],
            }

        if action in ("type_text", "hotkey", "mouse_click"):
            if not kwargs.get("confirm"):
                return {"ok": False, "error": "confirm_required_for_input_injection"}
            try:
                import pyautogui  # type: ignore
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"pyautogui_unavailable:{exc}",
                    "hint": "pip install pyautogui",
                }
            if action == "type_text":
                pyautogui.typewrite(str(kwargs.get("text") or ""), interval=float(kwargs.get("interval") or 0.02))
                return {"ok": True, "typed": True}
            if action == "hotkey":
                keys = kwargs.get("keys") or []
                if isinstance(keys, str):
                    keys = keys.split("+")
                pyautogui.hotkey(*[str(k) for k in keys])
                return {"ok": True, "hotkey": list(keys)}
            x, y = kwargs.get("x"), kwargs.get("y")
            if x is None or y is None:
                return {"ok": False, "error": "x_y_required"}
            pyautogui.click(int(x), int(y), clicks=int(kwargs.get("clicks") or 1))
            return {"ok": True, "clicked": [int(x), int(y)]}

        return {"ok": False, "error": f"unhandled:{action}"}


def sys_platform() -> str:
    import sys

    return sys.platform


def shutil_which(cmd: str) -> Optional[str]:
    import shutil

    return shutil.which(cmd)


def _safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in kwargs.items():
        if k in ("password", "token", "secret"):
            out[k] = "***"
        else:
            out[k] = v if not isinstance(v, str) or len(v) < 500 else v[:500] + "…"
    return out
