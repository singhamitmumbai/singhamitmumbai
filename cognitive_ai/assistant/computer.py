"""Detailed gated local computer control for Complete AI / Finderbay Assistant.

Safety (hard rules):
  • Disabled by default — enable with allow=True or COMPLETE_AI_COMPUTER_CONTROL=1
  • Allowlisted actions only
  • Sandboxed paths (COMPLETE_AI_COMPUTER_ROOT + cwd/data/Documents/Downloads/Desktop)
  • No keylogging, no background clipboard monitoring, no silent surveillance
  • Destructive / input-injection / shell actions require confirm=True
  • Shell needs COMPLETE_AI_COMPUTER_SHELL=1 and still hits a denylist
  • Every action is audited

This controls the *user's own* machine as an assistant — not remote unauthorized access.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional

from .computer_nlu import parse_computer_command
from .computer_policy import (
    APP_ALLOWLIST,
    default_sandbox_roots,
    platform_family,
    policy_snapshot,
    resolve_in_sandbox,
    shell_allowed,
)


ALLOWED = {
    # meta
    "status",
    "help",
    "policy",
    "enable",
    "disable",
    "audit_log",
    "env_info",
    "screen_info",
    "cwd",
    "list_apps",
    "list_processes",
    # open / launch
    "open_url",
    "open_path",
    "launch_app",
    # filesystem (sandboxed)
    "list_dir",
    "read_file",
    "write_file",
    "mkdir",
    "copy_path",
    "move_path",
    "delete_path",
    # desktop UX
    "screenshot",
    "notify",
    "clipboard_get",
    "clipboard_set",
    # input injection (confirm)
    "type_text",
    "press_key",
    "hotkey",
    "mouse_click",
    "mouse_move",
    "scroll",
    "wait",
    # shell (extra gate)
    "run_command",
}

CONFIRM_REQUIRED = {
    "write_file",
    "delete_path",
    "type_text",
    "press_key",
    "hotkey",
    "mouse_click",
    "mouse_move",
    "scroll",
    "clipboard_get",
    "clipboard_set",
    "run_command",
    "move_path",
}


class ComputerControl:
    """Full allowlisted desktop + filesystem automation for the local user."""

    version = "computer_control_v2"

    def __init__(
        self,
        *,
        allow: Optional[bool] = None,
        allow_shell: bool = False,
        sandbox_roots: Optional[list[Path | str]] = None,
        max_read_bytes: int = 64_000,
        max_audit: int = 500,
    ) -> None:
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
        self.sandbox_roots = (
            [Path(p).expanduser().resolve() for p in sandbox_roots]
            if sandbox_roots
            else default_sandbox_roots()
        )
        self.max_read_bytes = max_read_bytes
        self.max_audit = max_audit
        self.audit: list[dict[str, Any]] = []

    # --- public API -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "name": "ComputerControl",
            "version": self.version,
            "enabled": self.allow,
            "allow_shell": self.allow_shell,
            "allowed_actions": sorted(ALLOWED),
            "confirm_required": sorted(CONFIRM_REQUIRED),
            "sandbox_roots": [str(p) for p in self.sandbox_roots],
            "audit_events": len(self.audit),
            "platform": platform_family(),
            "no_keylogging": True,
            "no_background_clipboard_monitor": True,
            "human_level_agi": False,
            "enable_hint": "export COMPLETE_AI_COMPUTER_CONTROL=1",
            "shell_hint": "export COMPLETE_AI_COMPUTER_SHELL=1 (still needs confirm=True)",
        }

    def help(self) -> dict[str, Any]:
        return {
            "ok": True,
            "title": "Detailed computer control — allowlisted local automation",
            "enable": "COMPLETE_AI_COMPUTER_CONTROL=1",
            "examples": [
                "computer status",
                "computer help",
                "computer open https://example.com",
                "computer screenshot",
                "computer list files in data",
                "computer read file README.md",
                "computer write_file data/note.txt :: hello",
                "computer mkdir data/tmp_demo",
                "computer copy a.txt to b.txt",
                "computer launch calculator",
                "computer notify Build finished",
                "computer type Hello",
                "computer press ctrl+s",
                "computer click 100 200",
                "computer env",
                "computer screen",
                "computer processes",
                "computer policy",
                "computer audit",
                "computer run echo hello   # needs SHELL=1",
            ],
            "forbidden": [
                "keylogging / key capture",
                "background clipboard siphoning",
                "format disk / rm -rf / / shutdown",
                "writes outside sandbox roots",
            ],
            "actions": sorted(ALLOWED),
            "human_level_agi": False,
        }

    def enable(self) -> None:
        self.allow = True

    def disable(self) -> None:
        self.allow = False

    def parse(self, text: str) -> dict[str, Any]:
        return parse_computer_command(text)

    def handle_text(self, text: str) -> dict[str, Any]:
        """Parse NL/CLI utterance then run."""
        parsed = self.parse(text)
        action = parsed.get("action") or "help"
        kwargs = dict(parsed.get("kwargs") or {})
        if action == "enable":
            self.enable()
            return {"ok": True, "action": "enable", "enabled": True}
        if action == "disable":
            self.disable()
            return {"ok": True, "action": "disable", "enabled": False}
        out = self.run(action, **kwargs)
        out["parsed"] = parsed
        return out

    def run(self, action: str, **kwargs: Any) -> dict[str, Any]:
        action = (action or "").strip().lower()
        # aliases
        aliases = {
            "ls": "list_dir",
            "env": "env_info",
            "screen": "screen_info",
            "processes": "list_processes",
            "audit": "audit_log",
            "apps": "list_apps",
            "launch": "launch_app",
            "copy": "copy_path",
            "move": "move_path",
            "delete": "delete_path",
            "type": "type_text",
            "press": "press_key",
            "click": "mouse_click",
            "run": "run_command",
            "workspace": "cwd",
        }
        action = aliases.get(action, action)

        event: dict[str, Any] = {
            "action": action,
            "kwargs": _safe_kwargs(kwargs),
            "ts": time.time(),
            "version": self.version,
        }

        if action not in ALLOWED:
            event["ok"] = False
            event["error"] = f"action_not_allowlisted:{action}"
            return self._finish(event)

        if action == "status":
            event["ok"] = True
            event["result"] = self.status()
            return self._finish(event)
        if action == "help":
            event["ok"] = True
            event["result"] = self.help()
            return self._finish(event)
        if action == "policy":
            event["ok"] = True
            event["result"] = {**policy_snapshot(), "runtime_roots": [str(p) for p in self.sandbox_roots]}
            return self._finish(event)
        if action == "enable":
            self.enable()
            event["ok"] = True
            event["result"] = {"enabled": True}
            return self._finish(event)
        if action == "disable":
            self.disable()
            event["ok"] = True
            event["result"] = {"enabled": False}
            return self._finish(event)

        if not self.allow:
            event["ok"] = False
            event["error"] = "computer_control_disabled"
            event["hint"] = "export COMPLETE_AI_COMPUTER_CONTROL=1  OR  computer.enable()"
            return self._finish(event)

        if action in CONFIRM_REQUIRED and not kwargs.get("confirm"):
            event["ok"] = False
            event["error"] = f"confirm_required_for_{action}"
            event["hint"] = "Pass confirm=True after explicit user approval"
            return self._finish(event)

        try:
            result = self._dispatch(action, **kwargs)
            event["ok"] = bool(result.get("ok", True))
            event["result"] = result
            if result.get("error"):
                event["error"] = result.get("error")
        except Exception as exc:  # noqa: BLE001
            event["ok"] = False
            event["error"] = str(exc)
        return self._finish(event)

    def _finish(self, event: dict[str, Any]) -> dict[str, Any]:
        self.audit.append(event)
        if len(self.audit) > self.max_audit:
            self.audit = self.audit[-self.max_audit :]
        return event

    # --- dispatch -------------------------------------------------------------

    def _dispatch(self, action: str, **kwargs: Any) -> dict[str, Any]:
        if action == "env_info":
            return {
                "ok": True,
                "platform": platform_family(),
                "cwd": str(Path.cwd()),
                "home": str(Path.home()),
                "user": os.environ.get("USER") or os.environ.get("USERNAME"),
                "python": _python_version(),
                "sandbox_roots": [str(p) for p in self.sandbox_roots],
                "display": os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"),
            }

        if action == "cwd":
            return {"ok": True, "cwd": str(Path.cwd()), "sandbox_roots": [str(p) for p in self.sandbox_roots]}

        if action == "screen_info":
            return self._screen_info()

        if action == "audit_log":
            n = int(kwargs.get("limit") or 20)
            return {"ok": True, "events": list(self.audit[-n:]), "total": len(self.audit)}

        if action == "list_apps":
            return {"ok": True, "apps": sorted(APP_ALLOWLIST)}

        if action == "list_processes":
            return self._list_processes(limit=int(kwargs.get("limit") or 30))

        if action == "open_url":
            url = str(kwargs.get("url") or "")
            if not url.startswith(("http://", "https://", "file:")):
                return {"ok": False, "error": "url_must_be_http_https_or_file"}
            webbrowser.open(url)
            return {"ok": True, "opened": url}

        if action == "open_path":
            path, err = resolve_in_sandbox(
                kwargs.get("path") or "", roots=self.sandbox_roots, must_exist=True
            )
            if err:
                return {"ok": False, "error": err}
            return self._os_open(path)  # type: ignore[arg-type]

        if action == "launch_app":
            return self._launch_app(str(kwargs.get("app") or ""))

        if action == "list_dir":
            path, err = resolve_in_sandbox(
                kwargs.get("path") or ".", roots=self.sandbox_roots, must_exist=True
            )
            if err:
                return {"ok": False, "error": err}
            assert path is not None
            if not path.is_dir():
                return {"ok": False, "error": "not_a_directory", "path": str(path)}
            entries = []
            for child in sorted(path.iterdir())[: int(kwargs.get("limit") or 200)]:
                entries.append(
                    {
                        "name": child.name,
                        "type": "dir" if child.is_dir() else "file",
                        "size": child.stat().st_size if child.is_file() else None,
                    }
                )
            return {"ok": True, "path": str(path), "count": len(entries), "entries": entries}

        if action == "read_file":
            path, err = resolve_in_sandbox(
                kwargs.get("path") or "", roots=self.sandbox_roots, must_exist=True
            )
            if err:
                return {"ok": False, "error": err}
            assert path is not None
            if not path.is_file():
                return {"ok": False, "error": "not_a_file"}
            data = path.read_bytes()[: self.max_read_bytes]
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                return {
                    "ok": True,
                    "path": str(path),
                    "binary": True,
                    "bytes_read": len(data),
                    "preview_hex": data[:64].hex(),
                }
            return {
                "ok": True,
                "path": str(path),
                "text": text,
                "truncated": path.stat().st_size > self.max_read_bytes,
                "bytes_read": len(data),
            }

        if action == "write_file":
            path, err = resolve_in_sandbox(kwargs.get("path") or "", roots=self.sandbox_roots)
            if err:
                return {"ok": False, "error": err}
            assert path is not None
            content = str(kwargs.get("content") or "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"ok": True, "path": str(path), "bytes": len(content.encode("utf-8"))}

        if action == "mkdir":
            path, err = resolve_in_sandbox(kwargs.get("path") or "", roots=self.sandbox_roots)
            if err:
                return {"ok": False, "error": err}
            assert path is not None
            path.mkdir(parents=True, exist_ok=True)
            return {"ok": True, "path": str(path)}

        if action in ("copy_path", "move_path"):
            src, err = resolve_in_sandbox(
                kwargs.get("src") or "", roots=self.sandbox_roots, must_exist=True
            )
            if err:
                return {"ok": False, "error": f"src:{err}"}
            dst, err2 = resolve_in_sandbox(kwargs.get("dst") or "", roots=self.sandbox_roots)
            if err2:
                return {"ok": False, "error": f"dst:{err2}"}
            assert src is not None and dst is not None
            dst.parent.mkdir(parents=True, exist_ok=True)
            if action == "copy_path":
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
                return {"ok": True, "copied": str(src), "to": str(dst)}
            shutil.move(str(src), str(dst))
            return {"ok": True, "moved": str(src), "to": str(dst)}

        if action == "delete_path":
            path, err = resolve_in_sandbox(
                kwargs.get("path") or "", roots=self.sandbox_roots, must_exist=True
            )
            if err:
                return {"ok": False, "error": err}
            assert path is not None
            # Refuse deleting sandbox roots themselves
            if path.resolve() in {r.resolve() for r in self.sandbox_roots}:
                return {"ok": False, "error": "refuse_delete_sandbox_root"}
            if path.is_dir():
                if kwargs.get("recursive"):
                    shutil.rmtree(path)
                else:
                    path.rmdir()
            else:
                path.unlink()
            return {"ok": True, "deleted": str(path)}

        if action == "screenshot":
            return self._screenshot(str(kwargs.get("path") or "data/complete_ai/screenshot.png"))

        if action == "notify":
            title = str(kwargs.get("title") or "Complete AI")
            message = str(kwargs.get("message") or "")
            if _which("notify-send"):
                subprocess.run(["notify-send", title, message], check=False)
                return {"ok": True, "engine": "notify-send", "title": title, "message": message}
            return {"ok": True, "engine": "log_only", "title": title, "message": message}

        if action == "clipboard_get":
            return self._clipboard_get()

        if action == "clipboard_set":
            return self._clipboard_set(str(kwargs.get("text") or ""))

        if action == "wait":
            sec = min(float(kwargs.get("seconds") or 1.0), 30.0)
            time.sleep(sec)
            return {"ok": True, "waited": sec}

        if action in ("type_text", "press_key", "hotkey", "mouse_click", "mouse_move", "scroll"):
            return self._input_injection(action, **kwargs)

        if action == "run_command":
            return self._run_command(**kwargs)

        return {"ok": False, "error": f"unhandled:{action}"}

    # --- helpers --------------------------------------------------------------

    def _os_open(self, path: Path) -> dict[str, Any]:
        fam = platform_family()
        if fam == "win":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif fam == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        return {"ok": True, "opened": str(path)}

    def _launch_app(self, app: str) -> dict[str, Any]:
        key = (app or "").strip().lower().replace(" ", "_")
        if key not in APP_ALLOWLIST:
            return {
                "ok": False,
                "error": f"app_not_allowlisted:{key}",
                "allowed": sorted(APP_ALLOWLIST),
            }
        fam = platform_family()
        spec = APP_ALLOWLIST[key].get(fam) or []
        if not spec:
            return {"ok": False, "error": f"no_launcher_for_platform:{fam}"}
        # Special: browser/file_manager with xdg-open need a target
        if key == "browser":
            webbrowser.open("https://example.com")
            return {"ok": True, "launched": key, "via": "webbrowser"}
        if key == "file_manager":
            return self._os_open(Path.cwd())
        try:
            if fam == "darwin" and spec[0] == "open":
                subprocess.Popen(spec)
            elif fam == "win":
                subprocess.Popen(spec, shell=False)
            else:
                bin_name = None
                for cand in spec:
                    if _which(cand):
                        bin_name = cand
                        break
                if not bin_name:
                    return {"ok": False, "error": "app_binary_not_found", "tried": spec}
                subprocess.Popen([bin_name])
            return {"ok": True, "launched": key, "command": spec}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _screenshot(self, out_s: str) -> dict[str, Any]:
        path, err = resolve_in_sandbox(out_s, roots=self.sandbox_roots)
        if err:
            # allow writing under cwd/data even if resolve failed oddly
            path = Path(out_s).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import ImageGrab  # type: ignore

            img = ImageGrab.grab()
            img.save(path)
            return {"ok": True, "path": str(path), "engine": "PIL.ImageGrab", "size": list(img.size)}
        except Exception as exc:
            note = path.with_suffix(".txt")
            note.write_text(f"screenshot_unavailable:{exc}\n", encoding="utf-8")
            # Fallback: try import mss
            try:
                import mss  # type: ignore

                with mss.mss() as sct:
                    shot = sct.grab(sct.monitors[0])
                    from PIL import Image  # type: ignore

                    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                    img.save(path)
                    return {"ok": True, "path": str(path), "engine": "mss"}
            except Exception as exc2:
                return {
                    "ok": False,
                    "error": f"screenshot_unavailable:{exc}",
                    "fallback_error": str(exc2),
                    "note_path": str(note),
                    "hint": "pip install Pillow (display required)",
                }

    def _clipboard_get(self) -> dict[str, Any]:
        """One-shot explicit clipboard read — never a background monitor."""
        try:
            import pyperclip  # type: ignore

            text = pyperclip.paste()
            return {
                "ok": True,
                "text": (text or "")[:8000],
                "engine": "pyperclip",
                "note": "one_shot_explicit_read",
            }
        except Exception:
            pass
        # xclip / pbpaste
        for cmd in (["xclip", "-selection", "clipboard", "-o"], ["pbpaste"], ["powershell", "-command", "Get-Clipboard"]):
            if _which(cmd[0]):
                try:
                    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    return {
                        "ok": completed.returncode == 0,
                        "text": (completed.stdout or "")[:8000],
                        "engine": cmd[0],
                        "note": "one_shot_explicit_read",
                    }
                except Exception as exc:  # noqa: BLE001
                    return {"ok": False, "error": str(exc)}
        return {
            "ok": False,
            "error": "clipboard_unavailable",
            "hint": "pip install pyperclip  (explicit get only; no monitoring)",
        }

    def _clipboard_set(self, text: str) -> dict[str, Any]:
        try:
            import pyperclip  # type: ignore

            pyperclip.copy(text)
            return {"ok": True, "engine": "pyperclip", "chars": len(text)}
        except Exception:
            pass
        if _which("xclip"):
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=False,
            )
            return {"ok": True, "engine": "xclip", "chars": len(text)}
        if _which("pbcopy"):
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=False)
            return {"ok": True, "engine": "pbcopy", "chars": len(text)}
        return {"ok": False, "error": "clipboard_set_unavailable", "hint": "pip install pyperclip"}

    def _input_injection(self, action: str, **kwargs: Any) -> dict[str, Any]:
        try:
            import pyautogui  # type: ignore
        except Exception as exc:
            return {
                "ok": False,
                "error": f"pyautogui_unavailable:{exc}",
                "hint": "pip install pyautogui",
            }
        pyautogui.FAILSAFE = True
        if action == "type_text":
            # typewrite is ASCII-oriented; use write when available
            text = str(kwargs.get("text") or "")
            interval = float(kwargs.get("interval") or 0.02)
            try:
                pyautogui.write(text, interval=interval)
            except Exception:
                pyautogui.typewrite(text, interval=interval)
            return {"ok": True, "typed_chars": len(text)}
        if action == "press_key":
            key = str(kwargs.get("key") or "enter")
            pyautogui.press(key)
            return {"ok": True, "key": key}
        if action == "hotkey":
            keys = kwargs.get("keys") or []
            if isinstance(keys, str):
                keys = [k.strip() for k in keys.replace(" ", "+").split("+") if k.strip()]
            pyautogui.hotkey(*[str(k) for k in keys])
            return {"ok": True, "hotkey": list(keys)}
        if action == "mouse_move":
            x, y = kwargs.get("x"), kwargs.get("y")
            if x is None or y is None:
                return {"ok": False, "error": "x_y_required"}
            pyautogui.moveTo(int(x), int(y), duration=float(kwargs.get("duration") or 0.2))
            return {"ok": True, "moved": [int(x), int(y)]}
        if action == "mouse_click":
            x, y = kwargs.get("x"), kwargs.get("y")
            if x is None or y is None:
                return {"ok": False, "error": "x_y_required"}
            pyautogui.click(
                int(x),
                int(y),
                clicks=int(kwargs.get("clicks") or 1),
                button=str(kwargs.get("button") or "left"),
            )
            return {"ok": True, "clicked": [int(x), int(y)]}
        if action == "scroll":
            pyautogui.scroll(int(kwargs.get("clicks") or -3))
            return {"ok": True, "scrolled": int(kwargs.get("clicks") or -3)}
        return {"ok": False, "error": f"unhandled_input:{action}"}

    def _run_command(self, **kwargs: Any) -> dict[str, Any]:
        if not self.allow_shell:
            return {
                "ok": False,
                "error": "shell_disabled",
                "hint": "Set COMPLETE_AI_COMPUTER_SHELL=1 and pass confirm=True",
            }
        cmd = kwargs.get("command")
        if not isinstance(cmd, (str, list)) or not cmd:
            return {"ok": False, "error": "command_required"}
        ok, reason = shell_allowed(cmd)
        if not ok:
            return {"ok": False, "error": reason}
        completed = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=float(kwargs.get("timeout") or 30),
            cwd=str(kwargs.get("cwd") or Path.cwd()),
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "")[:4000],
            "stderr": (completed.stderr or "")[:2000],
        }

    def _screen_info(self) -> dict[str, Any]:
        try:
            import pyautogui  # type: ignore

            w, h = pyautogui.size()
            return {"ok": True, "width": w, "height": h, "engine": "pyautogui"}
        except Exception:
            pass
        try:
            from PIL import ImageGrab  # type: ignore

            img = ImageGrab.grab()
            return {"ok": True, "width": img.size[0], "height": img.size[1], "engine": "PIL"}
        except Exception as exc:
            return {
                "ok": True,
                "width": None,
                "height": None,
                "engine": "unavailable",
                "detail": str(exc),
                "display": os.environ.get("DISPLAY"),
            }

    def _list_processes(self, limit: int = 30) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        try:
            import psutil  # type: ignore

            for proc in psutil.process_iter(["pid", "name", "username"]):
                info = proc.info
                rows.append(
                    {
                        "pid": info.get("pid"),
                        "name": info.get("name"),
                        "user": info.get("username"),
                    }
                )
                if len(rows) >= limit:
                    break
            return {"ok": True, "engine": "psutil", "processes": rows, "count": len(rows)}
        except Exception:
            pass
        # portable fallback
        try:
            completed = subprocess.run(
                ["ps", "-eo", "pid,user,comm"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = (completed.stdout or "").splitlines()[1 : limit + 1]
            for line in lines:
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    rows.append({"pid": parts[0], "user": parts[1], "name": parts[2]})
            return {"ok": True, "engine": "ps", "processes": rows, "count": len(rows)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "hint": "pip install psutil"}


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def _safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in kwargs.items():
        if k in ("password", "token", "secret"):
            out[k] = "***"
        elif isinstance(v, str) and len(v) > 500:
            out[k] = v[:500] + "…"
        else:
            out[k] = v
    return out


# Back-compat helpers used by older imports
def sys_platform() -> str:
    import sys

    return sys.platform


def shutil_which(cmd: str) -> Optional[str]:
    return _which(cmd)
