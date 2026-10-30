"""Parse natural-language / CLI computer-control utterances into actions."""

from __future__ import annotations

import re
import shlex
from typing import Any


def parse_computer_command(text: str) -> dict[str, Any]:
    """Return {action, kwargs, raw} from assistant utterances.

    Accepts:
      computer status
      computer open_url https://example.com
      open https://example.com
      take a screenshot
      list files in data
      read file README.md
      launch calculator
      notify Hello world
      type Hello
      press ctrl+s
      click 100 200
      copy src.txt to dst.txt
      delete file tmp.txt
      env / screen / processes / audit
    """
    raw = (text or "").strip()
    lower = raw.lower()
    # Strip leading computer/control verbs
    lower = re.sub(r"^(?:computer|control|desktop|pc)\s+", "", lower).strip()
    raw_stripped = re.sub(r"^(?:computer|control|desktop|pc)\s+", "", raw, flags=re.I).strip()

    # Structured: action arg...
    try:
        parts = shlex.split(raw_stripped)
    except ValueError:
        parts = raw_stripped.split()

    if not parts:
        return {"action": "status", "kwargs": {}, "raw": raw}

    head = parts[0].lower()

    # Direct allowlisted verbs
    direct = {
        "status",
        "help",
        "env",
        "env_info",
        "screen",
        "screen_info",
        "audit",
        "audit_log",
        "policy",
        "cwd",
        "workspace",
        "processes",
        "list_processes",
        "screenshot",
        "notify",
        "open_url",
        "open_path",
        "list_dir",
        "ls",
        "read_file",
        "write_file",
        "mkdir",
        "copy",
        "copy_path",
        "move",
        "move_path",
        "delete",
        "delete_path",
        "launch",
        "launch_app",
        "apps",
        "list_apps",
        "type",
        "type_text",
        "hotkey",
        "press",
        "press_key",
        "click",
        "mouse_click",
        "move_mouse",
        "mouse_move",
        "scroll",
        "wait",
        "run",
        "run_command",
        "clipboard_get",
        "clipboard_set",
        "enable",
        "disable",
    }

    # Natural paraphrases
    if re.match(r"^(help|what can you do|commands)\b", lower):
        return {"action": "help", "kwargs": {}, "raw": raw}
    if re.search(r"\b(take\s+a\s+)?screenshot\b", lower) or lower in ("capture screen", "screen capture"):
        path = None
        m = re.search(r"(?:to|as|path)\s+(\S+)", raw_stripped, re.I)
        if m:
            path = m.group(1)
        elif len(parts) >= 2 and parts[0].lower() in ("screenshot", "capture"):
            path = parts[1]
        kw: dict[str, Any] = {}
        if path:
            kw["path"] = path
        return {"action": "screenshot", "kwargs": kw, "raw": raw}
    if re.match(r"^open\s+(https?://\S+)", lower) or re.match(r"^open_url\s+", lower):
        m = re.search(r"(https?://\S+|file:\S+)", raw_stripped, re.I)
        return {"action": "open_url", "kwargs": {"url": m.group(1) if m else parts[-1]}, "raw": raw}
    if re.match(r"^open\s+", lower) and not re.search(r"https?://", lower):
        # open path or launch app
        rest = raw_stripped.split(None, 1)[1] if " " in raw_stripped else ""
        if rest.lower() in ("calculator", "browser", "terminal", "text_editor", "file_manager"):
            return {"action": "launch_app", "kwargs": {"app": rest.lower()}, "raw": raw}
        return {"action": "open_path", "kwargs": {"path": rest}, "raw": raw}
    if re.match(r"^(list|ls|show)\s+(files|dir|directory|folder)\b", lower) or lower.startswith("list_dir"):
        m = re.search(r"(?:in|of|at)\s+(.+)$", raw_stripped, re.I)
        path = m.group(1).strip().strip("\"'") if m else (parts[1] if len(parts) > 1 and parts[0].lower() in ("list_dir", "ls") else ".")
        if parts[0].lower() in ("list", "ls", "show") and len(parts) > 2 and parts[1].lower() in ("files", "dir", "directory", "folder"):
            # "list files in X" already handled; else default
            pass
        return {"action": "list_dir", "kwargs": {"path": path or "."}, "raw": raw}
    if re.match(r"^(read|cat|show)\s+file\b", lower) or head == "read_file":
        m = re.search(r"file\s+(.+)$", raw_stripped, re.I)
        path = m.group(1).strip().strip("\"'") if m else (parts[1] if len(parts) > 1 else "")
        return {"action": "read_file", "kwargs": {"path": path}, "raw": raw}
    if re.match(r"^launch\b|^start\s+app\b|^open\s+app\b", lower) or head in ("launch", "launch_app"):
        app = parts[1] if len(parts) > 1 else ""
        if head in ("launch", "launch_app") and len(parts) > 1:
            app = parts[1]
        else:
            m = re.search(r"(?:launch|start app|open app)\s+(\w+)", lower)
            app = m.group(1) if m else app
        return {"action": "launch_app", "kwargs": {"app": app}, "raw": raw}
    if re.match(r"^notify\b|^send\s+notification\b", lower):
        msg = re.sub(r"^(?:notify|send\s+notification)\s*", "", raw_stripped, flags=re.I)
        return {"action": "notify", "kwargs": {"title": "Complete AI", "message": msg or "Hello"}, "raw": raw}
    if re.match(r"^type\b", lower) or head == "type_text":
        text_out = re.sub(r"^(?:type(?:_text)?)\s*", "", raw_stripped, flags=re.I)
        return {"action": "type_text", "kwargs": {"text": text_out, "confirm": True}, "raw": raw}
    if re.match(r"^(press|hotkey)\b", lower):
        keys = re.sub(r"^(?:press(?:_key)?|hotkey)\s*", "", raw_stripped, flags=re.I)
        action = "hotkey" if "+" in keys or " " in keys.strip() else "press_key"
        if action == "hotkey":
            return {"action": "hotkey", "kwargs": {"keys": keys.replace(" ", "+"), "confirm": True}, "raw": raw}
        return {"action": "press_key", "kwargs": {"key": keys, "confirm": True}, "raw": raw}
    if re.match(r"^click\b", lower) or head == "mouse_click":
        nums = re.findall(r"-?\d+", raw_stripped)
        if len(nums) >= 2:
            return {
                "action": "mouse_click",
                "kwargs": {"x": int(nums[0]), "y": int(nums[1]), "confirm": True},
                "raw": raw,
            }
        return {"action": "mouse_click", "kwargs": {"confirm": True}, "raw": raw}
    if re.match(r"^copy\b", lower) and " to " in lower:
        m = re.search(r"copy\s+(.+?)\s+to\s+(.+)$", raw_stripped, re.I)
        if m:
            return {
                "action": "copy_path",
                "kwargs": {"src": m.group(1).strip().strip("\"'"), "dst": m.group(2).strip().strip("\"'")},
                "raw": raw,
            }
    if re.match(r"^(delete|remove|rm)\b", lower) or head in ("delete", "delete_path"):
        m = re.search(r"(?:delete|remove|rm|delete_path)(?:\s+file)?\s+(.+)$", raw_stripped, re.I)
        path = m.group(1).strip().strip("\"'") if m else ""
        return {"action": "delete_path", "kwargs": {"path": path, "confirm": True}, "raw": raw}
    if re.match(r"^mkdir\b|^make\s+(dir|directory|folder)\b", lower):
        m = re.search(r"(?:mkdir|make\s+(?:dir|directory|folder))\s+(.+)$", raw_stripped, re.I)
        return {"action": "mkdir", "kwargs": {"path": (m.group(1) if m else "").strip().strip("\"'")}, "raw": raw}
    if re.match(r"^write\s+file\b|^write_file\b", lower):
        # write_file PATH :: CONTENT
        if "::" in raw_stripped:
            left, right = raw_stripped.split("::", 1)
            path = re.sub(r"^(?:write(?:_file)?(?:\s+file)?)\s*", "", left, flags=re.I).strip()
            return {
                "action": "write_file",
                "kwargs": {"path": path, "content": right.lstrip(), "confirm": True},
                "raw": raw,
            }
    if re.match(r"^(run|shell|exec)\b", lower) or head == "run_command":
        cmd = re.sub(r"^(?:run(?:_command)?|shell|exec)\s*", "", raw_stripped, flags=re.I)
        return {"action": "run_command", "kwargs": {"command": cmd, "confirm": True}, "raw": raw}
    if "clipboard" in lower and any(w in lower for w in ("get", "read", "show", "what")):
        return {"action": "clipboard_get", "kwargs": {"confirm": True}, "raw": raw}
    if "clipboard" in lower and any(w in lower for w in ("set", "put", "write", "copy text")):
        m = re.search(r"clipboard(?:\s+set)?\s+(.+)$", raw_stripped, re.I)
        content = m.group(1) if m else ""
        return {"action": "clipboard_set", "kwargs": {"text": content, "confirm": True}, "raw": raw}
    if re.match(r"^(env|environment|system info)\b", lower):
        return {"action": "env_info", "kwargs": {}, "raw": raw}
    if re.match(r"^(screen|display)\b", lower):
        return {"action": "screen_info", "kwargs": {}, "raw": raw}
    if "process" in lower:
        return {"action": "list_processes", "kwargs": {}, "raw": raw}
    if "audit" in lower:
        return {"action": "audit_log", "kwargs": {}, "raw": raw}
    if "policy" in lower or "sandbox" in lower:
        return {"action": "policy", "kwargs": {}, "raw": raw}
    if lower in ("enable", "enable control", "turn on"):
        return {"action": "enable", "kwargs": {}, "raw": raw}
    if lower in ("disable", "disable control", "turn off"):
        return {"action": "disable", "kwargs": {}, "raw": raw}

    # Fallback: treat first token as action
    action = head if head in direct else "help"
    kwargs: dict[str, Any] = {}
    if action in ("open_url",) and len(parts) > 1:
        kwargs["url"] = parts[1]
    elif action in ("open_path", "list_dir", "ls", "read_file", "mkdir") and len(parts) > 1:
        kwargs["path"] = parts[1]
        if action == "ls":
            action = "list_dir"
    elif action in ("launch", "launch_app") and len(parts) > 1:
        action = "launch_app"
        kwargs["app"] = parts[1]
    elif action == "notify":
        kwargs["title"] = "Complete AI"
        kwargs["message"] = " ".join(parts[1:]) or "Hello"
    elif action == "screenshot" and len(parts) > 1:
        kwargs["path"] = parts[1]
    elif action in ("run", "run_command"):
        action = "run_command"
        kwargs["command"] = " ".join(parts[1:])
        kwargs["confirm"] = True
    elif action in ("type", "type_text"):
        action = "type_text"
        kwargs["text"] = " ".join(parts[1:])
        kwargs["confirm"] = True
    elif action in ("copy", "copy_path") and len(parts) >= 3:
        action = "copy_path"
        kwargs["src"] = parts[1]
        kwargs["dst"] = parts[2]
    elif action in ("move", "move_path") and len(parts) >= 3:
        action = "move_path"
        kwargs["src"] = parts[1]
        kwargs["dst"] = parts[2]
    elif action in ("delete", "delete_path") and len(parts) > 1:
        action = "delete_path"
        kwargs["path"] = parts[1]
        kwargs["confirm"] = True
    elif action in ("env",):
        action = "env_info"
    elif action in ("screen",):
        action = "screen_info"
    elif action in ("processes",):
        action = "list_processes"
    elif action in ("audit",):
        action = "audit_log"
    elif action in ("apps",):
        action = "list_apps"
    elif action in ("workspace",):
        action = "cwd"

    return {"action": action, "kwargs": kwargs, "raw": raw}
