"""Stdlib HTTP chat UI for the symbolic brain — no LLM, no external deps."""

from __future__ import annotations

import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from ..brain import Brain
from ..knowledge import Rule, Triple
from ..version import __version__

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatSession:
    """One brain instance shared by the UI server."""

    def __init__(self, domain: str = "animals") -> None:
        self.lock = threading.RLock()
        self.brain = Brain(name="ChatBrain")
        self.brain.load_domain(domain)
        self.brain.mark_booted()
        self.brain.reason()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self.brain.status()

    def greeting(self) -> str:
        with self.lock:
            st = self.brain.status()
            return (
                f"Cognitive OS chat ready (v{__version__}).\n"
                f"Domain: {st['domain']} · {st['facts']} facts · {st['rules']} rules.\n"
                "Ask me anything introspective, or use commands like "
                "`assert`, `goal`, `think`, `why`, `prove`, `query`, `domain`.\n"
                "I am symbolic — no language model behind this chat."
            )

    def handle_message(self, message: str) -> str:
        with self.lock:
            return self._dispatch(message.strip())

    def _dispatch(self, msg: str) -> str:
        if not msg:
            return "Say something, or try: Who are you?"

        lower = msg.lower()
        # Slash-free OS-ish commands
        if lower in ("think", "think 3"):
            return self._think(3)
        m = re.match(r"^think\s+(\d+)\s*$", lower)
        if m:
            return self._think(int(m.group(1)))

        if lower == "reason":
            trace = self.brain.reason()
            derived = "\n".join(f"  {d}" for d in trace.get("derived", [])[:15])
            return f"{trace['explanation']}\n{derived}".rstrip()

        if lower.startswith("domain "):
            name = msg.split(None, 1)[1].strip()
            self.brain.load_domain(name)
            self.brain.reason()
            return f"Domain → {name} ({len(self.brain.kb)} facts). Forward reasoning applied."

        if lower.startswith("assert "):
            parts = msg.split()
            if len(parts) < 4:
                return "usage: assert <s> <p> <o>"
            self.brain.assert_fact(parts[1], parts[2], parts[3], learn=True)
            return f"Asserted ({parts[1]} {parts[2]} {parts[3]})."

        if lower.startswith("goal "):
            raw = msg.split(None, 1)[1]
            if "|" not in raw:
                return "usage: goal <description> | <s> <p> <o>"
            desc, trip = raw.split("|", 1)
            t = trip.split()
            if len(t) < 3:
                return "Need subject predicate object after |"
            g = self.brain.set_goal(desc.strip(), t[0], t[1], t[2])
            return f"Goal {g.goal_id}: {g.description} → {g.target}"

        if lower.startswith("why "):
            parts = msg.split()
            if len(parts) < 4:
                return "usage: why <s> <p> <o>"
            return self.brain.why(parts[1], parts[2], parts[3])

        if lower.startswith("prove "):
            parts = msg.split()
            if len(parts) < 4:
                return "usage: prove <s> <p> <o>"
            proof = self.brain.prove(parts[1], parts[2], parts[3])
            return f"success={proof['success']}: {proof['explanation']}"

        if lower.startswith("query"):
            parts = msg.split()
            s = parts[1] if len(parts) > 1 else "*"
            p = parts[2] if len(parts) > 2 else "*"
            o = parts[3] if len(parts) > 3 else "*"
            hits = self.brain.kb.query(s, p, o)
            if not hits:
                return "(no matches)"
            return "\n".join(str(t) for t in hits[:40])

        if lower.startswith("rule "):
            raw = msg[5:].strip()
            if "|" not in raw or "=>" not in raw:
                return "usage: rule <name> | (a b c) => (d e f)"
            name_part, rest = raw.split("|", 1)
            ants_s, cons_s = rest.split("=>", 1)
            ants = [
                Triple(m.group(1), m.group(2), m.group(3))
                for m in re.finditer(r"\((\S+)\s+(\S+)\s+(\S+)\)", ants_s)
            ]
            cons = [
                Triple(m.group(1), m.group(2), m.group(3))
                for m in re.finditer(r"\((\S+)\s+(\S+)\s+(\S+)\)", cons_s)
            ]
            if not ants or not cons:
                return "Need antecedent and consequent triples."
            rule = Rule(
                name=name_part.strip() or "user_rule",
                antecedents=ants,
                consequents=cons,
                learned=True,
            )
            self.brain.add_rule(rule)
            self.brain.learner.learn_rule(rule)
            return f"Rule '{rule.name}' installed."

        if lower in ("status", "ps", "mem"):
            st = self.brain.status()
            goals = "\n".join(
                f"  - [{g['status']}] {g['description']} → {g['target']}"
                for g in st.get("goals", [])
            ) or "  (none)"
            return (
                f"domain={st['domain']} facts={st['facts']} beliefs={st['beliefs']} "
                f"rules={st['rules']} open_goals={st['open_goals']}\nGoals:\n{goals}"
            )

        # Default: introspective / world ask
        return self.brain.ask(msg)

    def _think(self, n: int) -> str:
        report = self.brain.think(steps=n)
        lines = [f"think → {report.stopped_reason} ({len(report.cycles)} cycles)"]
        for c in report.cycles:
            lines.append(f"  [{c.step}] {c.plan_summary}")
            for a in c.actions:
                lines.append(f"    act: {a}")
            for u in c.utterances:
                lines.append(f"    say: {u}")
        return "\n".join(lines)


def make_handler(session: ChatSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # quieter default logging
            if "/api/" in (args[0] if args else ""):
                return
            super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self._send(code, data, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._serve_file(STATIC_DIR / "index.html")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                target = (STATIC_DIR / rel).resolve()
                if not str(target).startswith(str(STATIC_DIR.resolve())):
                    self._json(403, {"error": "forbidden"})
                    return
                self._serve_file(target)
                return
            if path == "/api/status":
                self._json(200, session.status())
                return
            if path == "/api/hello":
                self._json(200, {"greeting": session.greeting(), "version": __version__})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON"})
                return

            if path == "/api/chat":
                message = str(payload.get("message", ""))
                try:
                    reply = session.handle_message(message)
                    self._json(200, {"reply": reply, "status": session.status()})
                except Exception as exc:  # noqa: BLE001
                    self._json(500, {"error": str(exc)})
                return
            self._json(404, {"error": "not found"})

        def _serve_file(self, path: Path) -> None:
            if not path.is_file():
                self._json(404, {"error": "file not found"})
                return
            data = path.read_bytes()
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            if path.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif path.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif path.suffix == ".html":
                ctype = "text/html; charset=utf-8"
            self._send(200, data, ctype)

    return Handler


def serve_chat(
    host: str = "0.0.0.0",
    port: int = 8765,
    domain: str = "animals",
) -> None:
    session = ChatSession(domain=domain)
    handler = make_handler(session)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Cognitive OS chat UI → http://127.0.0.1:{port}/  (domain={domain})")
    print("Symbolic brain only — no LLM. Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down chat UI.")
    finally:
        httpd.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Launch Cognitive OS chat UI")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--domain", default="animals")
    args = p.parse_args(argv)
    serve_chat(host=args.host, port=args.port, domain=args.domain)
    return 0
