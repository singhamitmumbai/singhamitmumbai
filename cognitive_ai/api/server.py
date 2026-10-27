"""REST API for Complete AI / Finderbay — stdlib only."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from ..complete_ai import CompleteAI
from ..general.agent import GeneralNonLLMAI
from ..system import CompleteNonLLMAI
from ..version import __version__


class APIState:
    def __init__(self, mode: str = "complete") -> None:
        self.lock = threading.RLock()
        self.mode = mode
        self.complete: Any = None
        self.general: Any = None
        self.ai: Any = None
        self.assistant: Any = None
        if mode in ("complete", "complete-ai", "assistant"):
            self.complete = CompleteAI(seed_demo=True)
            self.general = self.complete.general
            self.ai = self.complete.finderbay
            if mode == "assistant":
                from ..assistant import FinderbayAssistant

                self.assistant = FinderbayAssistant(complete=self.complete, seed_demo=True)
        elif mode == "finderbay":
            self.ai = CompleteNonLLMAI(seed_demo=True)
        else:
            self.general = GeneralNonLLMAI(seed_demo=True)
            self.ai = self.general.task_ai


def serve_api(host: str = "127.0.0.1", port: int = 8787, *, mode: str = "complete") -> ThreadingHTTPServer:
    state = APIState(mode=mode)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quieter
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            with state.lock:
                if path in ("/", "/health"):
                    return self._json(
                        200,
                        {
                            "ok": True,
                            "product": "Complete AI",
                            "version": __version__,
                            "mode": state.mode,
                            "llm_at_runtime": bool(
                                state.complete and state.complete.llm.available
                            ),
                            "human_level_agi": False,
                        },
                    )
                if path == "/status":
                    if state.complete is not None:
                        return self._json(200, state.complete.status())
                    if state.general is not None:
                        return self._json(200, state.general.status())
                    return self._json(200, state.ai.status())
                if path == "/diagnostics":
                    if state.complete is None:
                        return self._json(503, {"error": "complete_ai_required"})
                    return self._json(200, state.complete.diagnostics())
                if path == "/completeness":
                    if state.complete is None:
                        return self._json(503, {"error": "complete_ai_required"})
                    return self._json(200, state.complete.completeness())
                if path == "/session":
                    if state.complete is None:
                        return self._json(503, {"error": "complete_ai_required"})
                    return self._json(
                        200,
                        {
                            "turns": state.complete.session_transcript(),
                            "memory": state.complete.memory_snapshot(),
                        },
                    )
                if path == "/alerts":
                    project = (qs.get("project_id") or [None])[0]
                    return self._json(200, state.ai.alerts.summary(project_id=project))
                if path == "/reports":
                    kind = (qs.get("kind") or ["invoices"])[0]
                    project = (qs.get("project_id") or [None])[0]
                    fmt = (qs.get("format") or ["json"])[0]
                    try:
                        payload = state.ai.reports.export(kind, project_id=project, fmt=fmt)
                        return self._json(200, payload)
                    except ValueError as exc:
                        return self._json(400, {"error": str(exc)})
                if path == "/vendors":
                    return self._json(200, {"vendors": state.ai.purchase_orders.list_vendors()})
                if path == "/purchase_orders":
                    project = (qs.get("project_id") or [None])[0]
                    return self._json(
                        200,
                        {
                            "purchase_orders": state.ai.purchase_orders.list_pos(project_id=project),
                            "summary": state.ai.purchase_orders.summarize(project_id=project),
                        },
                    )
                if path in ("/vision", "/vision/status"):
                    vision = getattr(state.ai, "vision", None)
                    if vision is None:
                        return self._json(503, {"error": "vision_unavailable"})
                    return self._json(
                        200,
                        {
                            "status": vision.status(),
                            "projects": vision.projects(),
                            "llm_at_runtime": False,
                            "human_level_agi": False,
                        },
                    )
            return self._json(404, {"error": "not_found", "path": path})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            data = self._read_json()
            with state.lock:
                if path in ("/ask", "/handle", "/complete", "/assistant"):
                    text = data.get("text") or data.get("message") or ""
                    if not text and not data.get("audio"):
                        return self._json(400, {"error": "text_required"})
                    mode = data.get("mode") or "auto"
                    if state.assistant is not None and path == "/assistant":
                        out = state.assistant.handle(
                            text or "",
                            mode=mode,
                            audio=data.get("audio"),
                            speak=bool(data.get("speak")),
                        )
                    elif state.complete is not None and data.get("route") != "finderbay":
                        out = state.complete.handle(text, mode=mode)
                    elif state.general is not None and data.get("route") != "finderbay":
                        out = state.general.handle(text)
                    else:
                        out = state.ai.handle(text)
                    return self._json(200, out)
                if path == "/computer":
                    if state.assistant is None:
                        from ..assistant import FinderbayAssistant

                        state.assistant = FinderbayAssistant(
                            complete=state.complete or CompleteAI(seed_demo=True),
                            seed_demo=True,
                        )
                    action = data.get("action") or "status"
                    args = {k: v for k, v in data.items() if k != "action"}
                    return self._json(200, state.assistant.control(action, **args))
                if path == "/speak":
                    if state.assistant is None:
                        from ..assistant import FinderbayAssistant

                        state.assistant = FinderbayAssistant(
                            complete=state.complete or CompleteAI(seed_demo=True),
                            seed_demo=True,
                        )
                    return self._json(200, state.assistant.speak(str(data.get("text") or "")))

                if path == "/session/reset":
                    if state.complete is None:
                        return self._json(503, {"error": "complete_ai_required"})
                    state.complete.reset_session()
                    return self._json(200, {"ok": True, "reset": True})
                if path == "/session/export":
                    if state.complete is None:
                        return self._json(503, {"error": "complete_ai_required"})
                    fmt = data.get("format") or "markdown"
                    return self._json(
                        200,
                        {
                            "format": fmt,
                            "content": state.complete.export_session(fmt=fmt),
                        },
                    )
                if path == "/completeness":
                    if state.complete is None:
                        return self._json(503, {"error": "complete_ai_required"})
                    return self._json(200, state.complete.completeness())
                if path == "/memory":
                    if state.complete is None:
                        return self._json(503, {"error": "complete_ai_required"})
                    key = data.get("key")
                    value = data.get("value")
                    if not key:
                        return self._json(400, {"error": "key_required"})
                    state.complete.remember(str(key), str(value or ""))
                    return self._json(200, {"ok": True, "memory": state.complete.memory_snapshot()})
                if path == "/login":
                    user = data.get("username")
                    pwd = data.get("password")
                    tok = state.ai.auth.login(user, pwd)
                    if not tok:
                        return self._json(401, {"error": "auth_failed"})
                    state.ai.executor.auth_token = tok
                    return self._json(200, {"token": tok, "logged_in": True})
                if path == "/vision/analyze":
                    vision = getattr(state.ai, "vision", None)
                    if vision is None:
                        return self._json(503, {"error": "vision_unavailable"})
                    image = data.get("image") or data.get("image_path") or data.get("path")
                    task = data.get("task") or "detect_objects"
                    if task == "vision_status":
                        return self._json(200, vision.status())
                    if not image:
                        return self._json(400, {"error": "image_path_required"})
                    if data.get("ask"):
                        target = state.complete or state.ai
                        out = target.handle(str(data["ask"]))
                        return self._json(200, out)
                    out = vision.analyze(image, task=task)
                    return self._json(200, out)
                if path == "/tool":
                    if state.complete is None:
                        return self._json(503, {"error": "complete_ai_required"})
                    name = data.get("name") or data.get("tool")
                    if not name:
                        return self._json(400, {"error": "tool_name_required"})
                    return self._json(200, state.complete.tools.run(str(name), data.get("args") or {}))
            return self._json(404, {"error": "not_found", "path": path})

    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Complete AI REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument(
        "--mode",
        choices=["complete", "complete-ai", "assistant", "general", "finderbay"],
        default="complete",
    )
    args = p.parse_args(argv)
    httpd = serve_api(args.host, args.port, mode=args.mode)
    print(
        f"Complete AI API http://{args.host}:{args.port}/  "
        f"(v{__version__}, mode={args.mode}, human_level_agi=false)"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
