"""Minimal REST API over CompleteNonLLMAI / GeneralNonLLMAI — stdlib only."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from ..general.agent import GeneralNonLLMAI
from ..system import CompleteNonLLMAI
from ..version import __version__


class APIState:
    def __init__(self, mode: str = "general") -> None:
        self.lock = threading.RLock()
        self.mode = mode
        if mode == "finderbay":
            self.ai: Any = CompleteNonLLMAI(seed_demo=True)
            self.general = None
        else:
            self.general = GeneralNonLLMAI(seed_demo=True)
            self.ai = self.general.task_ai


def serve_api(host: str = "127.0.0.1", port: int = 8787, *, mode: str = "general") -> ThreadingHTTPServer:
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
                            "version": __version__,
                            "mode": state.mode,
                            "llm_at_runtime": False,
                            "human_level_agi": False,
                        },
                    )
                if path == "/status":
                    if state.general is not None:
                        return self._json(200, state.general.status())
                    return self._json(200, state.ai.status())
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
                if path in ("/ask", "/handle"):
                    text = data.get("text") or data.get("message") or ""
                    if not text:
                        return self._json(400, {"error": "text_required"})
                    if state.general is not None and data.get("route") != "finderbay":
                        out = state.general.handle(text)
                    else:
                        out = state.ai.handle(text)
                    return self._json(200, out)
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
                    # Also accept natural-language ask through the full AI cycle
                    if data.get("ask"):
                        out = state.ai.handle(str(data["ask"]))
                        return self._json(200, out)
                    out = vision.analyze(image, task=task)
                    return self._json(200, out)
            return self._json(404, {"error": "not_found", "path": path})

    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Non-LLM AI REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--mode", choices=["general", "finderbay"], default="general")
    args = p.parse_args(argv)
    httpd = serve_api(args.host, args.port, mode=args.mode)
    print(f"API listening on http://{args.host}:{args.port}/  (v{__version__}, mode={args.mode})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
