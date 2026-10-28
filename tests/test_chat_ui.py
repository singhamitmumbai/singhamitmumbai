"""Chat UI frontend — static assets + API smoke (stdlib server)."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from cognitive_ai.ui.server import STATIC_DIR, ChatSession, make_handler
from cognitive_ai.version import __version__


def test_static_assets_exist():
    assert (STATIC_DIR / "index.html").is_file()
    assert (STATIC_DIR / "chat.css").is_file()
    assert (STATIC_DIR / "chat.js").is_file()
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "Complete AI" in html
    assert "/static/chat.css" in html
    assert "/static/chat.js" in html
    css = (STATIC_DIR / "chat.css").read_text(encoding="utf-8")
    assert "--accent" in css
    assert ".thinking" in css
    js = (STATIC_DIR / "chat.js").read_text(encoding="utf-8")
    assert "/api/chat" in js
    assert "/api/search_messages" in js
    assert "/api/vary_message" in js
    assert "renderMarkdown" in js
    assert "thinkingMarkdown" in js or "thinking_markdown" in js
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "msgSearchForm" in html or "Search messages" in html


def test_chat_session_complete_mode():
    s = ChatSession(domain="complete")
    assert s.mode == "general"
    assert s.complete is not None
    out = s.handle_message("Who are you?", mode="auto")
    assert out["human_level_agi"] is False
    assert out.get("reply")
    assert "complete ai" in out["reply"].lower()
    assert isinstance(out.get("thinking"), dict)
    assert out.get("thinking_brief")
    assert out.get("thinking_markdown")


def _get(conn: HTTPConnection, path: str):
    conn.request("GET", path)
    res = conn.getresponse()
    data = res.read()
    return res.status, data


def test_chat_ui_http_endpoints():
    session = ChatSession(domain="complete")
    handler = make_handler(session)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=60)

        status, body = _get(conn, "/")
        assert status == 200
        assert b"Complete AI" in body

        status, _ = _get(conn, "/static/chat.css")
        assert status == 200
        status, _ = _get(conn, "/static/chat.js")
        assert status == 200

        status, raw = _get(conn, "/api/hello")
        assert status == 200
        hello = json.loads(raw.decode("utf-8"))
        assert "greeting" in hello

        payload = json.dumps({"message": "Can penguins fly?", "mode": "auto"}).encode()
        conn.request(
            "POST",
            "/api/chat",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        chat = json.loads(res.read().decode("utf-8"))
        assert res.status == 200
        assert chat.get("human_level_agi") is False
        assert chat.get("reply")
        assert "cannot" in chat["reply"].lower() or "no" in chat["reply"].lower()
        assert isinstance(chat.get("thinking"), dict)
        assert chat.get("thinking_markdown") or chat.get("thinking_brief")

        status, raw = _get(conn, "/api/completeness")
        assert status == 200
        comp = json.loads(raw.decode("utf-8"))
        assert "ok" in comp or "complete_code" in comp

        # Message search + variation APIs
        payload = json.dumps({"query": "penguin"}).encode()
        conn.request(
            "POST",
            "/api/search_messages",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        search = json.loads(res.read().decode("utf-8"))
        assert res.status == 200
        assert search.get("ok") is True
        assert search.get("human_level_agi") is False

        payload = json.dumps({"text": "Show unpaid invoices for Project A", "n": 3}).encode()
        conn.request(
            "POST",
            "/api/vary_message",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        vary = json.loads(res.read().decode("utf-8"))
        assert res.status == 200
        assert vary.get("ok") is True
        assert len(vary.get("variations") or []) >= 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ui_version_available():
    assert __version__
