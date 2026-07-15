#!/usr/bin/env python3
"""
server — a tiny, dependency-free local web server for the research engine.

Stdlib only (http.server), matching agent.py's "no extra dependency" ethos.
It serves a single-page UI and streams research events to the browser so the
whole run is visible live. Runs on localhost only.

    python3 server.py                 # -> http://127.0.0.1:8765
    RESEARCH_WEB_PORT=9000 python3 server.py
"""

import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402

HOST = os.environ.get("RESEARCH_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("RESEARCH_WEB_PORT", "8765"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.environ.get("RESEARCH_MODEL", "gpt-oss:120b-fast")

# In-memory conversation store: session id -> messages. Single local user, so a
# plain dict is fine; restart clears it (like the CLI's /reset).
SESSIONS = {}


def list_models():
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            tags = json.load(r)
        return [m["name"] for m in tags.get("models", [])]
    except Exception:
        return []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "static", "index.html"), "rb") as f:
                # no-store so a browser refresh always picks up UI edits (no stale cache)
                self._send(200, f.read(), "text/html; charset=utf-8",
                           {"Cache-Control": "no-store"})
        elif path == "/api/models":
            # "app" marks this as our server so the launcher can find a running
            # instance by probing ports instead of starting a duplicate.
            self._send(200, json.dumps({"app": "local-research",
                                        "models": list_models(),
                                        "default": DEFAULT_MODEL}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/ask":
            self._send(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "bad json"}))
            return

        question = (req.get("q") or "").strip()
        model = req.get("model") or DEFAULT_MODEL
        depth = int(req.get("depth") or 0)
        sid = req.get("sid") or "default"
        if not question:
            self._send(400, json.dumps({"error": "empty question"}))
            return

        messages = SESSIONS.setdefault(sid, engine.new_conversation())
        base_len = len(messages)  # remember where this turn starts, for compaction
        messages.append({"role": "user", "content": question})

        # Stream events as Server-Sent-Events style frames.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        final_text = ""
        try:
            for ev in engine.research_events(model, messages, min_sources=depth):
                if ev.get("type") == "final":
                    final_text = ev.get("text", "")
                frame = "data: " + json.dumps(ev) + "\n\n"
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away / stopped the run
        finally:
            # Shrink this turn back to a clean Q+A so the next turn starts lean and
            # the session can't overflow the context window over a long conversation.
            engine.compact_turn(messages, base_len, question, final_text)


def main():
    # Try the preferred port, then a few above it, so a stale instance already on
    # 8765 doesn't crash us with "Address already in use" — we just move over.
    last_err = None
    for port in range(PORT, PORT + 10):
        try:
            srv = ThreadingHTTPServer((HOST, port), Handler)
        except OSError as e:
            last_err = e
            continue
        print(f"research web UI  ·  http://{HOST}:{port}  ·  default model {DEFAULT_MODEL}", flush=True)
        if port != PORT:
            print(f"(port {PORT} was busy — using {port} instead)", flush=True)
        print("Ctrl+C to stop.", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")
        return
    print(f"couldn't bind any port in {PORT}..{PORT + 9}: {last_err}")
    print("Free one of those ports, or set a different one: RESEARCH_WEB_PORT=9000 python3 webapp/server.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
