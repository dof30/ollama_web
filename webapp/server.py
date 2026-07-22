#!/usr/bin/env python3
"""
server — a tiny, dependency-free local web server for the research engine.

Stdlib only (http.server), matching agent.py's "no extra dependency" ethos.
It serves a single-page UI and streams research events to the browser so the
whole run is visible live. Runs on localhost only.

    python3 server.py                 # -> http://127.0.0.1:8765
    RESEARCH_WEB_PORT=9000 python3 server.py
"""

import importlib
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent    # noqa: E402  (engine's dependency; held here so we can hot-reload it)
import engine   # noqa: E402
import history  # noqa: E402  (write-only recorder; no HTTP route ever reads it)

# Hot-reload the research code on edit, so a fix on disk is a fix live — matching
# how index.html is already served fresh each request. Without this, agent.py /
# engine.py stay cached in the process and edits silently do nothing until a manual
# restart (which once looked like "the fix didn't work"). Reload only when a file's
# mtime changes; a lock keeps a concurrent request from seeing a half-reloaded module.
_reload_lock = threading.Lock()
_mtimes = {}


def _src_mtime(mod):
    try:
        return os.stat(mod.__file__).st_mtime
    except OSError:
        return None


def hot_reload():
    """Reload agent then engine (engine imports agent, so order matters) if either
    changed. Best-effort: a reload error leaves the last-good modules running."""
    with _reload_lock:
        for mod in (agent, engine):          # agent first: engine re-imports it
            m = _src_mtime(mod)
            if m is not None and _mtimes.get(mod.__file__) != m:
                try:
                    importlib.reload(mod)
                    _mtimes[mod.__file__] = m
                except Exception as e:
                    print(f"hot-reload of {mod.__name__} failed, keeping old: "
                          f"{type(e).__name__}: {e}", flush=True)


# Seed mtimes at startup so the first request is a no-op, not a needless reload.
for _m in (agent, engine):
    _mt = _src_mtime(_m)
    if _mt is not None:
        _mtimes[_m.__file__] = _mt

HOST = os.environ.get("RESEARCH_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("RESEARCH_WEB_PORT", "8765"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.environ.get("RESEARCH_MODEL", "gpt-oss:120b-fast")

# In-memory conversation store: session id -> messages. Single local user, so a
# plain dict is fine; restart clears it (like the CLI's /reset).
SESSIONS = {}

# Session ids whose user pressed "Answer now": the running research loop polls this
# set between steps and, if its sid is here, stops searching and writes the answer
# from what it has. Set from a separate request thread; GIL makes add/discard safe.
WRAP_UP = set()


def list_models():
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            tags = json.load(r)
        return [m["name"] for m in tags.get("models", [])]
    except Exception:
        return []


def unload_model(name):
    """Drop a model from RAM now. keep_alive=0 tells Ollama to unload immediately;
    an empty prompt means it won't reload first. (NOT -1 — that pins a model in RAM
    forever, which once wedged one with a year-2318 expiry.)"""
    body = json.dumps({"model": name, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_HOST + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the console quiet
        pass

    def _local_ok(self):
        """True only when the request is addressed to this machine. Browsers stop a
        malicious page from READING our responses (CORS), but nothing stops one from
        SENDING fire-and-forget requests at localhost — driving the GPU, unloading
        models, polluting history. A Host check defeats that (and DNS rebinding,
        where an attacker's domain resolves to 127.0.0.1 but Host betrays it)."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost", "[::1]")

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
        if not self._local_ok():
            self._send(403, json.dumps({"error": "forbidden"}))
            return
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
        if not self._local_ok():
            self._send(403, json.dumps({"error": "forbidden"}))
            return
        # Require a JSON Content-Type on every POST: our own UI always sends it, but
        # a cross-site page can't (setting it triggers a CORS preflight, which we
        # never approve) — so this one header cheaply blocks drive-by POSTs.
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._send(400, json.dumps({"error": "expected application/json"}))
            return
        hot_reload()  # pick up any edits to agent.py / engine.py before serving
        if self.path == "/api/unload":
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, json.dumps({"error": "bad json"}))
                return
            name = (req.get("model") or "").strip()
            if not name:
                self._send(400, json.dumps({"error": "no model"}))
                return
            try:
                unload_model(name)
                self._send(200, json.dumps({"ok": True, "unloaded": name}))
            except Exception as e:
                self._send(502, json.dumps({"error": f"{type(e).__name__}: {e}"}))
            return
        if self.path == "/api/answer_now":
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, json.dumps({"error": "bad json"}))
                return
            WRAP_UP.add(req.get("sid") or "default")
            self._send(200, json.dumps({"ok": True}))
            return
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
        WRAP_UP.discard(sid)      # a leftover flag from a past run must not cut this one short
        base_len = len(messages)  # remember where this turn starts, for compaction
        messages.append({"role": "user", "content": question})

        # Stream events as Server-Sent-Events style frames.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        final_text = ""
        sources = []
        t0 = time.time()
        try:
            for ev in engine.research_events(model, messages, min_sources=depth,
                                             should_wrap_up=lambda: sid in WRAP_UP):
                if ev.get("type") == "final":
                    final_text = ev.get("text", "")
                elif ev.get("type") == "source":
                    sources.append(ev.get("url", ""))
                frame = "data: " + json.dumps(ev) + "\n\n"
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away / stopped the run
        finally:
            WRAP_UP.discard(sid)  # this run is over; don't leak the flag into the next
            # Shrink this turn back to a clean Q+A so the next turn starts lean and
            # the session can't overflow the context window over a long conversation.
            engine.compact_turn(messages, base_len, question, final_text)
            # Lock the finished turn away in the local history DB (write-only from
            # here; read it later with `python3 webapp/history.py`).
            history.record(sid, model, depth, question, final_text,
                           sources, time.time() - t0)


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
