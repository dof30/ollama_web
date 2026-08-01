#!/usr/bin/env python3
"""
server — thin HTTP shell over pipeline.research().

Deliberately thin. All the intelligence is in pipeline.py; this file moves bytes,
holds session state, and enforces that only this machine can talk to it.

It serves the *existing* webapp/static/index.html unchanged. The event protocol in
events.py is a superset of what that page already understands, and its event
switch has no default branch, so the two new event types are ignored rather than
breaking it. Rewriting the UI can wait until the mesh underneath is worth looking at.

Runs on port 8770 so it can sit alongside the old webapp on 8765.
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh import config, ollama, pipeline  # noqa: E402

try:
    # Write-only turn recorder. Same DB as before (~/.local/share/research-web/
    # history.db), so nothing already recorded is lost by moving the module here.
    from mesh import history
except Exception:  # pragma: no cover - history is strictly optional
    history = None

PORT = int(os.environ.get("MESH_PORT", "8770"))
HOST = os.environ.get("MESH_HOST", "127.0.0.1")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# sid -> list of (question, answer). Bounded when rendered; see state.render_history.
SESSIONS = {}
# Backstop for sessions that never say goodbye — a crashed tab, a killed browser,
# a sendBeacon that didn't make it. Without this the map grows for the life of the
# process, one entry per page load.
MAX_SESSIONS = 40
# sids whose current turn the user asked to cut short ("Answer now").
WRAP_UP = set()
LOCK = threading.Lock()

# Tabs that have pinged recently. With none open we shorten keep-alive rather than
# holding 88 GB of models warm for nobody.
PRESENT = {}
PRESENCE_TTL = 90


def _anyone_present():
    now = time.monotonic()
    return any(now - t < PRESENCE_TTL for t in PRESENT.values())


def _presence_watcher():
    """Drop both models to a short keep-alive once every tab has gone away."""
    warm = True
    while True:
        time.sleep(30)
        here = _anyone_present()
        if here == warm:
            continue
        warm = here
        ka = config.KEEP_ALIVE_ACTIVE if here else config.KEEP_ALIVE_IDLE
        ollama.touch(config.BIG_MODEL, config.BIG_OPTIONS, ka)
        ollama.touch(config.SMALL_MODEL, config.SMALL_OPTIONS, ka)


def _forget(sid):
    """Drop a session's history and presence. Called when a page goes away.

    F5, "New chat", and closing the tab all end a session — the page mints a fresh
    sid each load, so nothing will ever ask for the old one again. Reaping here is
    what makes refreshing safe rather than slowly leaky.
    """
    with LOCK:
        SESSIONS.pop(sid, None)
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))  # dicts keep insertion order: oldest first
    PRESENT.pop(sid, None)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mesh"

    def log_message(self, fmt, *args):
        pass  # the default logger writes a line per request; too noisy for a local app

    # --- guards -------------------------------------------------------------

    def _local_only(self):
        """Reject anything not addressed to this machine by name.

        Blocks DNS-rebinding: a hostile page can make your browser POST here, but
        it cannot forge the Host header to say 127.0.0.1. Carried over from the
        old server, which is the only reason this is safe to leave running.
        """
        host = (self.headers.get("Host") or "").split(":")[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1"):
            self.send_error(403, "local only")
            return False
        return True

    def _json_body(self):
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            # Forces a CORS preflight for cross-origin callers, which we never answer.
            self.send_error(415, "json only")
            return None
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 1_000_000:
                raise ValueError
            return json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.send_error(400, "bad body")
            return None

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    # --- routes -------------------------------------------------------------

    def do_GET(self):
        if not self._local_only():
            return
        path = self.path.split("?")[0]
        if path == "/api/models":
            # The "app" field is how the launcher script tells our port apart from
            # anything else that happens to answer on 8765-8774.
            return self._send(200, {"app": "local-research",
                                    "models": ollama.list_models(),
                                    "big": config.BIG_MODEL,
                                    "small": config.SMALL_MODEL})
        if path in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        if path == "/icon.svg":
            return self._file("icon.svg", "image/svg+xml")
        self.send_error(404)

    def _file(self, name, ctype):
        # Fixed allow-list of filenames — nothing here is derived from the URL, so
        # there is no path to traverse.
        try:
            with open(os.path.join(STATIC, name), "rb") as f:
                return self._send(200, f.read(), ctype)
        except OSError:
            self.send_error(404)

    def do_POST(self):
        if not self._local_only():
            return
        path = self.path.split("?")[0]
        body = self._json_body()
        if body is None:
            return
        sid = str(body.get("sid") or "")

        if path == "/api/ping":
            PRESENT[sid] = time.monotonic()
            return self._send(200, {"ok": True})
        if path in ("/api/bye", "/api/new"):
            _forget(sid)
            return self._send(200, {"ok": True})
        if path == "/api/answer_now":
            WRAP_UP.add(sid)
            return self._send(200, {"ok": True})
        if path == "/api/unload":
            for m, o in ((config.BIG_MODEL, config.BIG_OPTIONS),
                         (config.SMALL_MODEL, config.SMALL_OPTIONS)):
                ollama.touch(m, o, "0s")
            return self._send(200, {"ok": True})
        if path == "/api/ask":
            return self._ask(body, sid)
        self.send_error(404)

    def _ask(self, body, sid):
        question = (body.get("q") or "").strip()
        if not question:
            return self._send(400, {"error": "empty question"})
        effort = str(body.get("effort") or "auto").lower()
        if effort not in pipeline.EFFORT and effort != "auto":
            effort = "auto"

        with LOCK:
            turns = list(SESSIONS.get(sid) or [])
        WRAP_UP.discard(sid)
        PRESENT[sid] = time.monotonic()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        started = time.monotonic()
        answer, sources = "", []
        try:
            for ev in pipeline.research(
                question, history=turns, effort=effort,
                should_stop=lambda: sid in WRAP_UP,
            ):
                if ev["type"] == "final":
                    answer = ev["text"]
                elif ev["type"] == "source":
                    sources.append(ev["url"])
                self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # user navigated away mid-answer; nothing to clean up
        except Exception as e:
            try:
                self.wfile.write(
                    b"data: " + json.dumps({"type": "error", "text": str(e)}).encode()
                    + b"\n\ndata: " + json.dumps({"type": "done"}).encode() + b"\n\n"
                )
                self.wfile.flush()
            except OSError:
                pass
        finally:
            WRAP_UP.discard(sid)
            if answer:
                with LOCK:
                    SESSIONS.setdefault(sid, []).append((question, answer))
            if history is not None:
                try:
                    history.record(
                        sid=sid, model=config.PRIMARY_MODEL,
                        depth=pipeline.EFFORT.get(effort, 0),
                        question=question, answer=answer, sources=sources,
                        seconds=time.monotonic() - started,
                    )
                except Exception:
                    pass  # history can never break a run


def main():
    threading.Thread(target=_presence_watcher, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    print(f"mesh  http://{HOST}:{PORT}")
    print(f"  judgment: {config.BIG_MODEL} (ctx {config.BIG_NUM_CTX})")
    print(f"  volume:   {config.SMALL_MODEL} (ctx {config.SMALL_NUM_CTX})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
