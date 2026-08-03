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

# sid -> id of the newest turn started for it. Turns can overlap (Stop, then re-ask
# before the stopped thread has noticed the dead connection), and only the newest one
# is allowed to write back to the session. The lock covers both this and SESSIONS.
TURNS = {}
_session_lock = threading.Lock()

# Intelligence is how hard the model THINKS, not how much it is forced to read. An
# earlier version armed the depth gate at "high", and asking for three prime numbers
# then took 124s because the gate made it web_fetch five sites for a maths question.
# The gate stays available to the CLI (--deep) and to the legacy `depth` field; the
# selector leaves it off so effort stays proportional to the question.
EFFORT_SOURCES = {"high": 0, "medium": 0, "low": 0}

# A browser tab outlives this process: it keeps the conversation in localStorage, so
# after a restart it asks with a sid we have never seen. Rebuild the session from the
# tab's own copy — otherwise the page shows a conversation the model has no memory of,
# and the next follow-up gets answered out of thin air. Bounded on every axis so a
# crafted payload can't quietly fill the context window.
RESTORE_TURNS = 8
RESTORE_CHARS = 4000


def seed_session(restore):
    """A fresh conversation, optionally pre-loaded with a tab's remembered turns."""
    messages = engine.new_conversation()
    if not isinstance(restore, list):
        return messages
    for turn in restore[-RESTORE_TURNS:]:
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("q") or "")[:RESTORE_CHARS]
        a = str(turn.get("a") or "")[:RESTORE_CHARS]
        if q and a:  # only complete pairs; a dangling question teaches it nothing
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    return messages


def list_models():
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            tags = json.load(r)
        return [m["name"] for m in tags.get("models", [])]
    except Exception:
        return []


def set_keep_alive(name, keep_alive, timeout=120):
    """Set how long `name` stays resident, without disturbing a running copy.

    Mirrors agent.ollama_chat_stream's request exactly — same endpoint, same options —
    with an empty message list so nothing is generated. The options are the whole point:
    Ollama matches a request against the loaded runner's config, so re-arming with a
    different num_ctx evicts the model and loads a fresh one instead of just touching
    its timer. That turned an F5 into a 28s reload."""
    payload = {"model": name, "messages": [], "stream": False,
               "keep_alive": keep_alive, "options": agent.chat_options()}
    req = urllib.request.Request(OLLAMA_HOST + "/api/chat",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def unload_model(name):
    """Drop a model from RAM now. keep_alive=0 unloads immediately; no messages means
    it won't generate. (NOT -1 — that pins RAM forever, and once wedged a model with a
    year-2318 expiry.)"""
    return set_keep_alive(name, 0)


def switch_model(old, new):
    """Hand RAM from one model to the next: drop `old`, then pull `new` in.

    Order matters on a box where two 100B-class models don't fit at once — freeing
    first means the load isn't racing the old model for memory. `old` is only touched
    if it is actually resident, since the empty-prompt request that unloads a model
    would otherwise *load* one that wasn't there just to expire it. Loading is the
    same empty-message call, so the new model arrives warm without generating a token.
    The long timeout is the point of the whole feature: a 120B cold load is ~30s, and
    a bigger one can take minutes."""
    unloaded = None
    if old and old != new and old in loaded_models():
        unload_model(old)
        unloaded = old
    set_keep_alive(new, agent.keep_alive_for(new) or IDLE_KEEP_ALIVE, timeout=600)
    return {"ok": True, "unloaded": unloaded, "loaded": new}


# ---------- tab presence -> how long the workhorse stays warm ----------
# The 45m sticky keep_alive buys instant answers, but it only earns its ~65 GB while
# someone is actually there to ask something. So each open tab sends a heartbeat: with
# at least one tab open the sticky 45m stands; when the last one closes we shorten the
# timer to the server's normal 5m default. Tabs that vanish without a goodbye beacon
# (crash, laptop sleep) age out via TAB_TTL — generous, because browsers throttle
# timers in background tabs, and a hidden tab is still an open tab.
TAB_TTL = 150.0     # seconds of silence before a tab counts as gone
TAB_SWEEP = 30.0    # how often the watcher re-evaluates
IDLE_KEEP_ALIVE = os.environ.get("RESEARCH_KEEP_ALIVE_IDLE", "5m")

PRESENCE = {}                # tab id -> last heartbeat (monotonic clock)
_presence_lock = threading.Lock()
_tabs_warm = None            # last applied state: True=some tab, False=none, None=unknown


def loaded_models():
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/ps", timeout=5) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def apply_presence():
    """Point sticky models' keep_alive at the current tab situation, on change only.

    Only ever touches a model that is ALREADY loaded: the same empty-prompt request
    that sets a timer would otherwise *pull the model into RAM* to do it — the exact
    opposite of the point when nobody has a tab open."""
    global _tabs_warm
    now = time.monotonic()
    with _presence_lock:
        for tab, seen in list(PRESENCE.items()):
            if now - seen > TAB_TTL:
                del PRESENCE[tab]
        warm = bool(PRESENCE)
        if warm == _tabs_warm:
            return
        _tabs_warm = warm
    ka = agent.KEEP_ALIVE_STICKY if warm else IDLE_KEEP_ALIVE
    for name in loaded_models():
        if name in agent.STICKY_MODELS:
            try:
                set_keep_alive(name, ka)
            except Exception:
                pass  # best effort — the server's global default still bounds it


def _presence_watcher():
    while True:
        time.sleep(TAB_SWEEP)
        try:
            apply_presence()
        except Exception:
            pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the console quiet
        pass

    def handle_one_request(self):
        # Pressing Stop or closing the tab mid-answer leaves http.server flushing into
        # a socket that is already gone. That is a normal end to a run, but it dumps a
        # twelve-line traceback into the log each time, which buries real errors.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

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
        if self.path in ("/api/ping", "/api/bye"):
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                req = {}
            tab = str(req.get("tab") or "")[:64]
            if tab:
                with _presence_lock:
                    if self.path == "/api/ping":
                        PRESENCE[tab] = time.monotonic()
                    else:
                        PRESENCE.pop(tab, None)
            apply_presence()  # react at once when the first tab opens / the last closes
            self._send(200, json.dumps({"ok": True}))
            return
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
        if self.path == "/api/switch":
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, json.dumps({"error": "bad json"}))
                return
            new = (req.get("to") or "").strip()
            if not new:
                self._send(400, json.dumps({"error": "no model"}))
                return
            try:
                self._send(200, json.dumps(switch_model((req.get("from") or "").strip(), new)))
            except Exception as e:
                self._send(502, json.dumps({"error": f"{type(e).__name__}: {e}"}))
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
        sid = req.get("sid") or "default"
        # The UI sends an intelligence level; gpt-oss takes it as reasoning effort.
        # "high" also opens the depth gate, so thinking harder means reading more
        # rather than only deliberating longer. `depth` is the older field name and
        # is still honoured so a stale cached page keeps working.
        effort = (req.get("effort") or "").strip().lower()
        if effort not in ("low", "medium", "high"):
            effort = None
        depth = EFFORT_SOURCES.get(effort, int(req.get("depth") or 0))
        if not question:
            self._send(400, json.dumps({"error": "empty question"}))
            return

        # Claim the session, then work on a private copy. A stopped turn's thread keeps
        # running until its next write fails, so it can still be alive when the next
        # question arrives; the copy keeps the two loops from corrupting each other's
        # context, and turn_id stops the older one writing back over the newer answer.
        with _session_lock:
            messages = SESSIONS.get(sid)
            if messages is None:   # unknown sid: a tab from before the last restart
                messages = SESSIONS[sid] = seed_session(req.get("restore"))
            turn_id = TURNS[sid] = TURNS.get(sid, 0) + 1
        work = engine.start_turn(messages, question)

        # Stream events as Server-Sent-Events style frames.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        final_text = ""
        sources = []
        aborted = False
        t0 = time.time()
        try:
            for ev in engine.research_events(model, work, min_sources=depth, effort=effort):
                if ev.get("type") == "final":
                    final_text = ev.get("text", "")
                elif ev.get("type") == "source":
                    sources.append(ev.get("url", ""))
                frame = "data: " + json.dumps(ev) + "\n\n"
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            aborted = True  # browser navigated away / stopped the run
        finally:
            with _session_lock:
                # Only the newest turn owns this session. A superseded one must not
                # graft its stale answer onto the session.
                if TURNS.get(sid) == turn_id:
                    if not aborted:
                        # Keep only the clean Q+A: the turn's research bulk is
                        # discarded with its working copy, so a long conversation
                        # can't creep up on the context window.
                        engine.commit_turn(messages, question, final_text)
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
        threading.Thread(target=_presence_watcher, daemon=True).start()
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
