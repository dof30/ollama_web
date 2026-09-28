#!/usr/bin/env python3
"""
server — a tiny, dependency-free local web server for the research engine.

Stdlib only (http.server), matching agent.py's "no extra dependency" ethos.
It serves a single-page UI and streams research events to the browser so the
whole run is visible live. Runs on localhost only.

    python3 server.py                 # -> http://127.0.0.1:8765
    RESEARCH_WEB_PORT=9000 python3 server.py
"""

import base64
import importlib
import json
import os
import re
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
# Deliberately NOT hot-reloaded: reloading either would drop a resident model (whisper,
# Kokoro) and pay the load again on the next click.
import voice    # noqa: E402  (dictation; inert if faster-whisper is absent)
import speech   # noqa: E402  (speaking;  inert if kokoro-onnx is absent)

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
# gemma4:26b (~17 GB), not gpt-oss:120b-fast (~61 GB): this box also runs llama.cpp
# servers that Ollama cannot see, and with one of those resident there is nowhere near
# 61 GB free. Defaulting to a model that fits is the difference between a slow answer
# and a frozen machine. Override with RESEARCH_MODEL when the box is otherwise empty.
DEFAULT_MODEL = os.environ.get("RESEARCH_MODEL", "gemma4:26b")

# How much request body we will read. Everything but dictation carries a few short
# fields; a recording is up to ~90s of 16 kHz mono WAV, base64-inflated by a third.
MAX_BODY = 64 * 1024
MAX_AUDIO_BODY = 6 * 1024 * 1024

# In-memory conversation store: session id -> messages. Single local user, so a
# plain dict is fine; restart clears it (like the CLI's /reset). Insertion order is
# the eviction order — see the cap in /api/ask.
SESSIONS = {}

# Each TAB is now a session, not each browser, so sids arrive faster than they used to
# (open four tabs, get four) and nothing here ever deleted one. Cap it and drop the
# least recently used. Eviction is safe rather than lossy: the tab keeps its own copy
# of the thread and sends it back as `restore`, which is the same path a tab takes
# after a server restart. The cap is far above any plausible number of open tabs, so
# the session a live turn is writing into is never the one evicted.
MAX_SESSIONS = 40

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

# Which model this server considers ACTIVE — the one whose weights we have committed
# RAM to. It has to live here rather than in a tab, because a tab only knows what it
# picked: switch models in one tab and every other tab kept showing the old name and
# would happily send the next question to it, silently evicting the model you just
# loaded. A plain string needs no lock (assignment is atomic under the GIL) and every
# tab learns it from /api/models on load and /api/ping while open.
ACTIVE_MODEL = DEFAULT_MODEL

# The band the temperature dial is allowed to move in — see the clamp in /api/ask.
TEMP_MIN = float(os.environ.get("RESEARCH_TEMP_MIN", "0.4"))
TEMP_MAX = float(os.environ.get("RESEARCH_TEMP_MAX", "1.0"))

# A browser tab outlives this process: it keeps its own conversation in sessionStorage,
# so after a restart it asks with a sid we have never seen. Rebuild the session from
# the tab's own copy — otherwise the page shows a conversation the model has no memory of,
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


# An embedding model cannot hold a conversation — /api/chat on one errors out — so it has
# no business in a picker whose only job is choosing who answers. Ollama's tag list carries
# no capability flag, but every embedding model on the hub says so in its name, and a chat
# model that happens to say "embed" is not a thing. Filtered here rather than in the UI: a
# model that cannot answer must not be selectable by ANY client, and the RAM-admission
# paths read this same list.
EMBEDDING_NAME = re.compile(r"embed", re.I)


def list_models():
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            tags = json.load(r)
        return [m["name"] for m in tags.get("models", [])
                if not EMBEDDING_NAME.search(m["name"])]
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
    APP_MODELS.discard(name)   # no longer ours to evict later
    return set_keep_alive(name, 0)


# Models this app pulled into RAM. Ollama will not tell us who loaded what, and the
# voice app on this machine legitimately keeps a model of its own resident — so we
# remember ours and leave anything we did not load alone. The exception is a model too
# large to share the box with anything (below), where politeness would just mean a
# failed load or a swap storm.
APP_MODELS = set()
# At or above this, a model gets the machine to itself: qwen3.5:122b is ~81 GB against
# 121 GB of RAM, so any other 100B-class model resident at the same time does not fit.
BIG_MODEL_BYTES = int(os.environ.get("RESEARCH_BIG_MODEL_BYTES", 70_000_000_000))
# Never plan to use the last of the machine's memory: the kernel, the desktop and the
# browser all still need room, and a load that only just fits is a load that swaps.
RAM_RESERVE = int(os.environ.get("RESEARCH_RAM_RESERVE", 6_000_000_000))


def model_sizes():
    """name -> on-disk size in bytes, for every installed model."""
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            return {m["name"]: m.get("size", 0) for m in json.load(r).get("models", [])}
    except Exception:
        return {}


# ---------- what is actually holding this machine's RAM ----------
# Ollama's /api/ps knows only about Ollama. A llama.cpp server is an ordinary process,
# invisible to it — which is exactly how this app once loaded 61 GB onto a box that was
# already committed, and froze it. So ask the kernel, not one runtime.

def mem_info():
    """(total, available) bytes, straight from /proc/meminfo.

    MemAvailable, not MemFree: it is the kernel's own estimate of what a new process
    can actually get, counting reclaimable cache. MemFree would read as near-zero on a
    healthy machine and make everything look impossible."""
    vals = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    vals[key] = int(rest.split()[0]) * 1024
    except OSError:
        return 0, 0
    return vals.get("MemTotal", 0), vals.get("MemAvailable", 0)


def llama_servers():
    """Every llama.cpp server running on this box, with what it is serving and costs.

    Found by walking /proc rather than asking any daemon, because there is no daemon to
    ask. argv[0] must END in llama-server, so a shell that merely mentions the name in
    a command line is not mistaken for the real process."""
    found = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return found
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
            if not argv or not argv[0].endswith("llama-server"):
                continue
            name = ""
            if "-m" in argv:
                path = argv[argv.index("-m") + 1]
                # "Qwen3-…-00001-of-00004.gguf" is a shard of one model, not a name.
                name = re.sub(r"-\d{5}-of-\d{5}$", "",
                              os.path.basename(path).removesuffix(".gguf"))
            rss = 0
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                        break
            found.append({"name": name or "llama.cpp server", "bytes": rss,
                          "backend": "llama.cpp"})
        except (OSError, IndexError, ValueError):
            continue          # process exited mid-walk, or is not ours to read
    return found


def machine_state():
    """Everything the header needs to say what is loaded and what is left."""
    total, available = mem_info()
    sizes = model_sizes()
    resident = [{"name": m, "bytes": sizes.get(m, 0), "backend": "ollama"}
                for m in loaded_models()] + llama_servers()
    return {"total": total, "available": available, "active": ACTIVE_MODEL,
            "resident": resident, "sizes": sizes, "reserve": RAM_RESERVE}


def admit(target):
    """May `target` be pulled into RAM without taking the machine down? (ok, detail)

    Admission control: the check has to happen BEFORE the request that causes the load,
    because a warning that arrives afterwards is not a guard. Fails closed — an unknown
    size is a refusal, since permitting it is how you find out the hard way."""
    _, available = mem_info()
    if target in loaded_models():
        return True, {"need": 0, "free": available, "frees": [], "blockers": []}
    sizes = model_sizes()
    plan = eviction_plan(target)
    room = available + sum(sizes.get(m, 0) for m in plan)
    blockers = [s for s in llama_servers() if s["bytes"] > 0]
    detail = {"need": sizes.get(target, 0), "free": room, "reserve": RAM_RESERVE,
              "frees": plan, "blockers": blockers}
    if not sizes.get(target):
        detail["why"] = f"{target} is not installed here, so its size is unknown"
        return False, detail
    return sizes[target] + RAM_RESERVE <= room, detail


def eviction_plan(target):
    """Which resident Ollama models would have to go for `target` to load.

    Split out from the eviction itself so admission control can ask what a load would
    cost without performing it — the plan is what makes "will it fit" honest, since
    the memory these models hold is about to come back."""
    loaded = [m for m in loaded_models() if m != target]
    if not loaded:
        return []
    takes_whole_box = model_sizes().get(target, 0) >= BIG_MODEL_BYTES
    return loaded if takes_whole_box else [m for m in loaded if m in APP_MODELS]


def free_ram_for(target):
    """Unload whatever must go before `target` can be resident, and say what went.

    Normally that is only the other models WE loaded — one model per app, but the voice
    app's model is left alone. For a model at BIG_MODEL_BYTES or above it is everything
    else on the machine, because nothing else fits alongside it."""
    gone = []
    for m in eviction_plan(target):
        try:
            unload_model(m)
            gone.append(m)
        except Exception:
            pass          # best effort; Ollama's own timer still bounds it
        APP_MODELS.discard(m)
    return gone


def switch_model(old, new):
    """Make `new` the resident model: free what has to go, then pull it in.

    Order matters on a box where two 100B-class models don't fit at once — freeing
    first means the load isn't racing the old model for memory. Only models that are
    actually resident are touched, since the empty-prompt request that unloads a model
    would otherwise *load* one that wasn't there just to expire it. Loading is the
    same empty-message call, so the new model arrives warm without generating a token.
    The long timeout is the point of the whole feature: a 120B cold load is ~30s, and
    a bigger one can take minutes."""
    global ACTIVE_MODEL
    unloaded = free_ram_for(new)
    set_keep_alive(new, agent.keep_alive_for(new) or IDLE_KEEP_ALIVE, timeout=600)
    APP_MODELS.add(new)
    ACTIVE_MODEL = new   # the new single source of truth every tab syncs to
    return {"ok": True, "unloaded": unloaded, "loaded": new, "active": new}


# ---------- tab presence -> how long the workhorse stays warm ----------
# The 45m sticky keep_alive buys instant answers, but it only earns its ~65 GB while
# someone is actually there to ask something. So each open tab sends a heartbeat: with
# at least one tab open the sticky window is renewed on every sweep, so it never runs
# down under someone who is reading rather than asking; when the last one closes we
# shorten the timer to the server's normal 5m default. Tabs that vanish without a
# goodbye beacon
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


def apply_presence(rearm=False):
    """Point sticky models' keep_alive at the current tab situation.

    With a tab open the sticky window has to keep sliding forward, so `rearm` re-applies
    it on every sweep rather than only when presence changes. Arming it once was enough
    to lose the model anyway: the 45m timer started counting at the moment the first tab
    opened, so reading for longer than that — without asking anything, which is what
    refreshes it — silently dropped the workhorse under a tab that was still open.
    Re-arming is deliberately a sliding finite timer and never -1: if this process dies
    or the box sleeps, nothing renews it and Ollama reclaims the RAM on its own.

    Only the sweep thread passes `rearm` — a heartbeat must not make a blocking Ollama
    call every 30s just to say the tab is still there.

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
        changed = warm != _tabs_warm
        _tabs_warm = warm
    # Cold is a one-shot: shorten the timer as the last tab goes, then let Ollama's
    # clock run. Only the warm side needs renewing.
    if not changed and not (warm and rearm):
        return
    ka = agent.KEEP_ALIVE_STICKY if warm else IDLE_KEEP_ALIVE
    for name in loaded_models():
        if agent.is_sticky(name):
            try:
                set_keep_alive(name, ka)
            except Exception:
                pass  # best effort — the server's global default still bounds it


def _presence_watcher():
    while True:
        time.sleep(TAB_SWEEP)
        try:
            apply_presence(rearm=True)
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

    def _reject(self, code, msg):
        """Error out AND hang up. Any path that answers without reading the body must:
        under HTTP/1.1 keep-alive the unread bytes are parsed as the next request line,
        so bailing out of a multi-megabyte audio POST would desync the socket."""
        self._send(code, json.dumps({"error": msg}), extra={"Connection": "close"})

    def _read_json(self, limit=MAX_BODY):
        """Parse this request's JSON body, or return None having already answered.
        Content-Length is caller-controlled, so it is checked BEFORE we allocate."""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = -1
        if length < 0:
            self._reject(400, "bad Content-Length")
            return None
        if length > limit:
            self._reject(413, "body too large")
            return None
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._reject(400, "bad json")
            return None

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
            # "voice"/"speech" tell the page whether to draw the mic and the speak
            # switch at all — this is already the handshake, so neither costs a round
            # trip, and a machine without the models simply doesn't show the controls.
            self._send(200, json.dumps({"app": "local-research",
                                        "models": list_models(),
                                        "default": DEFAULT_MODEL,
                                        "active": ACTIVE_MODEL,
                                        "voice": voice.available(),
                                        "speech": speech.available()}))
        elif path == "/api/state":
            # What is holding RAM right now, across BOTH runtimes, for the header.
            self._send(200, json.dumps(machine_state()), extra={"Cache-Control": "no-store"})
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _transcribe(self):
        """Recorded audio in (base64 WAV), text out. Dictation only: the transcript
        goes to the composer for you to read and send — nothing is asked from here.

        The WAV rides inside the ordinary JSON body rather than as a raw audio/wav
        upload, to keep the Content-Type check above free of exceptions: a carve-out
        for one route is how a security rule quietly starts to rot. Base64 costs a
        third more bytes over loopback, which is nothing."""
        if not voice.available():
            self._reject(503, "dictation is not available on this server")
            return
        req = self._read_json(limit=MAX_AUDIO_BODY)
        if req is None:
            return
        try:
            wav = base64.b64decode(req.get("wav") or "", validate=True)
        except Exception:
            self._send(400, json.dumps({"error": "wav must be base64"}))
            return
        # Check the bytes really are a RIFF/WAVE before handing them to a decoder.
        if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            self._send(400, json.dumps({"error": "not a WAV file"}))
            return
        try:
            self._send(200, json.dumps({"text": voice.transcribe(wav)}))
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def _speak(self):
        """One sentence in, WAV out — the reply to _transcribe.

        The response is raw audio rather than base64-in-JSON: the Content-Type rule
        that forced that choice for uploads is about REQUEST bodies, so it does not
        apply here, and the page can hand the blob straight to an <audio> element.

        What is safe to say has already been decided by the page, which is the only
        place that can tell a finished answer from the model's thinking as it streams.
        This end just refuses to be handed an essay."""
        if not speech.available():
            self._reject(503, "speech is not available on this server")
            return
        req = self._read_json()
        if req is None:
            return
        text = (req.get("text") or "").strip()
        if not text:
            self._send(400, json.dumps({"error": "nothing to say"}))
            return
        try:
            wav = speech.synth(text)
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
            return
        if not wav:
            self._send(400, json.dumps({"error": "nothing to say"}))
            return
        self._send(200, wav, "audio/wav", {"Cache-Control": "no-store"})

    def do_POST(self):
        if not self._local_ok():
            self._send(403, json.dumps({"error": "forbidden"}))
            return
        # Require a JSON Content-Type on every POST: our own UI always sends it, but
        # a cross-site page can't (setting it triggers a CORS preflight, which we
        # never approve) — so this one header cheaply blocks drive-by POSTs.
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._reject(400, "expected application/json")   # body goes unread
            return
        hot_reload()  # pick up any edits to agent.py / engine.py before serving
        if self.path == "/api/transcribe":
            self._transcribe()
            return
        if self.path == "/api/speak":
            self._speak()
            return
        if self.path in ("/api/ping", "/api/bye"):
            req = self._read_json()
            if req is None:
                return
            tab = str(req.get("tab") or "")[:64]
            if tab:
                with _presence_lock:
                    if self.path == "/api/ping":
                        PRESENCE[tab] = time.monotonic()
                    else:
                        PRESENCE.pop(tab, None)
            apply_presence()  # react at once when the first tab opens / the last closes
            # The heartbeat carries the active model back, so a tab that was open when
            # another one switched corrects itself instead of asking the wrong model.
            self._send(200, json.dumps({"ok": True, "active": ACTIVE_MODEL}))
            return
        if self.path == "/api/unload":
            req = self._read_json()
            if req is None:
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
            req = self._read_json()
            if req is None:
                return
            new = (req.get("to") or "").strip()
            if not new:
                self._send(400, json.dumps({"error": "no model"}))
                return
            # Ask before loading, not after. This is the door: a model that cannot fit
            # is refused here, while refusing is still cheap. 409 = "understood, but
            # not in this state" — the UI shows the reason inline.
            ok, why = admit(new)
            if not ok:
                gib = lambda b: round(b / 2**30, 1)
                held = ", ".join(f"{b['name']} ({gib(b['bytes'])} GB)" for b in why["blockers"])
                msg = why.get("why") or (
                    f"{new} needs {gib(why['need'])} GB and only {gib(why['free'])} GB is free")
                if held:
                    msg += f" — {held} is holding memory outside Ollama"
                self._send(409, json.dumps({"error": msg, "admission": why}))
                return
            try:
                self._send(200, json.dumps(switch_model((req.get("from") or "").strip(), new)))
            except Exception as e:
                self._send(502, json.dumps({"error": f"{type(e).__name__}: {e}"}))
            return
        if self.path != "/api/ask":
            self._send(404, json.dumps({"error": "not found"}))
            return
        req = self._read_json()
        if req is None:
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
        # Sampling temperature, clamped to the useful band for this model: below ~0.4 a
        # reasoning model starts repeating its own thinking verbatim, and above 1.0 it
        # wanders — 1.0 is OpenAI's recommendation and the default. Anything unparseable
        # falls back to that default rather than failing the request; a bad slider value
        # should never cost you an answer.
        try:
            temperature = min(TEMP_MAX, max(TEMP_MIN, float(req["temp"])))
        except (KeyError, TypeError, ValueError):
            temperature = None
        if not question:
            self._send(400, json.dumps({"error": "empty question"}))
            return
        # A question from a tab that is out of date about which model is loaded would
        # silently evict the active one and load its own — the exact accident switching
        # models in another tab used to cause. Refuse instead, and say what is actually
        # active so the tab can correct itself and hand the question back for a re-send.
        # Changing model is a deliberate, confirmed act now; it can't happen by typing.
        if model != ACTIVE_MODEL:
            self._send(409, json.dumps({"error": "stale model", "active": ACTIVE_MODEL}))
            return
        # Answering loads the model if it wasn't resident, so from here it is ours to
        # evict when the user switches away.
        APP_MODELS.add(model)

        # Claim the session, then work on a private copy. A stopped turn's thread keeps
        # running until its next write fails, so it can still be alive when the next
        # question arrives; the copy keeps the two loops from corrupting each other's
        # context, and turn_id stops the older one writing back over the newer answer.
        with _session_lock:
            messages = SESSIONS.get(sid)
            if messages is None:   # unknown sid: a new tab, or one from before a restart
                messages = SESSIONS[sid] = seed_session(req.get("restore"))
            else:                  # move to the end: asking counts as recently used
                SESSIONS[sid] = SESSIONS.pop(sid)
            turn_id = TURNS[sid] = TURNS.get(sid, 0) + 1
            while len(SESSIONS) > MAX_SESSIONS:
                stale = next(iter(SESSIONS))
                del SESSIONS[stale]
                TURNS.pop(stale, None)
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
            for ev in engine.research_events(model, work, min_sources=depth, effort=effort,
                                             temperature=temperature):
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


def adopt_resident_model():
    """Start up believing whatever Ollama already has in RAM.

    The server can be restarted (a code edit, a crash) while a model is resident. If we
    just assumed DEFAULT_MODEL, the UI would show that name while a different model held
    the RAM, and the first question would evict it. Prefer the default when it is the one
    loaded, else adopt whichever is. Best effort: if Ollama is down we keep the default."""
    global ACTIVE_MODEL
    try:
        resident = loaded_models()
    except Exception:
        return
    if resident and DEFAULT_MODEL not in resident:
        ACTIVE_MODEL = resident[0]
        print(f"adopting resident model {ACTIVE_MODEL} (default {DEFAULT_MODEL} is not loaded)",
              flush=True)
    # Whatever we adopt counts as ours from here on, so switching away frees it rather
    # than leaving it resident beside its replacement.
    if ACTIVE_MODEL in resident:
        APP_MODELS.add(ACTIVE_MODEL)


def main():
    adopt_resident_model()
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
