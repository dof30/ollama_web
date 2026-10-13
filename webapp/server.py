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
import contextlib
import importlib
import json
import os
import re
import sys
import threading
import time
import sqlite3
import urllib.parse
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
# muse-glimmer:30b (~17 GB), not gpt-oss:120b (~61 GB): a new tab with nothing loaded
# opens on this, and its first question pulls it into RAM — 17 GB is a load that fits
# beside anything, where 61 GB on a busy box is the difference between a slow answer
# and a frozen machine. (Was gemma4:26b until it was removed from this box.) Override
# with RESEARCH_MODEL.
DEFAULT_MODEL = os.environ.get("RESEARCH_MODEL", "gpt-oss:120b")   # kong, 2026-10-13

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

# The model most recently loaded or asked — what a NEW tab opens on. It used to be the one
# model every tab had to agree on, because loading a second model always evicted the
# first. Now models share the machine when they fit (see eviction_plan), so each tab
# keeps its own model and this is only a starting point. A plain string needs no lock
# (assignment is atomic under the GIL).
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

# Models pulled on this box but not wanted in the picker. Matched as name PREFIXES, so
# "qwen3.6" hides every tag of that family. This only hides — the weights stay on disk and
# `ollama run` still reaches them; it is about keeping the one control you choose from down
# to the models you actually choose between. qwen3.5 left the service on 2026-10-07: the
# 122b is outdated beside qwen3.8:27b, and at ~81 GB it was the one model that could not
# share the machine. On 2026-10-13 every Qwen left (kong: "gpt-oss and muse glimmer should
# be capable enough to deal with 99% of daily Q&As"); Qwen work lives in Visual Workbench,
# Workbench and Qwen-web. What remains (muse-glimmer:30b, gpt-oss:120b) fits side by side.
HIDDEN_MODELS = tuple(filter(None, (s.strip() for s in os.environ.get(
    "RESEARCH_HIDE_MODELS", "qwen").split(","))))

# Exact tags that survive HIDDEN_MODELS ("this family except this one build"). Empty since
# qwen3.8:27b, the one exception (2026-10-10), left with the rest of the Qwens. Exact tags
# only, never prefixes: a prefix here would silently re-open a family the hide list closed.
SHOWN_MODELS = frozenset(filter(None, (s.strip() for s in os.environ.get(
    "RESEARCH_SHOW_MODELS", "").split(","))))

# Ollama 0.40 lists a GGUF adopted from a llama.cpp install under its digest
# ("llamacpp:6e67eb…") and can list the same tag twice, once per digest. Neither is a
# model anyone picks by that name, and a picker with two identical "GPT-OSS" rows is
# worse than either: Ollama answers by NAME, so both rows reach the same weights.
DIGEST_NAME = re.compile(r":[0-9a-f]{64}$")


def list_models():
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            tags = json.load(r)
        names = [m["name"] for m in tags.get("models", [])
                 if not EMBEDDING_NAME.search(m["name"])
                 and not DIGEST_NAME.search(m["name"])
                 and (m["name"] in SHOWN_MODELS
                      or not m["name"].startswith(HIDDEN_MODELS))]
        return list(dict.fromkeys(names))   # de-duplicate, keeping Ollama's order
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


def _parent_is_ollama(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            ppid = next(l.split()[1] for l in f if l.startswith("PPid:"))
        with open(f"/proc/{ppid}/comm") as f:
            return f.read().strip() == "ollama"
    except (OSError, StopIteration):
        return False


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
            # Ollama runs its models in llama-server children of its own. Those are
            # already counted as Ollama models; listing them again here double-counted
            # the RAM and named Ollama as "holding memory outside Ollama".
            if _parent_is_ollama(pid):
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
            "resident": resident, "sizes": sizes, "reserve": RAM_RESERVE,
            "busy": sorted(busy_models())}


def admit(target):
    """May `target` be pulled into RAM without taking the machine down? (ok, detail)

    Admission control: the check has to happen BEFORE the request that causes the load,
    because a warning that arrives afterwards is not a guard. Fails closed — an unknown
    size is a refusal, since permitting it is how you find out the hard way."""
    _, available = mem_info()
    if target in loaded_models():
        return True, {"need": 0, "free": available, "frees": [], "blockers": []}
    sizes = model_sizes()
    plan, room, busy, foreign = eviction_plan(target, sizes, available)
    blockers = [s for s in llama_servers() if s["bytes"] > 0]
    detail = {"need": sizes.get(target, 0), "free": room, "reserve": RAM_RESERVE,
              "frees": plan, "blockers": blockers}
    if not sizes.get(target):
        detail["why"] = f"{target} is not installed here, so its size is unknown"
        return False, detail
    if busy:
        detail["why"] = (f"{target} would have to unload {', '.join(busy)}, which is "
                         f"answering in another tab right now — try again when it finishes")
        return False, detail
    if foreign:
        detail["why"] = (f"{target} needs the room {', '.join(foreign)} holds, and that model "
                         f"was loaded by another app — deactivate it there or in the AI Hub first")
        return False, detail
    return sizes[target] + RAM_RESERVE <= room, detail


# When each model last answered or was loaded, for least-recently-used eviction.
LAST_USED = {}


# Which models may be resident together is arithmetic, not a rule: the AI Hub keeps a
# table of every combination of models and whether their measured sizes fit (hubplan.py,
# plan.json), and this app, the other web app and the hub all ask the same table — kong
# (2026-10-11): "I trust math, not guess". Before that it was a fixed rule: only gpt-oss
# and qwen3.6 side by side, everything else one at a time. Without the hub, free memory
# alone decides.
AIHUB_DIR = os.environ.get("AIHUB_DIR", "/home/kong/aihub")


def _plan_table():
    """The hub's plan table, or None if the hub is not there."""
    try:
        if AIHUB_DIR not in sys.path:
            sys.path.append(AIHUB_DIR)        # appended: never shadows this app's modules
        import hubplan
        return hubplan.load()
    except Exception:
        return None


def eviction_plan(target, sizes=None, available=None):
    """Which resident Ollama models must go for `target` to load.
    -> (plan, room, busy, foreign)

    `room` is the memory there will be once the plan has run; `busy` is the part of the
    plan that is answering in some tab and so cannot go; `foreign` are models that may
    not stay beside `target` but that this app did not load (admit() refuses on both).

    Models share whenever they fit, by the hub's plan table: the fewest models are
    unloaded to make room — this app's own idle ones first, least recently used first.
    A model answering right now is never chosen if any plan spares it; if none does it
    is named, and admit() refuses with the reason. A model another app or program loaded
    is never pulled out from under it: this app cannot see that app's running answers,
    so the load is refused with the reason instead.
    Embedding models (Local_embedder keeps one resident) are never unloaded: they are
    small, never chat, and unloading them would break their own app.
    Split out from the eviction itself so admission control can ask what a load would
    cost without performing it."""
    sizes = model_sizes() if sizes is None else sizes
    if available is None:
        _, available = mem_info()
    others = [m for m in loaded_models() if m != target and not EMBEDDING_NAME.search(m)]
    busy_now = busy_models()
    theirs = [m for m in others if m not in APP_MODELS]
    table = _plan_table()
    if table is not None and table.knows(target):
        outside = sum(sizes.get(m, 0) for m in others if not table.knows(m)) / 2**30
        must = table.evict(others, target, keep=busy_now, avoid=theirs,
                           last_used=LAST_USED, extra_gib=outside)
        if must is None:      # only by unloading something mid-answer: name it, admit() refuses
            must = table.evict(others, target, avoid=theirs, last_used=LAST_USED,
                               extra_gib=outside) or [m for m in others if table.knows(m)]
    else:
        need, room, must = sizes.get(target, 0) + RAM_RESERVE, available, []
        for m in sorted(others, key=lambda m: (m in busy_now, m in theirs, LAST_USED.get(m, 0))):
            if room >= need:
                break
            must.append(m)
            room += sizes.get(m, 0)
    foreign = [m for m in must if m not in APP_MODELS]
    plan = [m for m in must if m in APP_MODELS]
    room = available + sum(sizes.get(m, 0) for m in plan)
    return plan, room, [m for m in plan if m in busy_now], foreign


def free_ram_for(target):
    """Unload whatever eviction_plan says must go before `target` loads; say what went."""
    gone = []
    for m in eviction_plan(target)[0]:
        try:
            unload_model(m)
            gone.append(m)
        except Exception:
            pass          # best effort; Ollama's own timer still bounds it
        APP_MODELS.discard(m)
    return gone


# Loads are planned one at a time. Two tabs loading two different models at once would
# each be admitted against the same free memory, and both would be wrong.
_load_lock = threading.Lock()


def ensure_loaded(model):
    """Make `model` resident if it is not, under admission control. -> (ok, detail)

    On success detail carries "unloaded" (what had to make room); on refusal it is
    admit()'s detail plus "msg", the sentence to show the user. Loading is the same
    empty-message call as set_keep_alive, so the model arrives warm without generating a
    token; the long timeout is the point — a 120B cold load is ~30s, a bigger one minutes.
    Order matters: freeing first means the load isn't racing the old model for memory."""
    global ACTIVE_MODEL
    LAST_USED[model] = time.monotonic()
    # Already resident is the common case and needs no lock: a tab asking holds its
    # model's turn lock, and eviction never touches a busy model. Without this, asking
    # a loaded model would queue behind another tab's 30s cold load of a different one.
    if model in loaded_models():
        APP_MODELS.add(model)
        ACTIVE_MODEL = model
        return True, {"unloaded": []}
    with _load_lock:
        if model in loaded_models():
            APP_MODELS.add(model)
            ACTIVE_MODEL = model
            return True, {"unloaded": []}
        ok, why = admit(model)
        if not ok:
            gib = lambda b: round(b / 2**30, 1)
            held = ", ".join(f"{b['name']} ({gib(b['bytes'])} GB)" for b in why["blockers"])
            msg = why.get("why") or (
                f"{model} needs {gib(why['need'])} GB and only {gib(why['free'])} GB is free")
            if held and not why.get("why"):
                msg += f" — {held} is holding memory outside Ollama"
            why["msg"] = msg
            return False, why
        unloaded = free_ram_for(model)
        set_keep_alive(model, agent.keep_alive_for(model) or IDLE_KEEP_ALIVE, timeout=600)
        APP_MODELS.add(model)
        ACTIVE_MODEL = model
        return True, {"unloaded": unloaded}


# ---------- one research turn per model at a time ----------
# Ollama runs each loaded model in ONE runner with ONE context slot (`-np 1`), and a
# research turn is not one request but a dozen: think, search, read, think again. Two
# tabs asking the same model at once therefore took turns on that slot step by step,
# each step throwing out the other conversation's cached context and restoring its own
# from a checkpoint. That is where two simultaneous questions to muse-glimmer came back
# with each other's thinking in them and both answers wrecked. Parallel answers were
# never the goal on this box, so the fix is the plain one: a turn holds its model until
# it is finished, and a second tab waits — visibly — for the first answer before its
# own starts. Different models have different runners and still answer side by side.
_turn_locks = {}
_turn_locks_guard = threading.Lock()


def turn_lock(model):
    with _turn_locks_guard:
        return _turn_locks.setdefault(model, threading.Lock())


@contextlib.contextmanager
def borrow_model(model):
    """Load and hold a helper model for the end of another model's turn -> (ok, why).

    The picture card uses this after a text-only model (gpt-oss) has written its answer:
    muse-glimmer comes in through the same admission door as any load (never evicting the
    busy asker; refused with the reason if it won't fit) and under its own turn lock, so it
    can't share its one context slot with a tab answering on it. It does not become the
    header's active model — it is a guest in this turn, not a switch.

    Descend: a helper this borrow had to load is unloaded again as soon as the job is
    done. Two resident models cost GPU memory for both (kong, 2026-10-13: "a descend
    mechanism to load and unload secondary model"). One that was already resident — a tab
    is using it as its own model — is left as it was."""
    global ACTIVE_MODEL
    lock = turn_lock(model)
    if not lock.acquire(timeout=10):
        yield False, f"{model} is answering in another tab"
        return
    loaded_here = False
    try:
        prev = ACTIVE_MODEL
        loaded_here = model not in loaded_models()
        try:
            ok, why = ensure_loaded(model)
        finally:
            ACTIVE_MODEL = prev
        yield ok, ("" if ok else why["msg"])
    finally:
        if loaded_here and model in loaded_models():
            try:
                unload_model(model)
                APP_MODELS.discard(model)
            except Exception:
                pass          # best effort; Ollama's own keep_alive timer still bounds it
        lock.release()


def busy_models():
    """Models with a research turn in flight right now; never evicted, never unloaded."""
    with _turn_locks_guard:
        return {m for m, lock in _turn_locks.items() if lock.locked()}


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
# conversation sid -> the tab id that owns it. A tab keeps its sid in sessionStorage, and
# DUPLICATING a tab (Duplicate tab, Ctrl/middle-click a link to the app) copies
# sessionStorage, sid and all. Both pages were then one conversation on the server: each
# page's model read the other page's questions as its own history, and the two turns
# superseded each other so one answer was silently dropped. The first live tab to use a
# sid owns it; any other tab is told to start a fresh one (keeping what it shows).
SID_OWNER = {}
_presence_lock = threading.Lock()
_tabs_warm = None            # last applied state: True=some tab, False=none, None=unknown


def claim_sid(sid, tab):
    """May `tab` speak in conversation `sid`? Claims it if nobody live already has.

    A request without a tab id (the CLI, a page cached from before this) claims nothing
    and is never refused. A reloaded tab gets a new tab id but must keep its sid: its
    old id said goodbye on pagehide (or ages out of PRESENCE), so the sid is free again."""
    if not tab or not sid:
        return True
    now = time.monotonic()
    with _presence_lock:
        owner = SID_OWNER.get(sid)
        if owner and owner != tab and now - PRESENCE.get(owner, -TAB_TTL * 2) <= TAB_TTL:
            return False
        SID_OWNER[sid] = tab
        return True


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
        for sid, tab in list(SID_OWNER.items()):
            if tab not in PRESENCE:
                del SID_OWNER[sid]
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
                                        "speech": speech.available(),
                                        # per-capacity step limits, for the cups' hover text
                                        "steps": {e: agent.steps_for(e)
                                                  for e in ("low", "medium", "high")},
                                        "extension": [agent.STEP_EXTENSION,
                                                      agent.MAX_EXTENSIONS]}))
        elif path == "/api/state":
            # What is holding RAM right now, across BOTH runtimes, for the header.
            self._send(200, json.dumps(machine_state()), extra={"Cache-Control": "no-store"})
        elif path == "/api/history" or path.startswith("/api/history/"):
            self._history(path)
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _history(self, path):
        """The Alt+H history window: read-only, and only for this app's own page.

        Until 2026-10-13 nothing here read history at all (kong's July rule: terminal
        only). He asked for a browser behind a shortcut, keeping the page focused. The
        Host check (do_GET) stops DNS rebinding; the custom header stops another site's
        page from reading this through the user's browser — a cross-origin request may
        not add it without a CORS preflight, which this server never approves. Deleting
        stays terminal-only (history.py purge)."""
        if self.headers.get("X-Research-History") != "1":
            self._send(403, json.dumps({"error": "forbidden"}))
            return
        q = urllib.parse.parse_qs(self.path.partition("?")[2])
        nocache = {"Cache-Control": "no-store"}
        try:
            if path == "/api/history":
                body = history.browse(term=(q.get("q") or [""])[0][:200],
                                      before=(q.get("before") or [None])[0],
                                      limit=(q.get("limit") or [50])[0])
            else:
                body = history.fetch(path.rsplit("/", 1)[1])
                if body is None:
                    self._send(404, json.dumps({"error": "no such turn"}), extra=nocache)
                    return
        except (ValueError, sqlite3.Error) as e:
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}), extra=nocache)
            return
        self._send(200, json.dumps(body), extra=nocache)

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
            sid = str(req.get("sid") or "")[:64]
            if tab:
                with _presence_lock:
                    if self.path == "/api/ping":
                        PRESENCE[tab] = time.monotonic()
                    else:
                        PRESENCE.pop(tab, None)
                        for s_, t_ in list(SID_OWNER.items()):
                            if t_ == tab:
                                del SID_OWNER[s_]
            apply_presence()  # react at once when the first tab opens / the last closes
            # A duplicated tab finds out on its very first heartbeat, before it can ask.
            taken = self.path == "/api/ping" and not claim_sid(sid, tab)
            self._send(200, json.dumps({"ok": True, "active": ACTIVE_MODEL,
                                        "sid_taken": taken}))
            return
        if self.path == "/api/unload":
            req = self._read_json()
            if req is None:
                return
            name = (req.get("model") or "").strip()
            if not name:
                self._send(400, json.dumps({"error": "no model"}))
                return
            if name in busy_models():
                self._send(409, json.dumps({"error": f"{name} is answering in another tab "
                                                     "— unload it when that answer finishes"}))
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
            # `dry` answers "what would this cost?" without loading anything, so the page
            # only asks for confirmation when a switch really deactivates something.
            if req.get("dry"):
                ok, why = admit(new)
                self._send(200, json.dumps({"ok": ok, "frees": why.get("frees", []),
                                            "resident": new in loaded_models()}))
                return
            # Ask before loading, not after. This is the door: a model that cannot fit
            # is refused here, while refusing is still cheap. 409 = "understood, but
            # not in this state" — the UI shows the reason inline.
            try:
                ok, why = ensure_loaded(new)
            except Exception as e:
                self._send(502, json.dumps({"error": f"{type(e).__name__}: {e}"}))
                return
            if not ok:
                self._send(409, json.dumps({"error": why["msg"], "admission": why}))
                return
            self._send(200, json.dumps({"ok": True, "unloaded": why["unloaded"],
                                        "loaded": new, "active": new}))
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
        # One conversation, one tab. A duplicated tab arrives with its source's sid; it
        # is told to start its own instead of writing into the other page's thread.
        if not claim_sid(sid, str(req.get("tab") or "")[:64]):
            self._send(409, json.dumps({"error": "this conversation belongs to another tab",
                                        "sid_taken": True}))
            return

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

        def emit(ev):
            self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        # Hold this model for the whole turn (see turn_lock). While another tab's answer
        # is still running on it, say so, and keep writing a comment frame so a page
        # closed while waiting is noticed and gives up its place instead of running.
        lock = turn_lock(model)
        try:
            if not lock.acquire(blocking=False):
                # Usually another tab; sometimes this tab's own just-stopped turn, which
                # lets go at its next write — so the wording names neither.
                emit({"type": "wait", "text": f"{model} is finishing another answer — yours starts right after"})
                while not lock.acquire(timeout=5):
                    self.wfile.write(b": waiting\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        try:
            self._answer(model, sid, turn_id, messages, work, question, depth, effort,
                         temperature, emit)
        finally:
            lock.release()

    def _answer(self, model, sid, turn_id, messages, work, question, depth, effort,
                temperature, emit):
        """Run one research turn on a model this request already holds."""
        final_text = ""
        sources = []
        aborted = False
        t0 = time.time()
        try:
            # Answering may need the model loaded first — through the same admission
            # door as a switch, since any tab can now ask any model.
            ok, why = ensure_loaded(model)
            if not ok:
                emit({"type": "error", "text": why["msg"]})
                emit({"type": "done"})
                return
            for ev in engine.research_events(model, work, min_sources=depth, effort=effort,
                                             temperature=temperature, borrow=borrow_model):
                if ev.get("type") == "final":
                    final_text = ev.get("text", "")
                elif ev.get("type") == "source":
                    sources.append(ev.get("url", ""))
                emit(ev)
        except (BrokenPipeError, ConnectionResetError):
            aborted = True  # browser navigated away / stopped the run
        except Exception as e:
            aborted = True  # Ollama refused or died mid-load; tell the page if it is there
            try:
                emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
            except Exception:
                pass
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
        # Only models this app's picker offers. Another app's model (Qwen-web's qwen3.6,
        # the embedder) must not become the header's model here — nor count as ours,
        # or switching away would unload it from under that app.
        offered = set(list_models())
        resident = [m for m in loaded_models() if m in offered]
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
