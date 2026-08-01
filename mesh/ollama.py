#!/usr/bin/env python3
"""
ollama — a small, honest client.

Two entry points, and the distinction matters:

  complete()  blocking, returns a string. Used by router and reader, which run
              MANY AT ONCE from a thread pool. Nothing streams, nothing is shown.

  stream()    yields (channel, text) as tokens arrive. Used only by the
              synthesiser, the one call whose output the user watches live.

Everything here is `requests` + stdlib. No Ollama SDK, matching the rest of the
repo's no-dependency habit.
"""

import json
import threading

import requests

from . import config

# One session per thread. requests.Session is not documented as thread-safe, and
# the reader fans out across a pool, so give each worker its own connection pool
# rather than sharing one and hoping.
_local = threading.local()


def _session():
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


class OllamaError(RuntimeError):
    pass


def _payload(model, messages, options, keep_alive, stream, think=None):
    body = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": keep_alive or config.KEEP_ALIVE_ACTIVE,
        # Sent identically on every call for a given model. See config.py: an
        # options mismatch makes Ollama drop and reload the model.
        "options": options,
    }
    if think is not None:
        # `think` is a per-request flag, not a load option, so unlike `options` it
        # can vary between calls without evicting the model. Turning it off for the
        # router and reader is worth ~17x: qwen3.6 spent 8s reasoning its way to the
        # same one-line JSON it produces in 0.5s flat. Classification and extraction
        # are recall, not deliberation. Only synthesis earns the thinking budget.
        body["think"] = think
    return body


def complete(model, messages, options, keep_alive=None, timeout=300, think=None):
    """Blocking chat completion. Returns the assistant's text (thinking stripped)."""
    body = _payload(model, messages, options, keep_alive, stream=False, think=think)
    try:
        r = _session().post(
            f"{config.OLLAMA_HOST}/api/chat", json=body, timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise OllamaError(f"{model}: {e}") from e
    return (data.get("message") or {}).get("content", "") or ""


def stream(model, messages, options, keep_alive=None, timeout=600, think=None):
    """Yield (channel, text) chunks. channel is "thinking" or "answer".

    Ollama puts reasoning-model chain-of-thought in a separate `thinking` field, so
    we can surface it as a dim side-channel without it polluting the answer text.
    """
    body = _payload(model, messages, options, keep_alive, stream=True, think=think)
    try:
        with _session().post(
            f"{config.OLLAMA_HOST}/api/chat", json=body, timeout=timeout, stream=True
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = chunk.get("message") or {}
                if msg.get("thinking"):
                    yield "thinking", msg["thinking"]
                if msg.get("content"):
                    yield "answer", msg["content"]
                if chunk.get("done"):
                    return
    except requests.RequestException as e:
        raise OllamaError(f"{model}: {e}") from e


def touch(model, options, keep_alive):
    """Reset a model's keep-alive timer without generating anything.

    An empty `messages` list makes Ollama load (or refresh) the model and return
    immediately. Options must match what real calls send, or this *causes* the
    reload it was meant to avoid.
    """
    try:
        _session().post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [],
                "stream": False,
                "keep_alive": keep_alive,
                "options": options,
            },
            timeout=120,
        )
    except requests.RequestException:
        pass  # best effort; a missed keep-alive costs a reload, never correctness


def warm(keep_alive=None):
    """Load both tiers so the first question doesn't pay for it."""
    ka = keep_alive or config.KEEP_ALIVE_ACTIVE
    touch(config.BIG_MODEL, config.BIG_OPTIONS, ka)
    touch(config.SMALL_MODEL, config.SMALL_OPTIONS, ka)


def list_models():
    """Names of locally available models, for the UI dropdown."""
    try:
        r = _session().get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except (requests.RequestException, KeyError, ValueError):
        return []
