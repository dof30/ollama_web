#!/usr/bin/env python3
"""
events — the wire protocol between the pipeline and any front-end.

Deliberately kept compatible with the old webapp/engine.py contract so the
existing HTML UI keeps working unchanged. Two new types are additive, and a
client that ignores them still renders correctly:

    {"type":"phase",  "name":str, "detail":str}   which stage we're in
    {"type":"note",   "url":str, "title":str, "text":str}   one distilled source

The rest are as before:

    {"type":"step",         "n":int}
    {"type":"thinking",     "text":str}
    {"type":"answer_chunk", "text":str}
    {"type":"tool_call",    "tool":str, "label":str}
    {"type":"observation",  "tool":str, "n":int, "ok":bool, "preview":str}
    {"type":"source",       "url":str}
    {"type":"notice",       "text":str}
    {"type":"final",        "text":str}
    {"type":"error",        "text":str}
    {"type":"done"}
"""


def step(n):
    return {"type": "step", "n": n}


def phase(name, detail=""):
    return {"type": "phase", "name": name, "detail": detail}


def thinking(text):
    return {"type": "thinking", "text": text}


def answer_chunk(text):
    return {"type": "answer_chunk", "text": text}


def tool_call(tool, label):
    return {"type": "tool_call", "tool": tool, "label": label}


def observation(tool, n, ok, preview):
    return {"type": "observation", "tool": tool, "n": n, "ok": ok, "preview": preview}


def source(url):
    return {"type": "source", "url": url}


def note(url, title, text):
    return {"type": "note", "url": url, "title": title, "text": text}


def notice(text):
    return {"type": "notice", "text": text}


def final(text):
    return {"type": "final", "text": text}


def error(text):
    return {"type": "error", "text": text}


def done():
    return {"type": "done"}
