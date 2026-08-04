#!/usr/bin/env python3
"""
engine — the research loop as an EVENT STREAM.

Same ReAct research behaviour as the terminal agent, but instead of printing to
a TTY it yields structured event dicts. Any front-end (the web UI here, or the
old CLI) can consume these. This is the whole point of the rebuild: the engine
stops being welded to the terminal.

It reuses the tools, prompt, Ollama client and parsing from agent.py so there is
a single source of truth for how research actually works.

Event types yielded by research_events():
  {"type":"step",        "n":int}                     a new turn begins
  {"type":"thinking",    "text":str}                  streamed reasoning (dim channel)
  {"type":"answer_chunk","text":str}                  streamed answer content
  {"type":"tool_call",   "tool":str, "label":str}     model asked for a tool (clears this
                                                       step's answer buffer on the client)
  {"type":"observation", "tool":str, "n":int, "ok":bool, "preview":str}
  {"type":"source",      "url":str}                   a page actually fetched (counts as depth)
  {"type":"notice",      "text":str}                  a nudge / gate message
  {"type":"final",       "text":str}                  the complete answer (authoritative)
  {"type":"error",       "text":str}
  {"type":"done"}
"""

import os
import sys

# Import the existing agent as a library. Importing does NOT run its CLI (guarded
# by __name__ == "__main__"), so we inherit its tools/prompt/client for free.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent  # noqa: E402


def new_conversation():
    return agent.new_conversation()


# How many past (question, answer) turns to keep in a session's context. Bounds
# growth so a long conversation can't slowly refill the window. Env-tunable.
MAX_HISTORY_TURNS = int(os.environ.get("RESEARCH_MAX_HISTORY_TURNS", "8"))


def start_turn(messages, question):
    """A private working copy of the conversation for one turn, ending in `question`.

    The research loop appends a lot of bulk as it goes — tool-call JSON, and
    OBSERVATION messages holding whole fetched pages — and it must never do that to
    the live session list. Two turns can overlap: press Stop and re-ask, and the
    browser unlocks Send immediately while the stopped turn's server thread lives on
    until its next write fails. Sharing one list means those two loops interleave
    their scratch work, and whichever finishes first truncates the other's context
    out from under it mid-generation.

    Each message dict is copied, not just the list: agent.py stamps the date onto the
    last message and rewrites the system prompt in place, which would otherwise reach
    through the copy and edit the session's own messages.
    """
    work = [dict(m) for m in messages]
    work.append({"role": "user", "content": question})
    return work


def commit_turn(messages, question, answer):
    """Add one finished turn to the session as a clean question + answer pair.

    Every bit of research scratch work stays behind in that turn's working copy. Left
    in the session it would silently overflow num_ctx: Ollama then truncates from the
    top (losing the system prompt) and can run out of room to generate, which is what
    used to make a long session "fold" mid-answer. The question stored here is the
    clean one from the request, not the date-stamped copy the loop worked with.

    A turn that produced no answer (Stop, or an error) is dropped WHOLE. Keeping the
    question alone would leave a user message with no reply, so the next question
    lands as a second user message in a row and the model stops reading it as a
    follow-up — that is what made a stopped-then-reworded question arrive with the
    earlier conversation apparently forgotten.
    """
    if not answer:
        return
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    keep = 1 + MAX_HISTORY_TURNS * 2  # system prompt + N (user, assistant) pairs
    if len(messages) > keep:
        del messages[1:len(messages) - (keep - 1)]  # drop oldest turns, keep system


# The research loop itself now lives in agent.research_events — a single source of
# truth shared by the terminal renderer (agent.run_agent) and this web engine. We
# wrap it at call time (rather than aliasing the function object) so a hot-reload of
# agent.py is picked up on the very next request without reloading this module too.
def research_events(model, messages, min_sources=0, effort=None, temperature=None):
    """Yield research events for the web UI. The event contract is documented in
    this file's header; the implementation is agent.research_events.
    effort: the intelligence level chosen in the UI ("low"/"medium"/"high"), handed
    straight to the model as its reasoning effort. None lets the CLI's source-count
    heuristic pick one instead.
    temperature: sampling temperature for this turn; None uses agent.TEMPERATURE."""
    yield from agent.research_events(model, messages, min_sources=min_sources,
                                     effort=effort, temperature=temperature)
