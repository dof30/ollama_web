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


def compact_turn(messages, base_len, question, answer):
    """Collapse a just-finished turn down to a clean question + answer.

    During research the loop appends a lot of bulk to `messages` — tool-call JSON,
    and OBSERVATION messages holding whole fetched pages (up to ~1.5k tokens each).
    That detail is essential *while* researching but only crowds the context window
    afterwards. Left in place across turns it silently overflows num_ctx: Ollama
    then truncates from the top (losing the system prompt) and can run out of room
    to generate, which is what makes a multi-turn session "fold" mid-answer.

    So after each turn we throw away everything the loop added and keep just the
    user's question and the final answer — the model still remembers the thread of
    the conversation, at a fraction of the tokens. We also cap how many past turns
    we retain. `base_len` is len(messages) *before* this turn's question was added.
    """
    del messages[base_len:]
    messages.append({"role": "user", "content": question})
    if answer:
        messages.append({"role": "assistant", "content": answer})
    keep = 1 + MAX_HISTORY_TURNS * 2  # system prompt + N (user, assistant) pairs
    if len(messages) > keep:
        del messages[1:len(messages) - (keep - 1)]  # drop oldest turns, keep system


# The research loop itself now lives in agent.research_events — a single source of
# truth shared by the terminal renderer (agent.run_agent) and this web engine. We
# wrap it at call time (rather than aliasing the function object) so a hot-reload of
# agent.py is picked up on the very next request without reloading this module too.
def research_events(model, messages, min_sources=0, should_wrap_up=None):
    """Yield research events for the web UI. The event contract is documented in
    this file's header; the implementation is agent.research_events.
    should_wrap_up: zero-arg predicate polled each step — true means the user hit
    "Answer now", so stop researching and write up what's already gathered."""
    yield from agent.research_events(model, messages, min_sources=min_sources,
                                     should_wrap_up=should_wrap_up)
