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
import time
import urllib.parse

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


def _stream_turn(model, messages):
    """Stream one completion, forwarding chunk events, and return (content, thinking).
    Mirrors agent._stream_turn but yields to the caller instead of printing."""
    content, thinking = "", ""
    for kind, chunk in agent.ollama_chat_stream(model, messages):
        if kind == "content":
            content += chunk
            yield {"type": "answer_chunk", "text": chunk}
        else:
            thinking += chunk
            yield {"type": "thinking", "text": chunk}
    # Keep the real answer / tool-call JSON in history; fall back to reasoning
    # when the model left content empty so a hidden tool call isn't lost.
    messages.append({"role": "assistant", "content": content or thinking})
    yield {"_turn": (content, thinking)}


# Two ways to force a final written answer, depending on why we're giving up:
_SYNTH_FROM_EVIDENCE = (
    "Enough research — do NOT call any more tools. Using ALL the evidence you "
    "gathered above, write the FINAL ANSWER now as prose, with inline [n] "
    "citations and a 'Sources:' list of the URLs you actually used. If some "
    "details stayed uncertain, say so.")
# Used when the model never managed to run a tool: don't pretend there's evidence,
# just have it answer from what it knows rather than surfacing "I need to search".
_SYNTH_FROM_KNOWLEDGE = (
    "You did not run any web search. Do NOT try to use tools now and do NOT say "
    "you need to search. Answer the question directly from your own knowledge, as "
    "prose. If you are not fully certain, say so in one short line at the end.")


def _synthesis(model, messages, instruction=_SYNTH_FROM_EVIDENCE):
    """Force a written answer (no more tools), streaming it as answer chunks."""
    messages.append({"role": "user", "content": instruction})
    content = thinking = ""
    for ev in _stream_turn(model, messages):
        if "_turn" in ev:
            content, thinking = ev["_turn"]
        else:
            yield ev
    yield {"type": "final", "text": (content or thinking).strip()}


def research_events(model, messages, min_sources=0):
    """Drive the research loop, yielding events. `messages` is mutated in place so
    web follow-ups keep context (same as the REPL). Set min_sources > 0 to force a
    depth gate (deep-research mode); 0 lets the model match effort to the question."""
    read_domains = set()
    unread_urls = []
    nudges = 0
    stalls = 0
    try:
        for step in range(1, agent.MAX_STEPS + 1):
            yield {"type": "step", "n": step}

            content = thinking = ""
            for ev in _stream_turn(model, messages):
                if "_turn" in ev:
                    content, thinking = ev["_turn"]
                else:
                    yield ev

            call = agent.extract_tool_call(content)
            if not call and not content.strip():
                call = agent.extract_tool_call(thinking)  # reasoning model hid it in `thinking`

            if not call:
                # Depth gate (opt-in): block a premature answer until enough read.
                if len(read_domains) < min_sources and nudges < agent.MAX_NUDGES:
                    nudges += 1
                    yield {"type": "notice",
                           "text": f"too shallow ({len(read_domains)}/{min_sources} read) — reading more"}
                    picks = [u for u in unread_urls
                             if urllib.parse.urlparse(u).netloc not in read_domains][:5]
                    hint = ("\nFetch one of these pages you already found:\n" +
                            "\n".join(f"- {u}" for u in picks)) if picks else ""
                    messages.append({"role": "user", "content": (
                        f"STOP — do not finalize yet, and do NOT run another web_search. You have "
                        f"READ only {len(read_domains)} independent source(s), which is too shallow. "
                        f"web_fetch a page from a website you have NOT read yet, then continue."
                        f"{hint}\nRespond now with a web_fetch tool call (JSON only).")})
                    continue
                if content.strip():
                    yield {"type": "final", "text": content.strip()}
                    yield {"type": "done"}
                    return
                # Only reasoning came back — nudge it to actually act, don't force
                # synthesis (that would answer from stale memory without searching).
                if stalls < agent.MAX_NUDGES:
                    stalls += 1
                    yield {"type": "notice", "text": "described a tool call but didn't emit it — nudging"}
                    messages.append({"role": "user", "content": (
                        "You described a tool call in words but did not emit it, so nothing ran. "
                        "Reply with ONLY the JSON object and nothing else, e.g.:\n"
                        '{"tool": "web_search", "args": {"query": "<what to look up>"}}\n'
                        "Fill in the query and send just that. If you truly don't need the web, "
                        "write the FINAL ANSWER as prose instead.")})
                    continue
                # Nudges exhausted. If it did some research, synthesize from that;
                # otherwise have it answer from its own knowledge (never surface the
                # raw "I need to search" reasoning as if it were the answer).
                instr = _SYNTH_FROM_EVIDENCE if read_domains else _SYNTH_FROM_KNOWLEDGE
                for ev in _synthesis(model, messages, instr):
                    yield ev
                yield {"type": "done"}
                return

            # --- a tool was requested ---
            tool, args = call
            label = args.get("query") or args.get("url") or ""
            yield {"type": "tool_call", "tool": tool, "label": label}
            t0 = time.time()
            try:
                result = agent.TOOLS[tool](args)
                obs = agent.format_observation(tool, result)
                n = len(result) if isinstance(result, list) else 1
                yield {"type": "observation", "tool": tool, "n": n, "ok": True,
                       "preview": f"{n} result(s) in {time.time() - t0:.1f}s"}
                if tool == "web_fetch":
                    dom = urllib.parse.urlparse(args.get("url", "")).netloc
                    if dom:
                        read_domains.add(dom)
                        yield {"type": "source", "url": args.get("url", "")}
                elif isinstance(result, list):
                    unread_urls.extend(it["url"] for it in result if it.get("url"))
            except Exception as e:
                obs = f"ERROR running {tool}: {type(e).__name__}: {e}"
                yield {"type": "observation", "tool": tool, "n": 0, "ok": False, "preview": obs}

            messages.append({"role": "user", "content":
                             f"OBSERVATION from {tool}:\n{obs}\n\n"
                             f"Continue: call another tool (JSON only) or write the FINAL ANSWER."})

            if nudges >= agent.MAX_NUDGES and len(read_domains) >= 1:
                break

        instr = _SYNTH_FROM_EVIDENCE if read_domains else _SYNTH_FROM_KNOWLEDGE
        for ev in _synthesis(model, messages, instr):
            yield ev
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "text": f"{type(e).__name__}: {e}"}
        yield {"type": "done"}
