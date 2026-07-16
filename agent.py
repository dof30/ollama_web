#!/usr/bin/env python3
"""
research — give your local Ollama models the ability to search the live web,
read pages, and pull arXiv papers, straight from the terminal.

No web UI. No API keys. Works with gemma4 (or any Ollama model) because it
drives the model with a plain-text ReAct loop instead of Ollama's native
tool-calling API (which gemma's chat template does not render reliably).

Usage:
    python3 agent.py "your question"            # one-shot research report
    python3 agent.py                            # interactive REPL (like `ollama run`)
    python3 agent.py -m gemma4:31b "question"   # pick a model
    RESEARCH_MODEL=gemma4:26b python3 agent.py   # or via env

REPL commands:  /model <name>   /reset   /exit
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

try:
    import readline  # enables arrow keys, Home/End, mid-line editing, and history in the REPL
except ImportError:
    readline = None  # not available on some platforms; input() still works, just without editing

# ---------- config ----------
OLLAMA_HOST   = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("RESEARCH_MODEL", "gpt-oss:120b-fast")
NUM_CTX       = int(os.environ.get("RESEARCH_NUM_CTX", "32768"))  # headroom for deep multi-fetch turns
MAX_STEPS     = int(os.environ.get("RESEARCH_MAX_STEPS", "12"))
FETCH_CHARS   = int(os.environ.get("RESEARCH_FETCH_CHARS", "6000"))
MIN_SOURCES   = int(os.environ.get("RESEARCH_MIN_SOURCES", "0"))  # 0 = no gate, model decides; --deep/--depth to force
MAX_NUDGES    = int(os.environ.get("RESEARCH_MAX_NUDGES", "4"))   # times we push it to go deeper
UA = "Mozilla/5.0 (X11; Linux x86_64) research-agent/1.0"

# ---------- terminal colors ----------
class C:
    dim    = "\033[2m"
    cyan   = "\033[36m"
    green  = "\033[32m"
    yellow = "\033[33m"
    red    = "\033[31m"
    bold   = "\033[1m"
    reset  = "\033[0m"
if not sys.stdout.isatty():
    for _k in list(vars(C)):
        if not _k.startswith("_"):
            setattr(C, _k, "")

def c(text, color):
    return f"{color}{text}{C.reset}"

# ========================================================================
# TOOLS  — the real capability the model gains
# ========================================================================

def web_search(query, max_results=5):
    """Search the web via DuckDuckGo. Returns a list of {title, url, snippet}."""
    from ddgs import DDGS
    max_results = max(1, min(int(max_results), 10))
    out = []
    for r in DDGS().text(query, max_results=max_results):
        out.append({
            "title": r.get("title", ""),
            "url":   r.get("href") or r.get("url") or r.get("link", ""),
            "snippet": r.get("body") or r.get("snippet", ""),
        })
    return out

def web_fetch(url):
    """Download a page and return its readable text (truncated)."""
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "xml" not in ctype and resp.text[:100].strip().startswith("{"):
        return resp.text[:FETCH_CHARS]  # looks like JSON/plain
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n"))
    text = re.sub(r"[ \t]+", " ", text).strip()
    body = text[:FETCH_CHARS]
    if len(text) > FETCH_CHARS:
        body += "\n...[truncated]..."
    return f"TITLE: {title}\nURL: {url}\n\n{body}"

def arxiv_search(query, max_results=5):
    """Search arXiv. Returns a list of {title, url, authors, summary}."""
    max_results = max(1, min(int(max_results), 10))
    q = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"http://export.arxiv.org/api/query?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        raw = r.read()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    out = []
    for e in root.findall("a:entry", ns):
        title = (e.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
        summ  = (e.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
        link  = (e.findtext("a:id", "", ns) or "").strip()
        authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
        out.append({
            "title": title,
            "url": link,
            "authors": ", ".join(a for a in authors if a)[:200],
            "summary": summ[:500],
        })
    return out

TOOLS = {
    "web_search":   lambda a: web_search(a.get("query", ""), a.get("max_results", 5)),
    "web_fetch":    lambda a: web_fetch(a.get("url", "")),
    "arxiv_search": lambda a: arxiv_search(a.get("query", ""), a.get("max_results", 5)),
}

# ========================================================================
# SYSTEM PROMPT
# ========================================================================

SYSTEM = """You are a research assistant running locally on the user's machine.
You can reach the LIVE internet through tools. Your training data is frozen and
may be outdated, so for anything current, factual, technical, or that you are
not fully certain about, you MUST use tools rather than guess.

TOOLS (call one at a time):
- web_search    -> {"tool":"web_search","args":{"query":"...","max_results":5}}
- web_fetch     -> {"tool":"web_fetch","args":{"url":"https://..."}}
- arxiv_search  -> {"tool":"arxiv_search","args":{"query":"...","max_results":5}}

HOW TO ACT — match your effort to the question:
1. Casual, conversational, or simple questions you already know cold: just
   answer directly. No tools, no multi-step analysis, no ceremony.
2. Questions needing current or verifiable facts: use tools. To call one, reply
   with ONLY a single JSON object exactly like the forms above, and nothing
   else on that turn. No prose around it. IMPORTANT: writing ABOUT a tool call
   ("I need to search for X", "Use web_search") does NOT call it — only the raw
   JSON object does. If you decide to search, your entire reply must BE that JSON.
3. You will then receive an OBSERVATION with the results. Read it.
4. For a quick lookup, one good source is enough — web_fetch it rather than
   trusting a search snippet, then answer.
5. Save the thorough treatment (several searches from different angles,
   multiple independent sources, cross-checking) for when the user explicitly
   asks for deep research or the stakes clearly demand it.
6. Write the FINAL ANSWER as normal prose (NOT JSON). If you used web sources,
   cite them inline as [1], [2], ... and list the full URLs under a "Sources:"
   heading; note any disagreements between sources.

RULES:
- Never invent URLs, facts, numbers, or citations. Only cite pages you actually
  fetched or that appeared in search results.
- Prefer primary sources (papers, official docs, repos) over blog summaries.
- Use as many tool calls as the question deserves and no more. When you can
  answer with reasonable confidence, stop and answer.
"""

# ========================================================================
# OLLAMA (streaming, via stdlib — no extra dependency)
# ========================================================================

def ollama_chat_stream(model, messages):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.4, "num_ctx": NUM_CTX},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_HOST + "/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msg = obj.get("message", {})
            # Reasoning models (gpt-oss) split output into a "thinking" channel and
            # a "content" channel. We yield both, tagged, because such models will
            # sometimes emit a tool call (or the whole answer) in "thinking" while
            # leaving "content" empty — if we ignored it, the turn would look blank.
            if msg.get("content"):
                yield ("content", msg["content"])
            if msg.get("thinking"):
                yield ("thinking", msg["thinking"])
            if obj.get("done"):
                break

# ========================================================================
# TOOL-CALL PARSING  (robust to gemma's messy output)
# ========================================================================

def _loads_lenient(span):
    """json.loads, with a fallback pass for the messes local models emit. Only the
    fallback is lenient — valid JSON always parses on the first, strict try, so we
    never risk mangling a well-formed object."""
    try:
        return json.loads(span)
    except Exception:
        pass
    # Common breakage: over-escaped quotes in a *bare* object, e.g.
    #   {"query": \"esp32-p4 dsi\"}   (the \" is invalid outside a JSON string)
    # and trailing commas before a closing brace/bracket. Repair and retry once.
    repaired = span.replace('\\"', '"')
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except Exception:
        return None

def _balanced_json_objects(text):
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                obj = _loads_lenient(text[start:i + 1])
                if obj is not None:
                    objs.append(obj)
                start = None
    return objs

def extract_tool_call(text):
    """Return (tool, args) if the model asked for a tool, else None."""
    calls = [o for o in _balanced_json_objects(text)
             if isinstance(o, dict) and "tool" in o and o.get("tool") in TOOLS]
    if not calls:
        return None
    call = calls[-1]
    return call["tool"], (call.get("args") or {})

def format_observation(tool, result):
    if isinstance(result, list):
        lines = []
        for i, item in enumerate(result, 1):
            if tool == "arxiv_search":
                lines.append(f"[{i}] {item['title']}\n    {item['url']}\n"
                             f"    authors: {item['authors']}\n    {item['summary']}")
            else:
                lines.append(f"[{i}] {item['title']}\n    {item['url']}\n    {item['snippet']}")
        return "\n".join(lines) if lines else "(no results)"
    return str(result)

# ========================================================================
# AGENT LOOP
# ========================================================================

def _stream_turn(model, messages):
    """Stream one model completion, yielding {"type":"answer_chunk"|"thinking"} as
    tokens arrive, then a final {"_turn": (content, thinking)} sentinel. Reasoning
    models split output across a content and a thinking channel and may hide the
    answer or the tool call in either, so we surface both and let the caller decide.
    This is the one streaming primitive behind both the CLI and the web UI."""
    content, thinking = "", ""
    for kind, chunk in ollama_chat_stream(model, messages):
        if kind == "content":
            content += chunk
            yield {"type": "answer_chunk", "text": chunk}
        else:
            thinking += chunk
            yield {"type": "thinking", "text": chunk}
    # Keep the actual answer/tool-call JSON in history; fall back to the reasoning
    # channel when the model left content empty so the tool call isn't lost.
    messages.append({"role": "assistant", "content": content or thinking})
    yield {"_turn": (content, thinking)}

SYNTH_FROM_EVIDENCE = (
    "Enough research — do NOT call any more tools. Using ALL the evidence you "
    "gathered above, write the FINAL ANSWER now as prose, with inline [n] "
    "citations and a 'Sources:' list of the URLs you actually used. If some "
    "details stayed uncertain, say so.")
SYNTH_FROM_KNOWLEDGE = (
    "You did not run any web search. Do NOT try to use tools now and do NOT say "
    "you need to search. Answer the question directly from your own knowledge, as "
    "prose. If you are not fully certain, say so in one short line at the end.")

def _synthesis(model, messages, instruction=SYNTH_FROM_EVIDENCE):
    """Force a written answer (no more tools): stream it as answer chunks, then end
    with a `final` event carrying the whole text."""
    messages.append({"role": "user", "content": instruction})
    content = thinking = ""
    for ev in _stream_turn(model, messages):
        if "_turn" in ev:
            content, thinking = ev["_turn"]
        else:
            yield ev
    yield {"type": "final", "text": (content or thinking).strip()}


def research_events(model, messages, min_sources=MIN_SOURCES):
    """The research loop as a STREAM OF EVENTS — the single source of truth behind
    both the terminal renderer (run_agent) and the web UI (webapp/engine.py). It
    drives the ReAct tool loop, mutating `messages` in place so follow-ups keep
    context. A depth gate (min_sources > 0) blocks a premature answer until enough
    distinct pages are read. Event types: step / thinking / answer_chunk /
    tool_call / observation / source / notice / final / error / done."""
    read_domains = set()    # distinct sites actually fetched — this is our "depth"
    unread_urls = []        # urls seen in search results but not yet fetched
    nudges = 0
    stalls = 0              # turns that returned only reasoning — no answer, no call
    try:
        for step in range(1, MAX_STEPS + 1):
            yield {"type": "step", "n": step}

            content = thinking = ""
            for ev in _stream_turn(model, messages):
                if "_turn" in ev:
                    content, thinking = ev["_turn"]
                else:
                    yield ev

            call = extract_tool_call(content)
            if not call and not content.strip():
                # Content came back empty — a reasoning model (gpt-oss) may have hidden
                # the tool call, or its whole answer, in the thinking channel instead.
                call = extract_tool_call(thinking)

            if not call:
                # Model wants to finalize. Enforce depth unless it has read enough.
                if len(read_domains) < min_sources and nudges < MAX_NUDGES:
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
                # Empty content, no tool call anywhere: only internal reasoning came
                # back. Nudge it to actually emit the call (or the answer) rather than
                # forcing synthesis, which would answer from stale memory without ever
                # searching. Bounded so a stuck model still terminates.
                if stalls < MAX_NUDGES:
                    stalls += 1
                    yield {"type": "notice", "text": "described a tool call but didn't emit it — nudging"}
                    messages.append({"role": "user", "content": (
                        "You described a tool call in words but did not emit it, so nothing ran. "
                        "Reply with ONLY the JSON object and nothing else, e.g.:\n"
                        '{"tool": "web_search", "args": {"query": "<what to look up>"}}\n'
                        "Fill in the query and send just that. If you truly don't need the web, "
                        "write the FINAL ANSWER as prose instead.")})
                    continue
                # Nudges exhausted: if it researched, synthesize from that; else answer
                # from its own knowledge (never surface raw "I need to search" reasoning).
                instr = SYNTH_FROM_EVIDENCE if read_domains else SYNTH_FROM_KNOWLEDGE
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
                result = TOOLS[tool](args)
                obs = format_observation(tool, result)
                n = len(result) if isinstance(result, list) else 1
                yield {"type": "observation", "tool": tool, "n": n, "ok": True,
                       "preview": f"{n} result(s) in {time.time() - t0:.1f}s"}
                if tool == "web_fetch":  # count it as a real source read
                    dom = urllib.parse.urlparse(args.get("url", "")).netloc
                    if dom:
                        read_domains.add(dom)
                        yield {"type": "source", "url": args.get("url", "")}
                elif isinstance(result, list):  # remember URLs we could fetch later
                    unread_urls.extend(it["url"] for it in result if it.get("url"))
            except Exception as e:
                obs = f"ERROR running {tool}: {type(e).__name__}: {e}"
                yield {"type": "observation", "tool": tool, "n": 0, "ok": False, "preview": obs}

            messages.append({"role": "user", "content":
                             f"OBSERVATION from {tool}:\n{obs}\n\n"
                             f"Continue: call another tool (JSON only) or write the FINAL ANSWER."})

            # Once we've pushed it enough, stop looping and make it write the report.
            if nudges >= MAX_NUDGES and len(read_domains) >= 1:
                break

        instr = SYNTH_FROM_EVIDENCE if read_domains else SYNTH_FROM_KNOWLEDGE
        for ev in _synthesis(model, messages, instr):
            yield ev
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "text": f"{type(e).__name__}: {e}"}
        yield {"type": "done"}


def run_agent(model, messages, min_sources=MIN_SOURCES):
    """Render the shared research event stream to the terminal and return the final
    answer text. The loop itself lives in research_events — this is just the CLI's
    renderer, the terminal twin of the web UI's event handler. `messages` is mutated
    in place so REPL follow-ups keep context."""
    answer = ""
    read = 0                 # distinct sources read so far (mirrors the depth gate)
    at_line_start = True     # so section markers always break cleanly off a stream line

    def newline():
        nonlocal at_line_start
        if not at_line_start:
            sys.stdout.write("\n")
            at_line_start = True

    for ev in research_events(model, messages, min_sources):
        t = ev.get("type")
        if t == "step":
            newline()
            gate = f", read {read}/{min_sources}" if min_sources > 0 else ""
            sys.stdout.write(c(f"\n  ┄ thinking (step {ev['n']}/{MAX_STEPS}{gate}) ┄\n", C.dim))
            at_line_start = True
        elif t in ("thinking", "answer_chunk"):
            sys.stdout.write(c(ev["text"], C.dim))
            at_line_start = ev["text"].endswith("\n")
        elif t == "tool_call":
            newline()
            sys.stdout.write(c(f"\n  ⚙ {ev['tool']}({ev.get('label', '')})\n", C.cyan))
            at_line_start = True
        elif t == "observation":
            newline()
            ok = ev.get("ok")
            sys.stdout.write(c(f"  {'✓' if ok else '✗'} {ev.get('preview', '')}\n",
                               C.green if ok else C.red))
            at_line_start = True
        elif t == "source":
            read += 1
        elif t == "notice":
            newline()
            sys.stdout.write(c(f"  ⤴ {ev['text']}\n", C.yellow))
            at_line_start = True
        elif t == "final":
            answer = ev.get("text", "")
        elif t == "error":
            newline()
            sys.stdout.write(c(f"  ⚠ {ev['text']}\n", C.red))
            answer = answer or ev["text"]
            at_line_start = True
        sys.stdout.flush()
    newline()
    return answer

# ========================================================================
# ENTRY POINTS
# ========================================================================

def new_conversation():
    return [{"role": "system", "content": SYSTEM}]

def check_model(model):
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            tags = json.load(r)
        names = [m["name"] for m in tags.get("models", [])]
        if model not in names and names:
            print(c(f"note: '{model}' not in `ollama list`. Available: {', '.join(names)}", C.yellow))
    except Exception:
        print(c(f"warning: can't reach Ollama at {OLLAMA_HOST} — is it running?", C.red))

def repl(model, min_sources=MIN_SOURCES):
    depth_label = str(min_sources) if min_sources > 0 else "auto"
    print(c(f"research agent · model={model} · depth={depth_label} · web_search + web_fetch + arxiv_search", C.bold))
    print(c("ask anything. commands: /model <name>  /depth <n>  /reset  /exit\n", C.dim))
    messages = new_conversation()
    while True:
        try:
            q = input(c("you › ", C.green)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not q:
            continue
        if q in ("/exit", "/quit"):
            print("bye"); return
        if q == "/reset":
            messages = new_conversation(); print(c("(context cleared)", C.dim)); continue
        if q.startswith("/depth"):
            parts = q.split()
            if len(parts) > 1 and parts[1].isdigit():
                min_sources = int(parts[1])
                print(c(f"(depth → {min_sources} sources)" if min_sources else "(depth → auto: model decides)", C.dim))
            else:
                cur = str(min_sources) if min_sources > 0 else "auto (model decides)"
                print(c(f"(depth is {cur}; /depth 0 = auto, /depth 5 = deep research)", C.dim))
            continue
        if q.startswith("/model"):
            parts = q.split()
            if len(parts) > 1:
                model = parts[1]; check_model(model); print(c(f"(model → {model})", C.dim))
            else:
                print(c(f"(model is {model})", C.dim))
            continue
        messages.append({"role": "user", "content": q})
        try:
            answer = run_agent(model, messages, min_sources)
        except KeyboardInterrupt:
            print(c("\n(interrupted)", C.yellow)); continue
        print(c("\n─── answer ───", C.bold))
        print(answer.strip(), "\n")

def one_shot(model, question, min_sources=MIN_SOURCES):
    messages = new_conversation()
    messages.append({"role": "user", "content": question})
    answer = run_agent(model, messages, min_sources)
    print(c("\n─── answer ───", C.bold))
    print(answer.strip())

def _take_opt(args, *names):
    for name in names:
        if name in args:
            i = args.index(name)
            val = args[i + 1] if i + 1 < len(args) else None
            del args[i:i + 2]
            return val
    return None

USAGE = f"""research — give your local Ollama models live web access, from the terminal.

USAGE:
  research "your question"        one-shot research report (prints, then exits)
  research                        interactive REPL (like `ollama run`, with web tools)

OPTIONS:
  -m, --model <name>   Ollama model to use (default: {DEFAULT_MODEL})
  --depth <n>          force it to READ at least n distinct sources before answering
                       (default: 0 = no gate, the model matches effort to the question)
  --deep               shortcut for --depth 5 (thorough research mode)
  --shallow            shortcut for --depth 1 (force at least one page read)
  -h, --help           show this help

REPL commands:  /model <name>   /depth <n>   /reset   /exit

EXAMPLES:
  research "latest humanoid robot VLA models in 2026"
  research --deep -m gpt-oss:120b-fast "compare ACT, Diffusion Policy, and pi0"
  research -m gemma4:31b            # faster interactive session

ENV: RESEARCH_MODEL, RESEARCH_MIN_SOURCES, RESEARCH_NUM_CTX, OLLAMA_HOST"""

def main():
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(USAGE)
        return
    model = _take_opt(args, "-m", "--model") or DEFAULT_MODEL
    min_sources = MIN_SOURCES
    depth_opt = _take_opt(args, "--depth")
    if depth_opt and depth_opt.isdigit():
        min_sources = int(depth_opt)
    if "--deep" in args:
        min_sources = 5; args.remove("--deep")
    if "--shallow" in args:
        min_sources = 1; args.remove("--shallow")
    question = " ".join(args).strip()

    check_model(model)
    if question:
        one_shot(model, question, min_sources)
    else:
        repl(model, min_sources)

if __name__ == "__main__":
    main()
