# research — web-capable local LLMs from the terminal

Gives your Ollama models (gemma4, qwen, anything) a real capability they don't
have on their own: **searching the live web, reading pages, and pulling arXiv
papers.** No web UI, no API keys, no accounts. Just the command line, like
`ollama run` — but the model can now go look things up.

## How it works

```
you ──▶ research ──▶ gemma4 (Ollama)
                       │  decides it needs info, emits a tool call
                       ▼
              ┌──────────────────┐
              │ web_search  (DuckDuckGo)   │
              │ web_fetch   (read any page)│
              │ arxiv_search (papers)      │
              └──────────────────┘
                       │  observation fed back to the model
                       ▼
              gemma4 reads, searches again if needed, writes a cited report
```

It drives the model with a plain-text **ReAct loop** — the model writes a small
JSON line to call a tool, gets the results back, and repeats until it has enough
to answer. This is why it works with **gemma** (gemma's chat template does not
support Ollama's native tool-calling API, so that path is avoided entirely).

## Use it

One-shot research report:
```bash
python3 agent.py "Compare ACT, Diffusion Policy, and pi0 for robot arm grasping"
```

Interactive, like `ollama run` but with web access:
```bash
python3 agent.py            # REPL. commands: /model <name>  /reset  /exit
```

Pick a model (default is `gemma4:latest`):
```bash
python3 agent.py -m gemma4:31b "latest humanoid robot vision models in 2026"
RESEARCH_MODEL=gemma4:26b python3 agent.py     # or via env
```

### Global command (already installed)

`research` is symlinked into `~/.local/bin` (which is on your PATH), so it works
from any directory in any terminal, and survives reboots. Just type:
```bash
research "how does pi0 differ from Octo?"   # uses default model gpt-oss:120b-fast
research -m gemma4:31b                       # faster interactive session
research --help
```
The launcher pins the miniforge python that has the deps, so it works even when
conda `base` isn't active. Nothing to start on boot — Ollama runs as a systemd
service and `gpt-oss:120b-fast` is a persistent model.

## Web UI (new — recommended for daily use)

The terminal is great for quickly *testing* a model, but a poor *home* for real
use: `Ctrl+C` is SIGINT (so it can't be copy), text selection is the emulator's
job, and shell quoting bites you (`research why ... (widely discussed)` errors in
bash before Python even runs). The web UI fixes all of that at once — it's a
single-page browser front-end over the *same* research engine.

```bash
research-web                      # one command: starts the server (if needed) AND opens the browser
research-web --status             # is it running, and where?
research-web --stop               # stop the background server
```

`research-web` is on your PATH (symlinked into `~/.local/bin`) and is safe to run
repeatedly — it finds an already-running instance instead of starting a duplicate.
There's also a **desktop icon**: search "Local Research" in your app launcher
(installed at `~/.local/share/applications/research-web.desktop`); clicking it
does the same start-and-open in one go. Manual way still works too:

```bash
python3 webapp/server.py          # foreground, Ctrl+C to stop
```

- **No shell quoting** — type parentheses, quotes, anything into the box.
- **Native selection & copy**, plus a Copy button on each answer.
- **Live research trace** — watch every search/fetch, with clickable sources.
- **Markdown answers** (headings, tables, links) instead of raw text.
- **Auto / Quick / Deep** depth selector and a model dropdown (from `ollama list`).
- **Stdlib only** — no Flask/FastAPI to install; `http.server` + one HTML file.

Architecture: `webapp/engine.py` runs the ReAct loop as an *event stream*;
`webapp/server.py` streams those events to the browser; `agent.py` (the CLI) is
untouched and still works for quick terminal tests. Audio in/out is the next
phase. Env vars: `RESEARCH_WEB_PORT` (default 8765), `RESEARCH_WEB_HOST`.

## Tuning (all optional, via env vars)

| var | default | meaning |
|-----|---------|---------|
| `RESEARCH_MODEL` | `gpt-oss:120b-fast` | which Ollama model to use |
| `RESEARCH_NUM_CTX` | `32768` | context window (bigger = reads more, uses more RAM) |
| `RESEARCH_MAX_STEPS` | `12` | max tool calls before it must answer |
| `RESEARCH_MIN_SOURCES` | `0` | forced depth gate; `0` = off, model matches effort to the question (`--deep` sets 5) |
| `RESEARCH_FETCH_CHARS` | `6000` | chars kept per fetched page |
| `RESEARCH_MAX_HISTORY_TURNS` | `8` | past Q/A turns kept per web session (see below) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |

**Why a session doesn't overflow:** during a turn the model reads whole pages
(each up to `RESEARCH_FETCH_CHARS`), which is a lot of tokens. If every turn's raw
pages piled up, a few deep questions would blow past `RESEARCH_NUM_CTX`, Ollama
would truncate from the top (losing the system prompt), and answers would start
cutting off mid-stream. So after each turn the web UI **compacts** it down to a
clean question + answer and drops the raw-page bulk — the model still remembers the
conversation, at a fraction of the tokens. `New chat` clears the session entirely.
(The `agent.py` CLI doesn't compact; use `/reset` there for a long REPL session.)

## Notes
- Bigger gemma (`gemma4:31b`) follows the tool protocol more reliably and writes
  better reports; `gemma4:latest` is fastest for quick lookups.
- Sources are real pages the model searched/fetched — it's told never to invent
  URLs — but always sanity-check important facts against the cited links.
- Everything runs locally except the web requests the tools make on your behalf.
