# mesh — two models, one answer

The research web app, rebuilt. Both models live in RAM at once on this machine
(AMD AI Max 395, 128 GB unified memory):

| role | model | RAM | job |
|---|---|---|---|
| primary | `gpt-oss:120b-fast` | ~65 GB | **reads the pages, and writes the answer** |
| assistant | `qwen3.6:35b` | ~23 GB | triage, screening links, JSON verdicts, (later) vision |

~88 GB resident, ~30 GB left for KV cache and the OS.

## The mistake worth learning from

The first version split them by *size*: big model for judgment, small model for
volume — qwen read every page and handed gpt-oss its notes. It was fast, and the
answers got noticeably worse in a way that took a while to name.

qwen is a coding and tool-use model. It is **terse by disposition**. Every page
reached gpt-oss already compressed into clipped bullet-facts, so the model that
actually writes had no *language* left to work from — no phrasing, no quotes, no
texture. It had run out of fuel. The evidence was technically all there; the prose
had been squeezed out en route.

So the split is by **kind of work**, not size. The model that writes is the model
that reads — it never sees a summary of something it could have read itself. qwen
gets the mechanical jobs it is genuinely better at: classifying, emitting JSON,
screening a list of links before the expensive reader touches them.

Measured effect: evidence notes went from 282–642 chars (qwen reading) to
1344–1932 chars (gpt-oss reading).

Worth knowing: **the two cannot compute simultaneously.** One GPU — two models
measured at 1.16x versus running back to back. The benefit isn't overlap, it's that
qwen is ~3x smaller, so the many small structured jobs cost ~3x less GPU time,
leaving that time for reading and writing.

## Why the old loop was slow

`webapp/` drove one 120b model through a ReAct loop, `MAX_STEPS = 12`, **one tool
call per step**. Each fetched page (6000 chars of raw HTML text) was appended to the
chat transcript, so by step 8 the model was reasoning over ~48 000 characters of nav
bars and cookie banners.

Four costs followed. Decoding slows as the KV cache grows, so late steps crawl. The
cache breaks whenever history is rewritten. KV RAM balloons. And — the one that
actually explains the behaviour — **a model swimming in junk context loses the
thread and searches again.** The bloated context was not a symptom of over-searching.
It was the cause.

## What changed

**0. Follow-ups are resolved first.** "Can *it* handle CSI cameras better than the
S3?" becomes "Can the **ESP32-S31** handle…" before anything else runs. Without this
the searches, the pages read and the answer all silently research whatever nouns
happen to be in the sentence — a bug that produced a confident, well-cited answer
about entirely the wrong chip.

**1. Note-taking, not summarising.** gpt-oss reads each page at *low* reasoning
effort and writes generous notes for its own later use — keeping the source's own
sentences, the numbers, and the reasoning. Not a summary: you cannot write a good
paragraph from a bad summary.

**2. The evidence ledger** (`state.py`) — the important one. The old loop used the
*chat transcript* as memory: append-only, so round 5's prompt contained rounds 1-4.
Here memory is a **structured object**, and every model call renders a *fresh,
bounded* prompt from it. Round 5's prompt is the same size as round 2's. O(n²) → O(n).
Extra rounds became cheap enough to be worth having.

**3. Parallel where it helps.** Eight pages come from eight different hosts, so they
are fetched concurrently (~1.5 s for 5 pages, measured). Search *queries* all hit one
endpoint, so they are serialised with jitter — same word "parallel", opposite
conclusion, depending on whether the load lands on one shoulder or many.

**4. Triage.** A 1.5 s router call decides how much the question deserves before any
of this starts.

**5. Reasoning effort where it pays, not everywhere.** Reading measured *faster and
better* at low effort (10.9 s / 1049 chars out) than high (20.2 s / 776 chars) —
reasoning hard about a page produces reasoning instead of evidence. Synthesis is the
opposite and keeps the full budget. On qwen, turning thinking off was 17x for
identical JSON.

**6. One page is one step.** A middle version batched eight pages per "round" and
read three per model call, copying how cloud platforms fan out. That was wrong for
this machine. There is one GPU and Ollama serialises, so a wide step is not faster —
it just becomes a long opaque stall, with coarser control over when to stop. Narrow
steps give steady visible progress, a place to re-decide after every page, and a
budget that means something.

**7. The budget is pages, not rounds.** That is what the user is actually spending,
it maps straight to time, and it needs no explaining. The selector sits next to Send:

| effort | pages read |
|---|---|
| Auto | the router decides — 3 for a lookup, 8 for a real question |
| Light | 3 |
| Normal | 6 |
| Thorough | 12 |
| Exhaustive | 20 |

Every 3 pages it re-asks "is this enough?", and stops early when it is. A wall-clock
ceiling (`MESH_MAX_SECONDS`, 420s) is checked between pages — writing still follows,
so the worst case lands under 9 minutes.

Net: **one** synthesis call per question, not twelve serial ones.

## Searching is the point

The router's question is deliberately *not* "could I answer this myself?" — it almost
always could, and that is the trap. It is **"would a source make this more correct or
more trustworthy?"** Anything a person maintains and can therefore change — an app's
menus, an API, a price, who holds a job, a version number — gets searched even when
the model feels sure. Only arithmetic, grammar, word meanings, settled maths, code
semantics, opinions and the conversation itself skip the web. When torn, it searches.

A wasted search costs a few seconds. A stale answer delivered confidently costs your
trust in every other answer.

## Run it

```bash
research-web              # start + open browser
research-web --status
research-web --stop
research-web --warm       # preload both models
```

Or directly: `python3 -m mesh.server` (port 8770, localhost only).

## Layout

```
config.py     roles -> models, and the options that must not vary
ollama.py     chat / stream / keep-alive
search.py     pluggable backends (DuckDuckGo now; Brave or SearXNG later)
fetch.py      parallel retrieval + text extraction
state.py      the evidence ledger  <- the heart of it
roles.py      router / reader / assess / synth, and their prompts
pipeline.py   triage -> rounds -> synthesis, as an event stream
server.py     thin HTTP shell, localhost-only
history.py    write-only turn recorder (unchanged, same DB)
static/       the UI
```

## Tuning

| var | default | meaning |
|---|---|---|
| `MESH_BIG_MODEL` | `gpt-oss:120b-fast` | the judgment model |
| `MESH_SMALL_MODEL` | `qwen3.6:35b` | the volume model |
| `MESH_BIG_NUM_CTX` | `16384` | big model context (was 32768; raw pages no longer land here) |
| `MESH_SMALL_NUM_CTX` | `8192` | small model context — it only ever sees one page |
| `MESH_FETCH_PER_ROUND` | `8` | pages read per round, concurrently |
| `MESH_MAX_ROUNDS` | `3` | rounds, not steps — 3 rounds reads up to 24 sources |
| `MESH_PAGE_CHARS` | `8000` | chars kept per page *before* distillation |
| `MESH_SEARCH_BACKEND` | `ddg` | search backend |
| `MESH_PORT` | `8770` | |

Note: `webapp/` pinned the big model at `num_ctx 32768`. Running both apps
alternately therefore costs one ~30 s reload each way.

## Session history

Unchanged and still write-only: `~/.local/share/research-web/history.db`, never
exposed over HTTP. Read it from the terminal:

```bash
python3 mesh/history.py          # last 20 turns
python3 mesh/history.py show 12
python3 mesh/history.py stats
```

## Known limits

- **Ollama serialises generation.** `OLLAMA_NUM_PARALLEL=3` was tried and measured
  at 1.01x — Ollama silently falls back to one slot because 88 GB of resident models
  leaves no room for 3x KV cache. The reader pool is still set to 3 so it costs
  nothing now and pays off if a smaller pair of models is ever used. Page *fetching*
  is genuinely parallel regardless.
- DuckDuckGo is scraped, not an API, so queries are throttled. `search.py` is written
  so a keyed backend can be added without touching anything above it.
