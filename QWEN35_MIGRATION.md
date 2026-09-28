# Rebuilding `Local_LLM_web` after gpt-oss:120b

**Target:** one person, one iPhone client, one Strix Halo box (Ryzen AI Max+ 395, 128 GB unified, Linux).
**Four goals:** image input · iPhone frontend · parallel web search · **better English prose**.
**Date:** 23 Aug 2026. Verify version-sensitive claims before acting — they are flagged ⚠.

---

## 0. The one-paragraph verdict

**Stay on Ollama, and stop asking one model to do both jobs.** Ollama's *official* `qwen3.5:122b`
ships vision tensors inside the main GGUF, so `/api/chat` takes images with no mmproj sidecar and no
llama.cpp rewrite — that preserves `keep_alive` stickiness, `/api/ps` adoption, the switch-model RAM
handoff, native tool calling, the `think` channel, hot reload, the history DB. But Qwen's English
prose is not its strength, so **route the final write-up to Gemma 4** (§1.5). Your `_synthesis()` is
already a separate, tool-free model call — this is a ~20-line change, not an architecture.
**Switch the GPU backend to ROCm** (Vulkan hangs outright on Qwen3.5). **Do not use an off-the-shelf
iOS app** — Reins and Enchanted talk straight to Ollama and would bypass your entire research engine;
make `webapp/static/index.html` an installable PWA behind `tailscale serve`. The parallel-search fix
is mostly already written: your loop at `agent.py:973` already accepts a batch of calls, it just runs
them one at a time.

---

## 1. The research/vision model: Qwen3.5-122B-A10B

| Property | Value |
|---|---|
| Params | 122B total / 10B active (MoE) |
| Architecture | Hybrid Gated DeltaNet (linear attn) + full attention, 3:1 ratio, sparse MoE |
| Context | 262 144 native (256K), YaRN-extensible to 1M |
| Modality | Native vision-language, early-fusion — text, image, video in |
| Modes | thinking / non-thinking |
| Licence | Apache 2.0 |
| MMLU-Pro / GPQA-D / SWE-bench-V | 86.1 / 85.5 / 72.4 |
| BrowseComp | 69.0 with context folding, 78.6 discard-all |

The 3:1 hybrid attention matters more than the benchmarks for you: only a quarter of layers carry a
real KV cache, so long-context research turns cost far less memory here than on gpt-oss. That is what
buys back the room the bigger weights take. BrowseComp at 69–78 is the number that justifies keeping
it in the *research* seat specifically — that benchmark is exactly your loop.

**Ollama:** `qwen3.5:122b` = **81 GB**, 256K context, Text+Image, capabilities `vision tools thinking`.
Vision is embedded; there is no projector to wire up.

**llama.cpp alternative** (only if you leave Ollama) — Unsloth GGUFs plus `mmproj-F16.gguf` (909 MB):

| Quant | Size | Note |
|---|---|---|
| UD-Q4_K_XL | 77 GB (3 shards) | best quality that fits |
| UD-IQ3_S | 46.6 GB | **frees ~60 GB for context** |
| UD-Q2_K_XL | 41.8 GB | Unsloth's stated floor |
| UD-IQ1_M | 34.2 GB | too lossy for agentic tool use |

---

## 1.5 The English problem — and the fix

gpt-oss writes stiff, hedged English. Qwen3.5 comprehends English fine but its prose reads
translated; article writing is not what it was optimised for. Neither is what you want narrating a
research report.

**You do not have to solve this with one model.** Look at your own engine:

```python
# agent.py:816 — no tools passed, prose is the only exit
def _synthesis(model, messages, instruction=SYNTH_FROM_EVIDENCE, ...):
```

Research and writing are *already* two separate phases in your code. They currently share a model
only because they always have. Split them:

| Phase | Wants | Model |
|---|---|---|
| Research + vision | tool calling, long ctx, speed, image understanding | `qwen3.5:122b` |
| Final write-up (`_synthesis`) | English prose | `gemma4:31b` |

Your own README already reached this conclusion for the old stack: *"Bigger gemma (`gemma4:31b`)
follows the tool protocol more reliably and **writes better reports**."* You observed it; you just
never routed to it.

### Why Gemma 4 for the writer seat

- **Ollama-official, vision on every tag** — `gemma4:31b` 20 GB (q4) / 34 GB (q8_0) / 63 GB (bf16),
  256K context, Text+Image. Capabilities: `vision tools thinking audio`.
- Dense 30.7B, all params active — slow per token on a bandwidth-bound box, but the synthesis call is
  *one* generation at the end of a run, not the twelve tool round-trips. This is the cheapest possible
  place to spend tokens/sec.
- ⚠ Note your README's claim that gemma "does not support Ollama's native tool-calling API" is now
  **stale** — the library lists `tools` for gemma4. `model_supports_tools()` probes `/api/show`, so it
  will pick this up on its own. Irrelevant for the writer seat (no tools passed) but worth knowing.

If you want the writer faster, `gemma4:26b` is an **MoE with ~4B active** — 18 GB (q4) / 28 GB (q8_0),
same 256K, same vision. Much faster generation, one size-class down in quality.

### Co-resident or swap?

```
qwen3.5:122b            81 GB
gemma4:31b   (q4)       20 GB   → 101 GB total, ~17 GB left for KV. Workable at 32K ctx.
gemma4:26b   (q4)       18 GB   →  99 GB total. Fastest writer.
gemma4:31b   (q8_0)     34 GB   → 115 GB. Does not fit alongside Qwen.
```

Two options, both already supported by code you wrote:

1. **Co-resident** — keep both warm. Only works at q4 for the writer, and leaves the KV budget thin.
   Your `apply_presence()` / `STICKY_MODELS` machinery handles keeping two models armed.
2. **Swap at synthesis** — let `switch_model()` do the handoff it was built for. Costs one model load
   (~30 s) per finished article. For a workflow that produces one report, that is nothing, and it lets
   you run the writer at q8_0 where prose quality actually lives.

Start with (2). It is strictly simpler and gives better prose. Move to (1) only if the 30 s bothers you.

### Implementation sketch

```python
WRITER_MODEL = os.environ.get("RESEARCH_WRITER_MODEL", "")   # "" = same model, current behaviour

def _synthesis(model, messages, instruction=SYNTH_FROM_EVIDENCE, effort=None, temperature=None):
    writer = WRITER_MODEL or model
    # ... swap if writer != model and co-residency is off
```

Two things to get right:

- **Sampling follows the model, not the request.** The writer needs Gemma's profile, not Qwen's — see
  §3. Do not hand `presence_penalty: 1.5` to Gemma.
- **The writer inherits the research transcript.** It has never seen the tool calls that produced it,
  which is fine (`_synthesis` passes no tools), but `SYNTH_FROM_EVIDENCE` should be re-read with fresh
  eyes: it currently assumes the reader is the same model that did the searching.

### The single-model alternative, and why it is not the default

**Mistral Small 4 (119B-A6B, `2603`)** is the best one-model answer on paper: Apache 2.0, native
vision, 256K, per-request reasoning effort, **6.5B active** (faster than Qwen's 10B on this box), and
a Western lab whose English is native rather than translated. Mistral folded Magistral (reasoning),
Pixtral (vision) and Devstral (agentic coding) into it. Unsloth GGUFs exist with mmproj —
UD-Q4_K_XL 74.2 GB, UD-IQ3_S 44.4 GB.

Two reasons it is not the recommendation:

1. ⚠ **Not in Ollama's official library** (ollama#15142 is still open). Importing the GGUF + mmproj
   means hitting exactly the vision-loader path that fails today — the reported error for
   Mistral-Small-3.1 was `unknown projector type: pixtral`. So Mistral Small 4 realistically means
   moving to `llama-server`, which is the large rewrite this whole plan avoids.
2. ⚠ **Agentic reports conflict.** One source has it "matching or surpassing GPT-OSS 120B across
   benchmarks while generating significantly shorter outputs"; another calls its agentic results "on
   the lower end" and says it is "not well suited to autonomous agent chains" — which is precisely
   what your loop is. Both are secondary sources. Unresolved.

Worth an evening of testing via llama.cpp before committing. Not worth rebuilding the server around
until it's tested.

---

## 2. GPU backend: ROCm, not Vulkan

This is the highest-risk item and it is already known-broken.

> **Vulkan hangs on inference for Qwen3.5.** GPU pegs at 100%, zero tokens out.
> Tracked as ollama#14487 and ollama#14509. ROCm works natively on gfx1151.

`/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=11.5.1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KEEP_ALIVE=5m"
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

- **Do NOT** set `OLLAMA_VULKAN=1`.
- **Do NOT** set `HIP_VISIBLE_DEVICES=-1` — that disables ROCm entirely.
- Keep `OLLAMA_KEEP_ALIVE` at `5m`, not `-1`. Your `agent.py:84` comment already explains why
  (`-1` once wedged a model with a year-2318 expiry) and `keep_alive_for()` supplies the 45m exception
  per-model. Don't let a systemd drop-in undo that design.

Verify:

```bash
journalctl -u ollama -b | grep -i "inference compute"
# want: library=ROCm compute=gfx1151 ... total="111.5 GiB" available="110.5 GiB"
ollama ps          # PROCESSOR must read GPU, not CPU
```

BIOS: set the UMA/framebuffer carve-out **small (512 MB)** and let GTT do the work. A large fixed VRAM
carve-out is the classic Strix Halo trap — it can leave Linux seeing only ~31 GB of the 128.

```bash
cat /sys/class/drm/card*/device/mem_info_gtt_total
```

If GTT comes back short, raise it with `amdgpu.gttsize=` or `amdttm.pages_limit=` on the kernel cmdline.

### Memory budget

```
ROCm reports available          110.5 GiB
qwen3.5:122b weights            81 GB   (Ollama's figure; GB vs GiB unstated)
+ gemma4:31b q4 (if co-resident) 20 GB
remaining for KV + compute      thin — read the real number off `ollama ps`
```

⚠ **gpt-oss:120b-fast (~65 GB) and qwen3.5:122b (81 GB) cannot be resident together.** Your
`switch_model()` at `server.py:168` already frees-then-loads for exactly this reason. But
`STICKY_MODELS` (`agent.py:86`) still names `gpt-oss:120b-fast` — update it, or you'll keep a 65 GB
model warm for 45 minutes that you no longer use.

**On unified memory an OOM is not graceful.** The amdgpu driver spams
`Not enough memory for command submission!` and can take the desktop down with it. Find the ceiling
deliberately.

---

## 3. Sampling: per-model profiles

Your existing tuning actively works against Qwen. `agent.chat_options()` (`agent.py:604`) returns:

```python
{"temperature": 1.0, "top_p": 1.0, "num_ctx": NUM_CTX}
```

`top_p: 1.0` was a deliberate gpt-oss choice — and **it silently clobbers Ollama's baked-in
`top_p: 0.95` for qwen3.5**. Ollama ships `{"temperature": 1, "top_k": 20, "top_p": 0.95}` in that
model's params blob; any key you send in `options` overrides it.

| | temp | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Qwen3.5 thinking, general | 1.0 | 0.95 | 20 | 0.0 | **1.5** |
| Qwen3.5 thinking, precise code | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Qwen3.5 non-thinking | 0.7 | 0.8 | 20 | 0.0 | 1.5 |
| gpt-oss (current) | 1.0 | 1.0 | — | — | — |

`presence_penalty: 1.5` is the one you cannot skip — without it Qwen3.5 repeats. Ollama does *not*
bake it in. ⚠ Qwen's own docs warn that raising it "may occasionally result in language mixing" —
relevant to you specifically, since Simplified Chinese leaking into English output is a reported
Qwen3.5 behaviour. If you see it, add an explicit English instruction to the system prompt rather than
dropping the penalty. (Good news: Qwen3.5's *internal reasoning* is reported to stay in English.)

Make the profile a property of the model:

```python
SAMPLING = {
    "qwen3.5": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5},
    "gemma4":  {"temperature": 1.0, "top_p": 0.95, "top_k": 64},   # ⚠ confirm against the tag's params
    "gpt-oss": {"temperature": 1.0, "top_p": 1.0},
}
```

This matters more once you split roles — the writer and the researcher are different models and must
not share a profile.

Keep the `TEMP_MIN=0.4 / TEMP_MAX=1.0` clamp; the reasoning behind it (a reasoning model below ~0.4
repeats its own thinking and re-fetches the same pages) transfers cleanly.

### Reasoning effort — verify before you trust the dial

1. ⚠ Ollama's `think` accepts a bool **or** a level string (`low`/`medium`/`high`). Confirm
   `qwen3.5:122b` takes levels, not just `true`/`false`. If bool-only, `effort_for()` (`agent.py:830`)
   collapses to a two-position switch and Quick/Auto/Deep needs re-mapping.
2. ⚠ **Some Qwen3.5 chat templates auto-disable thinking whenever tools are active** (documented as
   "Fix 19" in the community 21-fix template, QwenLM/Qwen3#1831). If Ollama's official template does
   that, your reasoning-effort dial becomes a no-op during *every research turn* — the only place it
   matters. Test: tools-enabled request at `think: "high"`, check the `thinking` channel is non-empty.

That second one would be an invisible regression. Check it before tuning anything else.

---

## 4. Image input

Ollama's `/api/chat` takes images as base64 on the message:

```json
{"role": "user", "content": "what's in this?", "images": ["<base64, no data: prefix>"]}
```

| Where | Change |
|---|---|
| `server.py:78` `MAX_BODY = 64*1024` | Too small for any photo. Add `MAX_IMAGE_BODY` (~10 MB), following the `MAX_AUDIO_BODY` precedent at line 79. |
| `server.py:457` `/api/ask` | Accept `images: [b64]` alongside `q`. Ollama takes JPEG, PNG, WebP. Validate magic bytes the way `_transcribe` validates `RIFF/WAVE` at line 351. |
| `engine.start_turn` | Attach images to the user message it appends. |
| `agent.ollama_chat_stream` | Nothing — images ride on the message dict, passed through untouched. |
| `engine.commit_turn` | **Drop the images.** Same argument as dropping fetched-page bulk: an image left in history is re-prefilled on every follow-up turn, forever. Keep a one-line text stand-in. |
| `history.record` | Store image *count*, never the bytes. |
| `index.html` | `<input type="file" accept="image/*" capture="environment">` gives the iPhone camera + photo picker for free. |

**Downscale in the browser, not on the server.** Vision token count scales with resolution, and
prefill on a ~250 GB/s bus is your bottleneck. A 12 MP iPhone photo sent raw stalls for seconds before
the first token. Resize to ~1024 px long edge via canvas before upload — ten lines, more impact than
any server-side tuning.

**System-prompt work:** `SYSTEM_NATIVE` (`agent.py:492`) has no notion of an image in the
conversation. Add a branch: *an image in the question is primary evidence — read it before deciding
whether to search; do not search for a description of something you were shown.* Without this it will
reflexively `web_search` a photo you just handed it.

---

## 5. iPhone frontend

### Why not Reins / Enchanted / Ollama Mobile

They connect **directly to Ollama's API**. Everything that makes this project yours lives *between*
the browser and Ollama: the ReAct loop and its step budget, the three tools and the concurrent preview
fetcher, the depth gate, the nudges, the loop detector, the step refund, per-turn compaction, the live
trace, the write-only history DB — and now the researcher/writer split. An off-the-shelf app gets you
a chat box with no web access and no trace.

### The architecture

```
iPhone Safari (PWA, Home Screen)
   │  https://<box>.<tailnet>.ts.net
   ▼
tailscale serve  ── terminates TLS, real cert, no open ports
   │  http://127.0.0.1:8765
   ▼
webapp/server.py  ── unchanged bind, unchanged engine
   ▼
Ollama :11434
```

```bash
tailscale serve --bg 8765
tailscale serve status
```

Tailscale's ACLs are your authentication. The server keeps binding to `127.0.0.1`.

### Four blockers

**1. `_local_ok()` rejects every request from the phone.** `server.py:261` allowlists `Host` ∈
{`127.0.0.1`, `localhost`, `[::1]`}. Behind `tailscale serve` the Host header is your tailnet name, so
everything 403s. Extend the allowlist — **do not remove the check**; the comment there is right.

```python
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]",
                 os.environ.get("RESEARCH_TAILNET_HOST", "")} - {""}
```

**2. HTTPS is not optional** — Add-to-Home-Screen, service worker, *and* `getUserMedia`. Your
dictation mic will not work on the phone over plain HTTP at all. `tailscale serve` solves all three.

**3. PWA scaffolding.** `index.html` needs `<meta name="viewport" content="width=device-width,
initial-scale=1, viewport-fit=cover">`, `apple-mobile-web-app-capable`, apple-touch-icon, a
`manifest.json` (`display: standalone`), and a trivial service worker — three more branches in `do_GET`.

**4. The hard one: iOS suspends background tabs, and your answers take minutes.**
A deep run is 300 s+; the phone locks in 30 s. iOS suspends the tab, the SSE socket dies,
`server.py:526` catches `BrokenPipeError`, sets `aborted = True`, and `commit_turn` is **skipped** —
the whole turn is discarded. You unlock to nothing.

- `/api/ask` starts a turn and returns a `turn_id` immediately.
- Buffer that turn's events server-side in a bounded ring keyed by `turn_id`.
- `GET /api/stream?turn_id=…&from=<seq>` replays from a sequence number, then tails live.
- Client reconnects on `visibilitychange`, resumes from the last `seq`.
- Commit when the **engine** finishes, not when the socket survives — move `commit_turn` off `aborted`.

This also fixes a bug you have on the desktop today: closing the lid mid-answer loses the turn.

---

## 6. Parallel web search

**Not your code.** gpt-oss emits one tool call per turn, full stop. Your loop already anticipated the
alternative — `agent.py:973`:

```python
for call_idx, (tool, args) in enumerate(calls):   # "a native model may batch several per turn"
```

Qwen3.5's default template natively supports parallel function calls, so the model side fixes itself
on switch. Three things on your side:

**(a) Execute the batch concurrently.** That loop runs sequentially — three searches cost three round
trips for nothing. Copy the pattern from `_attach_previews` (`agent.py:160`).
- **Preserve order.** Tool result messages must append in the model's original call order or the
  template mis-pairs them. Submit concurrently, collect by index.
- `step_usable` stays per-step (refund only if *every* call failed) — line 1048 is already correct.
- Yield `tool_call` / `observation` events as they complete so the trace stays live.
- ⚠ Parallel calls have a known interleaving bug in some Qwen3.5 templates (sglang#7117, "Fix 15" in
  QwenLM/Qwen3#1831). Test with a question forcing three searches; read the raw transcript.

**(b) Fix the loop detector.** `searches_in_a_row` (`agent.py:882`, incremented at 1016) counts *tool
calls* and scolds at 3. A legitimate three-query fan-out trips it instantly and gets told to stop
searching — the exact opposite of the goal. **Count turns that contained only list-tools.**

**(c) Re-budget the counters.** `MAX_STEPS = 12` / `MAX_STEPS_HIGH = 16` were measured against
one-call-per-step gpt-oss (your comment at line 54: medium 8 steps/7 calls, high 10/9). At three calls
per step, 12 steps is up to 36 tool calls. Switch the budget to **count tool calls, not steps**.

**(d) Optional: `web_search_many`.** A tool taking `queries: [str]` that fans out internally — gives
you parallel search even if template batching is flaky, and makes the capability explicit in the tool
description. ~20 lines of insurance.

---

## 7. Build order

| Phase | Work | Done when |
|---|---|---|
| **0** | ROCm override.conf, BIOS UMA 512 MB, verify GTT | `ollama ps` shows GPU, gfx1151 in logs |
| **1** | `ollama pull qwen3.5:122b`. Update `DEFAULT_MODEL`, `STICKY_MODELS`. Verify `think` levels + thinking-with-tools (§3) | text-only research run matches gpt-oss quality |
| **2** | Per-model `SAMPLING` profiles; stop sending `top_p: 1.0` to Qwen | no repetition, no language mixing |
| **3** | **`WRITER_MODEL` in `_synthesis`** — `gemma4:31b` at q8_0 via `switch_model` | side-by-side: same research, two write-ups. Read them. |
| **4** | Concurrent batch execution, loop-detector fix, tool-call budget | forced 3-query test fans out, ordered, no false scolding |
| **5** | Images: `MAX_IMAGE_BODY`, `/api/ask` field, browser downscale, `commit_turn` strip, prompt branch | photo → answer; follow-ups don't re-prefill it |
| **6** | Detached turns + resumable SSE (§5 blocker 4) | lock phone mid-answer, unlock, answer is there |
| **7** | `tailscale serve`, `ALLOWED_HOSTS`, manifest + SW + viewport | Home Screen icon, camera works, mic works |

Phase 3 is the one that answers the English complaint, and it is deliberately early — it needs nothing
from the phases after it. Phases 0–3 are all reversible in minutes.

---

## 8. Measure these, don't assume them

Your project's character is that decisions carry their measurement in a comment. Five worth earning
the same way:

1. **The writer split, blind.** Run one research question, synthesise it twice — once with Qwen, once
   with Gemma 4 — and read them without knowing which is which. This is the whole premise of §1.5; if
   Gemma doesn't win, don't build it.
2. **Quant vs context.** Q4 (81 GB, thin headroom) against a Q3-class quant (~47 GB, ~60 GB headroom)
   on the same deep question. On a bandwidth-bound box the smaller quant may simply win.
3. **Prefill cost per image** at 512 / 1024 / 1536 px long edge. Sets your downscale constant.
4. **Steps vs calls, gpt-oss vs Qwen3.5**, on a question you have already run. Your line-54 comment has
   the baseline. Parallel search should cut *steps* sharply while raising *calls* — confirm, then
   re-tune the budget.
5. **Long-context stability.** Walk `num_ctx` up and watch `dmesg` for
   `Not enough memory for command submission!`. Find the cliff on purpose, before you find it
   accidentally mid-answer.

---

## Sources

- [Qwen/Qwen3.5-122B-A10B · Hugging Face](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
- [ollama.com/library/qwen3.5 — tags, sizes, capabilities](https://ollama.com/library/qwen3.5)
- [qwen3.5:122b-a10b-q4_K_M params blob](https://ollama.com/library/qwen3.5:122b-a10b-q4_K_M/blobs/f6417cb1e269)
- [ollama.com/library/gemma4/tags](https://ollama.com/library/gemma4/tags)
- [Gemma 4 model overview — Google AI for Developers](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 31B vs Qwen3.6 35B-A3B](https://regolo.ai/gemma-4-31b-vs-qwen3-6-35b-a3b-when-to-use-which/)
- [unsloth/Mistral-Small-4-119B-2603-GGUF file list](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF/tree/main)
- [ollama#15142 — Add Mistral Small 4 to Ollama Models](https://github.com/ollama/ollama/issues/15142)
- [Mistral Small 4 local hardware guide](https://runaihome.com/blog/mistral-small-4-local-ai-hardware-guide-2026/)
- [Mistral Small 4 review (agentic caveats)](https://designforonline.com/ai-models/mistral-mistral-small-4/)
- [ollama#14855 — Strix Halo gfx1151 ROCm working guide](https://github.com/ollama/ollama/issues/14855)
- [ollama#14730 — qwen35moe + mmproj fails on imported GGUF](https://github.com/ollama/ollama/issues/14730)
- [QwenLM/Qwen3#1831 — 21-fix chat template](https://github.com/QwenLM/Qwen3/issues/1831)
- [Qwen3.5 thinking-mode parameters](https://docs.bswen.com/blog/2026-03-24-qwen35-thinking-mode-parameters/)
- [Qwen3.5 internal reasoning stays in English](https://github.com/QwenLM/Qwen3.8/issues/35)
- [llama.cpp: Vulkan vs ROCm on Strix Halo](https://www.soothill.io/blog/2026/08/03/llamacpp-vulkan-vs-rocm-strix-halo/)
- [Tailscale — Enabling HTTPS](https://tailscale.com/kb/1153/enabling-https)
