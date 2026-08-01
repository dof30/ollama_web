#!/usr/bin/env python3
"""
config — who does what, and with which knobs.

The whole design rests on one idea: **a big model for judgment, a small model for
volume.** Reading ten web pages is volume. Deciding what is true is judgment. So
we run two Ollama models side by side and give each the job it is actually good at.

On the target box (AMD AI Max 395, 128 GB unified memory) both fit at once:

    gpt-oss:120b-fast   ~65 GB    judgment  (synthesis)
    qwen3.6:35b         ~23 GB    volume    (triage, page reading)
    ------------------------------------------------------
                        ~88 GB    leaves ~30 GB for KV cache and the OS

One sharp edge worth knowing: **Ollama keys a loaded model by its options.** Send
a different `num_ctx` for the same model and it reloads from scratch (~30 s). The
router and the reader share qwen3.6:35b, so they MUST share one options dict. That
is why options live per *model tier* here, not per *role*.
"""

import os

# --- Ollama endpoint ---------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# --- The two tiers -----------------------------------------------------------
# BIG   = judgment. Sees only distilled evidence, never a raw page.
# SMALL = volume. Sees one page at a time, in parallel, many times per turn.

# Two models, but NOT split the way the first attempt split them.
#
# Attempt one made qwen the reader, and it broke the answers in a way that took a
# while to see. qwen is a coding and tool-use model: terse by disposition. Every page
# reached gpt-oss already compressed into clipped notes, so the model that actually
# writes had no *language* left to write from — no phrasing, no quotes, no texture.
# It ran out of fuel. The evidence was still technically there; the prose it needed
# had been squeezed out on the way.
#
# So the division is by KIND of work, not by size:
#
#   PRIMARY  (gpt-oss)  reads real page prose, and writes. Never sees a summary of
#                       something it could have read itself.
#   ASSIST   (qwen)     the mechanical jobs it is genuinely better at: classifying,
#                       emitting JSON, screening a list of links, and later vision
#                       (gpt-oss is text-only, qwen is not).
#
# Note the two cannot compute simultaneously — one GPU, measured at 1.16x for two
# models vs 4.3s run back to back. The win is not overlap, it is that qwen is ~3x
# smaller, so the many small structured jobs cost ~3x less GPU time than they would
# on the 120b, leaving that time for reading and writing.
PRIMARY_MODEL = os.environ.get("MESH_PRIMARY_MODEL", "gpt-oss:120b-fast")
ASSIST_MODEL = os.environ.get("MESH_ASSIST_MODEL", "qwen3.6:35b")

# Back-compat aliases: BIG is the reader/writer, SMALL the helper.
BIG_MODEL = PRIMARY_MODEL
SMALL_MODEL = ASSIST_MODEL

# Context sizes. The big model's window can be far smaller than the old design's
# 32768 precisely *because* it never receives raw pages any more — distilled notes
# are ~700 chars each, so 20 of them still fit comfortably. Halving num_ctx also
# hands several GB of KV cache back, which is what buys room for the second model.
#
# Note: the old webapp/ pins this model at 32768. Alternating between the two apps
# therefore costs one ~30 s reload each way. Set MESH_BIG_NUM_CTX=32768 if you want
# to run both interchangeably without that.
BIG_NUM_CTX = int(os.environ.get("MESH_BIG_NUM_CTX", "16384"))

# The small model only ever sees ONE page (capped at PAGE_CHARS) plus a short
# instruction, so 8k is generous. Shared by router and reader — see module docstring.
SMALL_NUM_CTX = int(os.environ.get("MESH_SMALL_NUM_CTX", "8192"))

# Options are frozen per tier so a model never gets evicted mid-turn by a mismatch.
BIG_OPTIONS = {
    "num_ctx": BIG_NUM_CTX,
    "temperature": 0.4,
}
SMALL_OPTIONS = {
    "num_ctx": SMALL_NUM_CTX,
    # Extraction and classification, not prose. Near-greedy keeps the reader from
    # embellishing what the page actually said.
    "temperature": 0.1,
}

# The footgun this guards: Ollama keys a loaded model by its options. If both roles
# point at the SAME model, differing options make every role switch evict and reload
# 65 GB — turning a 2 s call into a 30 s one, silently. Only bites when someone sets
# MESH_ASSIST_MODEL to the primary, but that is exactly when it would be baffling.
if SMALL_MODEL == BIG_MODEL:
    SMALL_OPTIONS = BIG_OPTIONS
    SMALL_NUM_CTX = BIG_NUM_CTX

# Reasoning effort. gpt-oss ignores a boolean `think` — it takes a level
# ("low"/"medium"/"high"), which is why `think: false` measured *slower* than `true`
# (18.6 s vs 13.9 s): the flag did nothing and the model reasoned freely.
#
# Measured on gpt-oss:120b-fast reading one page:
#     low     10.9 s   78 chars reasoning  1049 chars out
#     medium  11.8 s  699 chars             925 chars out
#     high    20.2 s 1652 chars             776 chars out
#
# Reading rewards LOW: it is recall, and reasoning hard about a page produces
# reasoning instead of evidence — note the output got *shorter* as effort rose.
# Synthesis is the opposite; that is the judgment step and it gets the budget.
READ_THINK = os.environ.get("MESH_READ_THINK", "low")
BIG_THINK = os.environ.get("MESH_BIG_THINK", "high")
# qwen takes a boolean; its jobs here are classification and JSON, never deliberation.
SMALL_THINK = os.environ.get("MESH_SMALL_THINK", "0") not in ("0", "false", "no")

# Pages per read call. Batching cuts read latency ~3x for the same work, which is
# what makes many rounds affordable.
READ_BATCH = int(os.environ.get("MESH_READ_BATCH", "3"))

# --- Keep-alive --------------------------------------------------------------
# Never "-1" (pins RAM forever). The server shortens these when no tab is open.
KEEP_ALIVE_ACTIVE = os.environ.get("MESH_KEEP_ALIVE", "45m")
KEEP_ALIVE_IDLE = os.environ.get("MESH_KEEP_ALIVE_IDLE", "5m")

# --- Search / fetch budgets --------------------------------------------------

# Results requested per search query.
RESULTS_PER_QUERY = int(os.environ.get("MESH_RESULTS_PER_QUERY", "6"))

# How many pages we actually fetch+distill in one round. These run concurrently,
# so the wall-clock cost of 8 is roughly the cost of the slowest single page.
FETCH_PER_ROUND = int(os.environ.get("MESH_FETCH_PER_ROUND", "8"))

# Parallelism. Fetching hits many different hosts, so it can be wide. Distilling
# hits ONE Ollama instance, which serialises internally anyway — going wider than
# a few just queues requests and burns KV cache slots.
FETCH_WORKERS = int(os.environ.get("MESH_FETCH_WORKERS", "8"))
READ_WORKERS = int(os.environ.get("MESH_READ_WORKERS", "3"))

# Chars kept from a fetched page before it goes to the reader. Lowered from 8000
# now that READ_BATCH pages share one 16k-context call: 3 x 4500 leaves comfortable
# room for the instructions and three notes back. The tail of a long page is nearly
# always navigation and related-links anyway.
PAGE_CHARS = int(os.environ.get("MESH_PAGE_CHARS", "4500"))

# Hard cap on rounds for the "deep" path. Unlike the old MAX_STEPS=12 this is a
# *round* limit and each round reads up to 8 pages, so this is far more work than
# the old twelve steps ever did — the ledger keeps the prompt flat, so rounds stay
# cheap no matter how many there are. `assess` still stops early once the evidence
# is enough; this is the ceiling, not the target.
MAX_ROUNDS = int(os.environ.get("MESH_MAX_ROUNDS", "11"))

# HTTP fetch timeout, seconds.
FETCH_TIMEOUT = int(os.environ.get("MESH_FETCH_TIMEOUT", "20"))

# A stock browser UA. Not to deceive — to avoid being served a degraded page and
# to avoid tagging the user's ordinary reading as bot traffic.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0"
)

# --- Search backend ----------------------------------------------------------
# "ddg" today. The interface in search.py is deliberately thin so a keyed API
# (Brave, Tavily) or a self-hosted SearXNG can be dropped in without touching
# anything above this line.
SEARCH_BACKEND = os.environ.get("MESH_SEARCH_BACKEND", "ddg")
