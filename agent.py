#!/usr/bin/env python3
"""
research — give your local Ollama models the ability to search the live web,
read pages, and pull arXiv papers, straight from the terminal.

No web UI. No API keys. Models that support Ollama's native tool-calling API
(gpt-oss, qwen3, ...) are driven through it — the model emits tool calls in the
channel it was trained on and they arrive as parsed objects, not as JSON we
scrape out of prose. Models without tool support (gemma4, whose chat template
does not render tools reliably) automatically fall back to the original
plain-text ReAct loop.

Usage:
    python3 agent.py "your question"            # one-shot research report
    python3 agent.py                            # interactive REPL (like `ollama run`)
    python3 agent.py -m gemma4:31b "question"   # pick a model
    RESEARCH_MODEL=gemma4:26b python3 agent.py   # or via env

REPL commands:  /model <name>   /reset   /exit
"""

import contextlib
import datetime
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
DEFAULT_MODEL = os.environ.get("RESEARCH_MODEL", "muse-glimmer:30b")  # ~17 GB, fits beside anything
NUM_CTX       = int(os.environ.get("RESEARCH_NUM_CTX", "32768"))  # headroom for deep multi-fetch turns
MAX_STEPS     = int(os.environ.get("RESEARCH_MAX_STEPS", "12"))
FETCH_CHARS   = int(os.environ.get("RESEARCH_FETCH_CHARS", "6000"))
FETCH_MAX_BYTES = 4 * 1024 * 1024   # a page bigger than this is read only this far
# view_image: a model with vision looks at the picture itself; one without (gpt-oss) has
# it read by this helper, whose text transcription comes back as the observation.
# Measured on a log-scale KLD chart (2026-10-13): muse-glimmer read every asked-for
# point right and spotted the outlier (~2 min); qwen3.6 was 4x faster but put points at
# the wrong size and invented one; qwen3.8 missed a whole series. Wrong numbers cost
# more than slow ones. At 17 GB it also fits beside gpt-oss.
VISION_MODEL  = os.environ.get("RESEARCH_VISION_MODEL", "muse-glimmer:30b")
IMAGE_MAX_SIDE = 1600               # px; bigger images are scaled down before viewing
IMAGE_MAX_BYTES = 20 * 1024 * 1024
MIN_SOURCES   = int(os.environ.get("RESEARCH_MIN_SOURCES", "0"))  # 0 = no gate, model decides; --deep/--depth to force
MAX_NUDGES    = int(os.environ.get("RESEARCH_MAX_NUDGES", "4"))   # times we push it to go deeper
# Default sampling temperature. 1.0 is OpenAI's recommendation for gpt-oss — see
# chat_options() for why lower is counterproductive on a reasoning model. The UI can
# override it per request; this stays the value everything falls back to.
TEMPERATURE   = float(os.environ.get("RESEARCH_TEMPERATURE", "1.0"))
# The band the dial may move in: under ~0.4 a reasoning model repeats its own thinking
# verbatim, over 1.0 it wanders. Both ends are clamps, not suggestions.
TEMP_MIN      = float(os.environ.get("RESEARCH_TEMP_MIN", "0.4"))
TEMP_MAX      = float(os.environ.get("RESEARCH_TEMP_MAX", "1.0"))
# High reasoning effort explores more per run, so it needs a bigger budget than the 12
# that was tuned for medium. Measured on one research question, effort isolated:
# medium finished in 8 steps / 7 tool calls, high took 10 steps / 9 tool calls (+25%)
# and additionally burned a step on an invented tool. Given the same ~1.5x headroom
# over its measured need that medium enjoys, high wants ~16. Context is NOT the limit
# here — that run peaked at 43% of num_ctx, so the extra steps have room to breathe.
MAX_STEPS_HIGH = int(os.environ.get("RESEARCH_MAX_STEPS_HIGH", "16"))
# How many refunded (all-tools-failed) steps a single run may take before we stop
# refunding. Bounds the loop: iterations <= max_steps + MAX_WASTED_STEPS + 1.
MAX_WASTED_STEPS = int(os.environ.get("RESEARCH_MAX_WASTED_STEPS", "4"))
# Refunding only covers steps where every tool ERRORED. The commoner disappointment is
# a page that loads perfectly and says nothing useful — that spends a step and the
# refund never fires, because from the outside it looks like a successful read. Only
# the model knows the difference, so let it say so: `more_steps` is a tool it can call
# to buy itself more room when the sources it drew were poor. Bad luck with sources
# should not force a thin answer.
# Bounded, because "let it decide" cannot mean "let it run forever": at most
# MAX_EXTENSIONS grants of STEP_EXTENSION steps each.
STEP_EXTENSION = int(os.environ.get("RESEARCH_STEP_EXTENSION", "6"))
MAX_EXTENSIONS = int(os.environ.get("RESEARCH_MAX_EXTENSIONS", "2"))


def steps_for(effort):
    """Research budget for this reasoning effort. Only `high` differs: see above."""
    return MAX_STEPS_HIGH if effort == "high" else MAX_STEPS
# A stock browser UA, deliberately: a custom "research-agent" token would label
# every fetch as a bot to sites and network observers — worse for privacy AND
# more likely to be blocked. Blend in with the browser traffic this machine
# already produces.
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0"

# ---------- how long Ollama keeps a model resident after a turn (keep_alive) ----------
# The global floor for EVERY model lives on the Ollama server, not here — set
# OLLAMA_KEEP_ALIVE (e.g. 5m via a systemd drop-in). Model lifecycle is an ops
# decision. The app carries only the ONE exception a single global value can't
# express: the workhorse we reach for constantly is worth keeping warm (a cold load
# costs ~30s), so for sticky models we override keep_alive to 45min, refreshed by each
# use and renewed while a tab is open (see apply_presence in webapp/server.py); the
# web UI's Deactivate button is the manual off-switch. Non-sticky models send
# no keep_alive field at all, so the server's global default governs them. (A finite
# duration on purpose — never -1, which pins a model in RAM "forever" and once wedged
# one with a year-2318 expiry.)
KEEP_ALIVE_STICKY = os.environ.get("RESEARCH_KEEP_ALIVE_STICKY", "45m")
# Models that stay resident. Comma-separated env override, matched as PREFIXES: the
# entry was once the exact name "gpt-oss:120b-fast", which meant that running on plain
# "gpt-oss:120b" — a different tag of the same workhorse — got no protection at all and
# fell to the 5m global, dropping ~65 GB out from under an open tab. A prefix covers
# every tag of the family, and a full name still matches itself exactly.
STICKY_MODELS = set(filter(None,
    (s.strip() for s in os.environ.get("RESEARCH_STICKY_MODELS", "gpt-oss").split(","))))

def is_sticky(model):
    """True for a model we hold in RAM for as long as someone has a tab open."""
    return any(model.startswith(prefix) for prefix in STICKY_MODELS)

def keep_alive_for(model):
    """The keep_alive to send for this model, or None to defer to the server's
    global OLLAMA_KEEP_ALIVE default."""
    return KEEP_ALIVE_STICKY if is_sticky(model) else None

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

class UnusableSource(Exception):
    """The tool ran without crashing but returned nothing a text model can work with —
    a PDF, a page with no readable body, a search with no hits. Raised rather than
    returned so it joins the same path as a 403 or a timeout: the research loop treats
    every one of these as a step that cost the user nothing and gives the step back."""


# A search used to return ten one-line snippets, and reading any of them cost a whole
# extra step: another full generation to decide which URL, then the fetch. Measured on
# this machine, a 13-step run spent 87% of its 296s generating tokens and only 13% on
# the network — so every round trip we can delete is worth far more than any speed-up
# to the network itself.
#
# So a search now also OPENS the top few results and returns their text. The fetches
# run concurrently: they are different hosts, so three cost about as long as one, and
# this is the one place in the whole loop where real parallelism is available (the GPU
# cannot be parallelised on this box; the network can). The model can usually answer, or
# decide what to read properly, straight from the search observation.
# Sizing these is a genuine trade, and the first attempt got it wrong. At 3 previews of
# 1800 chars, a deep question measured 432s against 296s before — network fell from 39s
# to 7s while GENERATION rose from 257s to 425s. Every preview stays in the transcript
# and is re-read on every later step, and on this machine context costs more than the
# network ever did. The step count did not fall, so the bloat bought nothing.
#
# So: enough to settle a lookup without a second round trip, not enough to be a page
# dump. Two previews at 900 chars is a third of the original bulk. Set
# RESEARCH_SEARCH_PREVIEW=0 to switch previews off entirely.
SEARCH_PREVIEW_N = int(os.environ.get("RESEARCH_SEARCH_PREVIEW", "2"))
SEARCH_PREVIEW_CHARS = int(os.environ.get("RESEARCH_SEARCH_PREVIEW_CHARS", "900"))


def _preview_one(url):
    """Body text of `url`, trimmed to preview length. Never raises — a preview that
    fails is simply absent, and the model still has the snippet and can web_fetch."""
    try:
        page = web_fetch(url)
    except Exception:
        return ""
    body = page.split("\n\n", 1)[-1]        # drop the TITLE/URL header web_fetch adds
    body = body.strip()
    if len(body) > SEARCH_PREVIEW_CHARS:
        body = body[:SEARCH_PREVIEW_CHARS].rstrip() + " …"
    return body


def _attach_previews(items):
    """Fetch several pages at once and hang the text on each item, in place."""
    items = [it for it in items if it.get("url")]
    if not items:
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = {pool.submit(_preview_one, it["url"]): it for it in items}
        for fut in as_completed(futures):
            try:
                futures[fut]["preview"] = fut.result()
            except Exception:
                futures[fut]["preview"] = ""


def web_search(query, max_results=5):
    """Search the web via DuckDuckGo, and open the top few results.

    Returns a list of {title, url, snippet, preview} — `preview` is the start of the
    actual page for the highest-ranked results, empty for the rest (and for pages that
    would not load, which is normal and not an error)."""
    from ddgs import DDGS
    max_results = max(1, min(int(max_results), 10))
    out = []
    for r in DDGS().text(query, max_results=max_results):
        out.append({
            "title": r.get("title", ""),
            "url":   r.get("href") or r.get("url") or r.get("link", ""),
            "snippet": r.get("body") or r.get("snippet", ""),
            "preview": "",
        })
    if SEARCH_PREVIEW_N > 0:
        _attach_previews(out[:SEARCH_PREVIEW_N])
    return out

_FIGURE_WORDS = re.compile(r"chart|plot|graph|figure|fig[-_ ]?\d|bench|score|result|"
                           r"diagram|table|curve|comparison|perf", re.I)

def _image_marker(img, page_url):
    """'[image: alt — url, WxH]' for an image worth mentioning, else None.
    Logos, badges, avatars and icons are noise; an image with alt text, or one whose
    file name says figure/chart/benchmark, is kept. The marker is only a pointer —
    whether the picture matters for the question is the model's call (view_image)."""
    src = img.get("src") or img.get("data-src") or ""
    if not src or src.startswith("data:"):
        return None
    src = urllib.parse.urljoin(page_url, src)
    alt = (img.get("alt") or img.get("title") or "").strip()
    name = urllib.parse.unquote(urllib.parse.urlparse(src).path.rsplit("/", 1)[-1])
    if re.search(r"shields\.io|badge|avatar|logo|icon|emoji|button|sprite|pixel", src + " " + alt, re.I):
        return None
    try:
        w, h = int(img.get("width") or 0), int(img.get("height") or 0)
    except ValueError:
        w = h = 0
    if 0 < max(w, h) < 120:
        return None                       # declared tiny: an icon, not a figure
    if not alt and not _FIGURE_WORDS.search(name):
        return None
    size = f", {w}x{h}" if w and h else ""
    return f"[image: {alt or name} — {src}{size}]"

def web_fetch(url, part=1):
    """Download a page and return its readable text, FETCH_CHARS at a time.
    `part` 2, 3, ... is the next stretch of the same page: the research loop asks for it
    when the model fetches a URL it has already read, so "scroll down" is a real action
    instead of a step that hands back the same first 6000 characters again."""
    import requests
    from bs4 import BeautifulSoup
    # requests' timeout bounds each socket wait, not the whole download: a server that
    # trickles bytes kept one fetch alive for 380s (2026-10-12, SBC run). So read the body
    # ourselves against a wall-clock deadline, and cap its size too.
    deadline = time.time() + 25
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=(10, 10), stream=True)
    resp.raise_for_status()
    body = bytearray()
    for chunk in resp.iter_content(65536):
        body += chunk
        if time.time() > deadline:
            resp.close()
            if len(body) < 2000:
                raise TimeoutError(f"{url} took longer than 25s to send its page")
            break                      # use what arrived; the first screenful is what counts
        if len(body) > FETCH_MAX_BYTES:
            break
    resp._content = bytes(body)        # so .content / .text below read what we got
    ctype = resp.headers.get("content-type", "")
    if "pdf" in ctype.lower() or resp.content[:5] == b"%PDF-":
        # Without this, a PDF arrives as compressed-binary soup: the model burns a
        # step reading garbage, concludes "maybe another PDF will extract better",
        # and burns the next step the same way (watched it happen twice in a row).
        raise UnusableSource(
            f"{url} is a PDF — web_fetch cannot extract PDF text, so fetching it (or "
            "any other PDF link) will not work. Find an HTML page with the same "
            "information instead: a docs page, spec page, abstract page, or article.")
    # Encoding: when a response has no charset in its Content-Type, requests falls back
    # to ISO-8859-1 per the old HTTP spec, and every non-ASCII character arrives mangled
    # — "Ryzen™" reads as "Ryzenâ¢". Hand BeautifulSoup the raw BYTES instead and let it
    # use the page's own <meta charset>, which is what the browser does.
    if "html" not in ctype and "xml" not in ctype and resp.content[:100].strip().startswith(b"{"):
        resp.encoding = resp.encoding or resp.apparent_encoding
        return resp.text[:FETCH_CHARS]  # looks like JSON/plain
    soup = BeautifulSoup(resp.content, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    # get_text() drops images silently, so a page whose key result is a chart read as
    # if the chart weren't there (HF KLD-vs-size plot, 2026-10-13). Leave a marker the
    # model can act on with view_image: alt text, link, and size when the page says.
    for img in soup.find_all("img"):
        marker = _image_marker(img, resp.url or url)
        if marker:
            img.replace_with(f"\n{marker}\n")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n"))
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 80:
        # A JS-only app or consent wall extracts to essentially nothing. The bar is
        # deliberately low: a short page can still be a real source (example.com is
        # 142 chars), and wrongly discarding a good fetch costs more than reading a
        # thin one.
        raise UnusableSource(
            f"{url} returned almost no readable text ({len(text)} chars) — it is likely "
            "JavaScript-rendered or behind a consent wall. Try a different source.")
    total = -(-len(text) // FETCH_CHARS)            # ceil: how many parts the page has
    if part > total:
        raise UnusableSource(
            f"You have already read all of {url} ({total} part(s)) — fetching it again "
            "returns nothing new. Use what it said, or read a different page.")
    body = text[(part - 1) * FETCH_CHARS: part * FETCH_CHARS]
    where = f" (part {part} of {total})" if total > 1 else ""
    if part < total:
        body += (f"\n...[page continues — web_fetch the same URL again to read "
                 f"part {part + 1} of {total}]")
    return f"TITLE: {title}\nURL: {url}{where}\n\n{body}"

VIEW_PROMPT = """You are the eyes of a research assistant that cannot see images. It asks:
{question}

Answer from the picture, briefly (under 200 words):
- First line: what it is (bar chart, line plot, table, diagram, photo) and its title.
- Then the answer, with numbers read off the image. Read values against the
  gridlines and axis labels — note a logarithmic axis — and mark readings
  approximate (~). Name points by their printed labels.
- Then anything in the image that changes the answer: an outlier, a series that
  breaks the trend, a footnote.
Report only what is in the image. If something is too small to read, say so."""


def _image_png_b64(url, max_side=None):
    """Download an image and return (base64 PNG, (w, h)), ready for Ollama's `images`.
    Re-encoded as PNG because Ollama decodes only a few formats (no WebP/AVIF/GIF on
    every build), flattened onto white because a transparent chart becomes black-on-
    black once alpha is dropped, and scaled to IMAGE_MAX_SIDE so a 4000px screenshot
    doesn't spend thousands of context tokens on detail no model reads."""
    import base64, io
    import requests
    from PIL import Image
    deadline = time.time() + 25
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=(10, 10), stream=True)
    resp.raise_for_status()
    data = bytearray()
    for chunk in resp.iter_content(65536):
        data += chunk
        if time.time() > deadline:
            raise TimeoutError(f"{url} took longer than 25s to send the image")
        if len(data) > IMAGE_MAX_BYTES:
            raise UnusableSource(f"{url} is over {IMAGE_MAX_BYTES >> 20} MB — too big to view.")
    ctype = resp.headers.get("content-type", "").lower()
    if "svg" in ctype or bytes(data[:256]).lstrip().startswith(b"<"):
        raise UnusableSource(
            f"{url} is not a picture view_image can open ({ctype or 'markup'}; SVG and HTML "
            "pages are not supported). If it is a page, web_fetch it instead.")
    try:
        im = Image.open(io.BytesIO(bytes(data)))
        im.seek(0)                        # first frame of an animation
        im.load()
    except Exception:
        raise UnusableSource(f"{url} is not an image ({ctype or 'unknown type'}).")
    if max(im.size) < 64:
        raise UnusableSource(f"{url} is a {im.size[0]}x{im.size[1]} icon — nothing to read.")
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    side = max_side or IMAGE_MAX_SIDE
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), im.size


def view_image(url, question, viewer):
    """Have `viewer` (a vision model) read one image; return its account as text.
    A one-shot request in a fresh context: the picture's tokens are spent once, here,
    instead of riding along in the research conversation for every later step, and a
    text-only asker (gpt-oss) gets the same thing a vision asker does. Thinking off
    and a cool temperature: this is transcription, not deliberation, and neither
    setting reloads the runner (see chat_options)."""
    if not url.startswith(("http://", "https://")):
        raise UnusableSource(f"{url!r} is not an image URL — use the link from an [image: ...] marker.")
    b64, (w, h) = _image_png_b64(url)
    msgs = [{"role": "user", "images": [b64], "content": VIEW_PROMPT.format(
        question=question.strip() or "what the image shows")}]
    text = ""
    for kind, chunk in ollama_chat_stream(viewer, msgs, effort=False, temperature=0.3):
        if kind == "content":
            text += chunk
    if not text.strip():
        raise UnusableSource(f"{viewer} returned nothing for {url} — work from the page text.")
    return f"IMAGE: {url} ({w}x{h}), read by {viewer}\n\n{text.strip()}"


# ---------- encyclopedia card ----------
# A "what is X" question has a settled answer that an encyclopedia already states, with a
# picture — the card Firefox shows for a Wikipedia term (kong, 2026-10-13: "at least one
# image to the page"). Looked up by the loop itself before the first turn, never by the
# model: about a second, no research step, and no relying on gpt-oss to call a tool it
# wasn't forced into. Shown only when the article's title is plainly what the question
# names, so a long question that merely mentions "BBC" doesn't get a BBC card.
WIKI_UA = "research-web/1.0 (local research assistant)"
_CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")

def _wiki_lang(text):
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    return "zh" if _CJK.search(text) else "en"

def _words(text):
    """Lower-case word set, with a trailing plural 's' also counted as its singular."""
    ws = set(re.findall(r"[a-z0-9]+", text.lower()))
    return ws | {w[:-1] for w in ws if len(w) > 3 and w.endswith("s")}

def _names_title(title, question, qwords):
    if _CJK.search(title):
        return re.sub(r"\s*[(（].*?[)）]", "", title) in question
    tw = re.findall(r"[a-z0-9]+", re.sub(r"\s*\(.*?\)", "", title).lower())
    if not tw or not set(tw) <= qwords:
        return False
    # A bare acronym ("BBC", "TV") is the subject of a short question, not a long one.
    return not (len(tw) == 1 and len(tw[0]) <= 3 and len(qwords) > 6)

# Question words around the subject: "what is a quill" names the article "quill".
_ASK_WORDS = set("""what whats who whom whose which is are was were be a an the of do does did
tell me about explain define definition meaning means describe please can could would you
your i we my know give brief overview summary on for to in and or any some
something anything little bit things stuff facts quick interesting""".split())
_ZH_ASK = re.compile(r"(什么是|是什么|是谁|是什麼|什麼是|介绍一下|介紹一下|请问|請問|的意思|吗|嗎|呢|[？?。！!，,])")
_CARD_CACHE = {}

def _subject(q):
    """The term a question names: question words and punctuation stripped."""
    if _CJK.search(q):
        return _ZH_ASK.sub("", q).strip()
    ws = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.]*", q)
    core = [w for w in ws if w.lower().strip("'") not in _ASK_WORDS]
    return " ".join(core).rstrip(".")

def _wiki_get(url, params=None):
    """GET from Wikipedia, retrying once on 429: Wikimedia throttles bursts of anonymous
    calls, and a refused lookup silently cost a card (and sent gpt-oss searching for
    "quill multiple meanings", 2026-10-13)."""
    import requests
    for attempt in (0, 1):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": WIKI_UA}, timeout=5)
        except (requests.Timeout, requests.ConnectionError):
            if attempt:
                raise
            continue                      # this link drops requests; one retry is cheap
        if r.status_code != 429 or attempt:
            return r
        time.sleep(1.5)

def _wiki_summary(api, title):
    r = _wiki_get(api + "/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
    return r.json() if r.status_code == 200 else None

def encyclopedia_card(question, report=None):
    """{title, extract, image, url, source, lang} for the article the question names, or None.
    Silent on every failure: a missing card costs nothing, a slow one would cost the
    user's first second, so each request is held to a few seconds. Frugal on purpose:
    Wikimedia answers a burst of anonymous calls with 429, so the direct title lookup
    (one call) comes first, search only after it, and answers are cached."""
    import requests
    q = re.sub(r"https?://\S+", " ", question).strip()
    if not q or q != question.strip() and len(q.split()) < 3:
        return None                       # "summarize <url>" is about the page, not a term
    lang = _wiki_lang(q)
    api = f"https://{lang}.wikipedia.org"
    subj = _subject(q)
    if not subj:
        return None
    key = (lang, subj.lower())
    if key in _CARD_CACHE:
        return _CARD_CACHE[key]
    d = None
    try:
        if len(subj.split()) <= 4:        # short enough to be an article title itself
            d = _wiki_summary(api, subj)
            if d and d.get("type") == "disambiguation":
                _CARD_CACHE[key] = None   # "mercury": several articles — don't guess one
                return None
        if not d or d.get("type") != "standard":
            hits = _wiki_get(api + "/w/api.php", {"action": "query", "list": "search",
                                                  "srsearch": subj[:250], "srlimit": 6,
                                                  "format": "json"}).json()["query"]["search"]
            qwords = _words(q)
            title = next((h["title"] for h in hits if _names_title(h["title"], q, qwords)), None)
            d = _wiki_summary(api, title) if title else None
    except Exception as e:
        if report is not None:            # said in the trace, not swallowed (Ada, 2026-10-13)
            report["why"] = f"Wikipedia lookup failed: {type(e).__name__}"
        return None                       # not cached: a 429 or timeout may pass
    card = None
    if d and d.get("type") == "standard" and d.get("extract"):
        card = {"title": d.get("title", subj), "extract": d["extract"],
                "image": (d.get("thumbnail") or {}).get("source", ""),
                "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
                "source": "Wikipedia", "lang": lang}
    _CARD_CACHE[key] = card
    return card


# ---------- one photo for the card, when there is no encyclopedia card ----------
# The first version showed every image from every page read, and kong's Orange Pi answer
# came back with a column of pictures longer than the answer — a UniFi switch, a person —
# (2026-10-13: "compare which image is the most helpful ... I just only need one brief
# card"). So: at most ONE picture, and only a photo that plainly shows what the question
# is about. Charts, screenshots and logos never qualify (kong elsewhere: a report picture
# is a photo of a real thing; someone else's chart is not one).
_NOT_A_PHOTO = re.compile(r"chart|graph|plot|diagram|table|screenshot|screen ?shot|"
                          r"benchmark|score|result|fastfetch|neofetch|terminal|console|"
                          r"output|logo|icon|banner|badge|avatar|author|profile|headshot|"
                          r"sponsor|advert|\bad\b|thumbnail|placeholder|newsletter", re.I)
_CARD_STOP = _ASK_WORDS | set("""why how when where not very much more most than other others
existing considering compared compare versus vs between best better worse good bad really
still also just only even does doesn't isn't aren't new latest current popular""".split())

def photo_score(alt, src, qwords):
    """Key words an image's caption/file name shares with the question; 0 for non-photos.
    Only a pre-filter: the vision model looks at the survivors and picks (card_stage)."""
    name = urllib.parse.unquote(urllib.parse.urlparse(src).path.rsplit("/", 1)[-1])
    text = f"{alt} {re.sub(r'[_-]+', ' ', name)}"
    if _NOT_A_PHOTO.search(text):
        return 0
    text = re.sub(r"([a-z])([A-Z0-9])", r"\1 \2", text)      # "OrangePi6" -> "Orange Pi 6"
    tw = set(re.findall(r"[a-z0-9]+", text.lower()))
    tw |= {w[:-1] for w in tw if len(w) > 3 and w.endswith("s")}
    # Each question word counts once, matched as written or as its singular.
    shared = sum(1 for w in qwords if w in tw or (w.endswith("s") and w[:-1] in tw))
    return shared

def card_words(question):
    """The question's key words, as written (no singular copies)."""
    return {w for w in re.findall(r"[a-z0-9]+", question.lower())
            if w not in _CARD_STOP and w not in ("its", "it", "this", "that", "they")
            and len(w) > 1}


def arxiv_search(query, max_results=5):
    """Search arXiv. Returns a list of {title, url, authors, summary}."""
    max_results = max(1, min(int(max_results), 10))
    q = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"https://export.arxiv.org/api/query?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    # 60s, not 25: arXiv's API on a slow day takes 30s+ for multi-word relevance
    # queries (measured), and a timeout here costs the model its paper search.
    with urllib.request.urlopen(req, timeout=60) as r:
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
    # Needs to know which model is asking, so research_events runs it itself; this
    # entry is what makes it a known tool name.
    "view_image":   lambda a: (_ for _ in ()).throw(ValueError("view_image runs in the research loop")),
}

# The same three tools as JSON-schema specs for Ollama's NATIVE tool-calling API.
# With these passed to /api/chat, a tool-trained model (gpt-oss, qwen3) emits calls
# through the channel it was trained on and Ollama hands them to us already parsed —
# no scraping JSON out of prose, so no escaping bugs and no "described but didn't
# emit" stalls. Models without tool support fall back to the in-band protocol.
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "plan",
        "description": (
            "Optional: show the user how you read the question and its parts. They see "
            "it as a checklist — it is how they can tell you understood them. Call it "
            "again to mark a part done or not_found. FREE: it does not use a research "
            "step. Skip it for a question you answer straight away."),
        "parameters": {"type": "object", "properties": {
            "reading": {"type": "string", "description": (
                "the question in one clean line, as you understood it. Fix garbled or "
                "misheard words here (dictation: 'view con' = 'view count').")},
            "parts": {"type": "array", "description": "1-6 parts, split by where each answer lives",
                      "items": {"type": "object", "properties": {
                "part": {"type": "string", "description": "what this part answers, a few words"},
                "how": {"type": "string", "enum": ["know", "search", "ask"], "description": (
                    "know = you can answer it yourself (word meanings, phrasing, settled "
                    "facts); search = needs world data AND the question says what to look "
                    "up; ask = the question leaves its target open (which programme? which "
                    "year?) — ask in your answer instead of searching for random examples")},
                "status": {"type": "string", "enum": ["todo", "done", "not_found"]},
            }, "required": ["part", "how"]}},
        }, "required": ["reading", "parts"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live web (DuckDuckGo). Returns titles, URLs and snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer", "description": "how many results, 1-10 (default 5)"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": "Download one web page and return its readable text.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "full http(s) URL of the page to read"},
        }, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "view_image",
        "description": (
            "Look at one image from a page you fetched — the [image: ...] markers in "
            "web_fetch text — and get back what it shows: its text, axes, legend, and "
            "the data read off it as numbers. Use it when the answer is IN a figure "
            "(benchmark chart, plot, diagram, table screenshot), not for logos or photos."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "the image URL from the [image: ...] marker"},
            "question": {"type": "string", "description": (
                "what you need from it, e.g. 'KLD for each Unsloth quant at its size'")},
        }, "required": ["url", "question"]},
    }},
    {"type": "function", "function": {
        "name": "arxiv_search",
        "description": "Search arXiv for papers. Returns titles, URLs, authors and abstracts.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer", "description": "how many results, 1-10 (default 5)"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "more_steps",
        "description": (
            "Ask for more research steps when the ones you have are not enough to "
            "answer properly — typically because the pages you fetched were poor, "
            "off-topic or contradicted each other. Use it when you know what to try "
            "next and are only short of room. Do NOT use it to keep looking for "
            "confirmation of something you have already established."),
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string",
                       "description": "what went wrong and what you intend to try next"},
        }, "required": ["reason"]},
    }},
    {"type": "function", "function": {
        "name": "ask_user",
        "description": (
            "Ask the user which thing they mean, when they point at one specific thing "
            "('the X show', 'that film') and several real ones fit about equally. Give "
            "each option its short answer when you have it. This ENDS your turn. A plain "
            "topic or famous name is not ambiguous: answer its main meaning instead."),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string",
                         "description": "one short question, e.g. 'Which one did you mean?'"},
            "options": {"type": "array", "items": {"type": "string"},
                        "description": (
                            "2-5 options, each ONE plain string. Describe each by what the user would RECOGNISE "
                            "(format, who was in it, setting, years), not only by its name "
                            "— people often ask precisely because they forgot the name. "
                            "Add the short answer for that option if you already have it.")},
        }, "required": ["question", "options"]},
    }},
]

_CAPS = {}  # model name -> its Ollama capabilities, probed once per process

def model_caps(model):
    """What Ollama says this model can do ("tools", "vision", "thinking", ...).
    Conservative: any failure (old Ollama without `capabilities`, unknown model,
    Ollama down) means none, so tools fall back to the in-band JSON protocol and
    images go to the helper. A failure is not cached, so a model pulled later is seen."""
    if model not in _CAPS:
        try:
            body = json.dumps({"model": model, "name": model}).encode()  # old Ollama wants "name"
            req = urllib.request.Request(OLLAMA_HOST + "/api/show", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                _CAPS[model] = set(json.load(r).get("capabilities") or [])
        except Exception:
            return set()
    return _CAPS[model]

def model_supports_tools(model):
    return "tools" in model_caps(model)

def model_has_vision(model):
    return "vision" in model_caps(model)

# ========================================================================
# SYSTEM PROMPT
# ========================================================================

def date_context():
    """The date-awareness block stitched into the system prompt at request time.

    Local models are anchored to their training window (2023–2024) and will read
    "current", "latest", or "in recent years" as relative to THAT unless told
    otherwise — worse, some treat a post-cutoff date as a typo and quietly ignore
    it. So this block is rebuilt from the machine's clock on every request (see
    system_prompt), states the date is authoritative, and pins all relative time
    language to it."""
    today = datetime.date.today()
    return (
        "TODAY'S DATE: {}.\n"
        "This comes from the machine's clock and is authoritative. It is later "
        "than your training data — that is not an error; your knowledge is "
        "simply out of date, and events between your training cutoff and today "
        "are unknown to you. Therefore:\n"
        "- Interpret ALL relative time language (\"now\", \"current\", "
        "\"latest\", \"recent\", \"in recent years\", \"this year\") against "
        "today's date above, never against your training data.\n"
        "- Never state or imply that the present year is the year your training "
        "data ends. Do not say \"as of my knowledge\" — say what is true as of "
        "today, searching the web if needed.\n"
        "- For anything that may have changed since your training cutoff, "
        "search rather than answer from memory."
    ).format(today.strftime("%A, %B %d, %Y").replace(" 0", " "))


_FIGURE_RULE = """   Fetched text marks pictures as [image: alt — url]. Text alone misses what a
   figure shows: when the question's answer, or a page summary's key result,
   sits in a chart, plot, benchmark figure or diagram, call view_image on it
   rather than guessing from the caption — and say in the answer what the
   figure shows. Skip logos, photos and decoration.
"""
_FIGURE_RULE_BLIND = """   Fetched text marks pictures as [image: alt — url]. You cannot see them: when
   a chart or figure holds a result the question needs, say so in the answer
   and link it, rather than guessing what it shows from the caption.
"""

def system_prompt(native, vision=True):
    """Final system prompt: fresh date block + the protocol-appropriate prompt. A model
    without vision is not offered view_image (it would load a second model mid-turn —
    kong, 2026-10-13: don't load the helper until the main model has finished)."""
    text = SYSTEM_NATIVE if native else SYSTEM
    if not vision:
        text = (text.replace(_FIGURE_RULE, _FIGURE_RULE_BLIND)
                    .replace("arxiv_search, view_image)", "arxiv_search)")
                    .replace('- view_image    -> {"tool":"view_image","args":{"url":"https://...png","question":"what you need from it"}}\n', ""))
    return date_context() + "\n\n" + text


# The system-prompt date block alone loses to habit under a real task: the model
# reads "recent years", pattern-completes a training-era year into its reasoning
# ("as of today (July 2024)"), then searches with that stale year and never even
# sees current pages. The last message gets the strongest attention, so a short
# stamp on the question itself — right next to the words that trigger the habit —
# is where the date actually lands. Deliberately minimal (one bracketed line, no
# lecture) to keep it from leaking into answers.
DATE_STAMP_PREFIX = "\n\n[context: today is "


def date_stamp():
    today = datetime.date.today()
    return (DATE_STAMP_PREFIX + today.strftime("%A, %B %d, %Y").replace(" 0", " ")
            + ' — "recent"/"latest"/"current" mean relative to this date, and '
            "search queries about the present should use the current year]")


# The ask-back rule lost to habit the same way the date did. With it only in the
# system prompt, 0 of 10 answers to a vague question ("the Jeremy Clarkson show",
# "the Steve Jobs movie") asked which one the user meant: the model never thinks it
# "cannot tell" — it picks the popular reading and feels sure. So the reminder rides
# on the question too, worded on what the words COULD mean rather than on whether
# the model feels unsure.
# Worded too broadly it overcorrected: "Tell me something about Prometheus" got six
# searches and a five-way menu instead of the myth. So it now asks only when the
# user points at ONE specific thing they can't name, and a plain topic gets answered.
ASK_BACK_STAMP = ("\n[if the user points at one specific thing ('the X show', 'that "
                  "film') and several real ones fit, call ask_user; a plain topic or "
                  "famous name gets its main meaning answered, with at most one line "
                  "naming other meanings]")


def stamp_current_question(messages):
    """Append today's date and the ask-back reminder to the question being answered.
    Only the live turn carries a stamp: it is applied to the turn's own working copy,
    and the session keeps the clean question, so stamps never pile up in history.
    Idempotent in case of a double call."""
    if not messages or messages[-1].get("role") != "user":
        return
    if DATE_STAMP_PREFIX in messages[-1]["content"]:
        return
    messages[-1]["content"] += date_stamp() + ASK_BACK_STAMP


DEEP_BRIEF_PREFIX = "\n\n[DEEP RESEARCH MODE"


def stamp_deep_brief(messages, min_sources):
    """Tell the model up front that this run is two ordered stages: COLLECT, then
    conclude. The depth gate already enforces this after the fact (nudging when it
    tries to finalize too shallow), but a correction arriving mid-run makes long
    runs wander — the model half-commits to an answer, gets pushed back out, and
    re-plans. Stating the shape before step 1 lets it budget the whole run: gather
    from several angles first, judge once, land once. Same idempotent stamping
    pattern as the date."""
    if not messages or messages[-1].get("role") != "user":
        return
    if DEEP_BRIEF_PREFIX in messages[-1]["content"]:
        return
    messages[-1]["content"] += (
        f"{DEEP_BRIEF_PREFIX}: before your FIRST tool call, sketch a plan in your "
        f"reasoning — what must be looked up vs what you will derive yourself, and "
        f"roughly how many searches/fetches each part needs. Your reasoning persists "
        f"between steps: check progress against that plan as results arrive, and "
        f"revise it rather than wander. Then work in two stages, in order. "
        f"STAGE 1 — COLLECT: search this from several different angles and "
        f"web_fetch at least {min_sources} DISTINCT websites; read them before "
        f"judging anything, and do not draw conclusions yet. "
        f"STAGE 2 — CONCLUDE: only after collecting, weigh the evidence and write "
        f"the report with citations. If the question also needs derivation "
        f"(budgets, feasibility), the numbers you fetched are inputs to your own "
        f"analysis — the analysis itself is yours.]")


SYSTEM = """You are a research assistant running locally on the user's machine.
You can reach the LIVE internet through tools. Your knowledge of the WORLD is
frozen and may be outdated — for facts out there (current events, versions,
products, prices, other people's work) use tools rather than guess. Your
REASONING is not frozen: problems that can be solved by thinking are yours to
solve, and no tool can do it for you.

TOOLS (call one at a time):
- web_search    -> {"tool":"web_search","args":{"query":"...","max_results":5}}
- web_fetch     -> {"tool":"web_fetch","args":{"url":"https://..."}}
- arxiv_search  -> {"tool":"arxiv_search","args":{"query":"...","max_results":5}}
- view_image    -> {"tool":"view_image","args":{"url":"https://...png","question":"what you need from it"}}
- more_steps    -> {"tool":"more_steps","args":{"reason":"what went wrong, what you'll try next"}}
- ask_user      -> {"tool":"ask_user","args":{"question":"Which one did you mean?","options":["...","..."]}}
- plan          -> {"tool":"plan","args":{"reading":"the question, one clean line","parts":[{"part":"...","how":"know|search|ask","status":"todo|done|not_found"}]}}

HOW TO ACT — first decide where the answer LIVES, then match your effort:
1. FOLLOW-UPS — the conversation above is yours. When the question refers to
   something already in it (a table, a list, an answer you wrote), that content
   IS the source: reuse it exactly as written, keep the user's wording, and do
   NOT search to re-derive it or to find outside backing for it. Reach for
   tools only for what the conversation does not already contain.
   A short reply right after your answer ("TV show", "no, the other one",
   "the 90s one") usually narrows or corrects the question you just
   answered: answer THAT question again with the new detail — do not treat
   it as a fresh topic.
2. Casual, conversational, or simple questions you already know cold: just
   answer directly. No tools, no multi-step analysis, no ceremony.
3. DERIVABLE problems — logic puzzles, math, proofs, code tracing, anything
   self-contained in the question: the answer lives in the problem, not on the
   web. Solve it in your own reasoning and present the derivation as the
   answer. Do NOT search for confirmation, similar problems, or something to
   cite: your proof IS the source, and a derived answer gets no "Sources:"
   section. Only reach for tools if the problem itself references outside
   facts you don't have.
4. HYBRID problems — feasibility studies, designs, sizing/budget questions
   that mix derivation with real-world data: do the derivation YOURSELF first
   (set up the budgets and inequalities from the stated constraints), then use
   tools only to fetch the specific numbers your analysis needs (a motor's
   mass, a part's torque, a price). Fetched numbers are INPUTS to your math;
   a product's marketing claim ("lifts 8 kg") is never a substitute for
   checking the constraints yourself. If a fetched number contradicts your
   physics, distrust it and say so in the answer.
5. WORLD questions needing current or verifiable facts: use tools. To call one,
   reply with ONLY a single JSON object exactly like the forms above, and
   nothing else on that turn. No prose around it. IMPORTANT: writing ABOUT a
   tool call ("I need to search for X", "Use web_search") does NOT call it —
   only the raw JSON object does. If you decide to search, your entire reply
   must BE that JSON.
6. You will then receive an OBSERVATION with the results. Read it.
7. web_search already OPENS the top few results: those come back with "page text
   (already fetched)" under them, which is the real page, not a snippet. Read
   that text and answer from it. Do NOT web_fetch a URL whose page text you were
   already given — that costs a whole extra step and returns what you have. Use
   web_fetch for the OTHER results (the ones with only a snippet), or to read a
   previewed page in full when the preview cuts off mid-point.
   For a quick lookup, one good source is enough.
   Fetched text marks pictures as [image: alt — url]. Text alone misses what a
   figure shows: when the question's answer, or a page summary's key result,
   sits in a chart, plot, benchmark figure or diagram, call view_image on it
   rather than guessing from the caption — and say in the answer what the
   figure shows. Skip logos, photos and decoration.
8. Save the thorough treatment (several searches from different angles,
   multiple independent sources, cross-checking) for when the user explicitly
   asks for deep research or the stakes clearly demand it.
9. Write the FINAL ANSWER as normal prose (NOT JSON). If you used web sources,
   cite them inline as [1], [2], ... and list the full URLs under a "Sources:"
   heading; note any disagreements between sources. Plain text and markdown
   only: write math in plain symbols (169 GB ≈ 47 × 3.58 GB), NEVER LaTeX
   markup (\\[ \\], \\text, \\frac), and never 【 】 citation tokens.

RULES:
- Multi-step research needs a budget: before the first tool call, note in
  your reasoning what to find vs what to derive and about how many steps
  it deserves; each observation tells you which step you are on — check
  yourself against your plan and converge before the budget runs out.
- Never invent URLs, facts, numbers, or citations. Only cite pages you actually
  fetched or that appeared in search results.
- Prefer primary sources (papers, official docs, repos) over blog summaries.
- Use as many tool calls as the question deserves and no more. When you can
  answer with reasonable confidence, stop and answer.
- PLAN (optional, free — it costs no step): for a question with several parts,
  call plan so the user can see how you read it. `reading` = the question in
  one clean line; dictated questions arrive with misheard sound-alikes ("view
  con" = "view count", "nameberg" = "number"), so fix them there. `parts` = the
  question split by where each answer LIVES: know (what a word means, how to
  phrase something, settled facts — answer these yourself, never search them),
  search (world data, and only when the question says WHAT to look up), ask (it
  leaves its target open: which programme? which year? — do not search for
  random examples; answer the rest and end with one short question). Call plan
  again when a part is done or not found. A question you answer straight away
  needs no plan.
- Your answers to the 'know' parts stand: searching adds to them, it never
  replaces them.
- Two searches for the same thing that both come back without it are a
  finding, not a cue for a third rephrasing: it may not be published in that
  form. Mark the part not_found, say so plainly in the answer, and give the
  nearest thing that IS published — never fill the gap from memory.
- Most words have a main meaning: a plain topic or a famous name ("tell me
  about Prometheus", "what is Python") gets that meaning answered straight
  away. Do not search the other meanings; at most add one closing line naming
  them ("If you meant the software or the film, say so").
- Ask only when the user points at ONE specific thing they cannot name ("the
  Jeremy Clarkson show", "that Steve Jobs movie") and several real ones fit
  about equally. Then call ask_user: describe each option by what the user
  would recognise — the format, who was in it, the setting, the years — and
  give each its short answer if you have it. Never claim to know what they
  meant ("the show you're thinking of") when you are guessing.
- Running out of steps because your sources let you down is not a reason to give
  a thin answer. If the pages you fetched were paywalled, off-topic, or simply
  wrong, and you know what to try instead, call more_steps and say what you will
  try. That is what it is for. It is NOT for hunting more confirmation of
  something you have already established, or for putting off writing the answer.
"""

# For models driven through Ollama's native tool-calling API. The tool definitions
# travel in the API request, so this prompt carries none of the in-band JSON
# protocol — asking for JSON-in-prose while also passing native tools would pull
# the model in two directions at once.
SYSTEM_NATIVE = """You are a research assistant running locally on the user's machine.
You can reach the LIVE internet through the tools provided (web_search, web_fetch,
arxiv_search, view_image). Your knowledge of the WORLD is frozen and may be outdated — for
facts out there (current events, versions, products, prices, other people's
work) use the tools rather than guess. Your REASONING is not frozen: problems
that can be solved by thinking are yours to solve, and no tool can do it for you.

HOW TO ACT — first decide where the answer LIVES, then match your effort:
1. FOLLOW-UPS — the conversation above is yours. When the question refers to
   something already in it (a table, a list, an answer you wrote), that content
   IS the source: reuse it exactly as written, keep the user's wording, and do
   NOT search to re-derive it or to find outside backing for it. Reach for
   tools only for what the conversation does not already contain.
   A short reply right after your answer ("TV show", "no, the other one",
   "the 90s one") usually narrows or corrects the question you just
   answered: answer THAT question again with the new detail — do not treat
   it as a fresh topic.
2. Casual, conversational, or simple questions you already know cold: just
   answer directly. No tools, no multi-step analysis, no ceremony.
3. DERIVABLE problems — logic puzzles, math, proofs, code tracing, anything
   self-contained in the question: the answer lives in the problem, not on the
   web. Solve it in your own reasoning and present the derivation as the
   answer. Do NOT search for confirmation, similar problems, or something to
   cite: your proof IS the source, and a derived answer gets no "Sources:"
   section. Only reach for tools if the problem itself references outside
   facts you don't have.
4. HYBRID problems — feasibility studies, designs, sizing/budget questions
   that mix derivation with real-world data: do the derivation YOURSELF first
   (set up the budgets and inequalities from the stated constraints), then use
   tools only to fetch the specific numbers your analysis needs (a motor's
   mass, a part's torque, a price). Fetched numbers are INPUTS to your math;
   a product's marketing claim ("lifts 8 kg") is never a substitute for
   checking the constraints yourself. If a fetched number contradicts your
   physics, distrust it and say so in the answer.
5. WORLD questions needing current or verifiable facts: call the tools.
   Deciding to search is not searching — actually invoke the tool, don't
   narrate the plan.
6. Read each tool result before deciding your next step.
7. web_search already OPENS the top few results: those come back with "page text
   (already fetched)" under them, which is the real page, not a snippet. Read
   that text and answer from it. Do NOT web_fetch a URL whose page text you were
   already given — that costs a whole extra step and returns what you have. Use
   web_fetch for the OTHER results (the ones with only a snippet), or to read a
   previewed page in full when the preview cuts off mid-point.
   For a quick lookup, one good source is enough.
   Fetched text marks pictures as [image: alt — url]. Text alone misses what a
   figure shows: when the question's answer, or a page summary's key result,
   sits in a chart, plot, benchmark figure or diagram, call view_image on it
   rather than guessing from the caption — and say in the answer what the
   figure shows. Skip logos, photos and decoration.
8. Save the thorough treatment (several searches from different angles,
   multiple independent sources, cross-checking) for when the user explicitly
   asks for deep research or the stakes clearly demand it.
9. Write the FINAL ANSWER as normal prose. If you used web sources, cite them
   inline as [1], [2], ... and list the full URLs under a "Sources:" heading;
   note any disagreements between sources. Plain text and markdown only: write
   math in plain symbols (169 GB ≈ 47 × 3.58 GB), NEVER LaTeX markup
   (\\[ \\], \\text, \\frac), and never 【 】 citation tokens.

RULES:
- Multi-step research needs a budget: before the first tool call, note in
  your reasoning what to find vs what to derive and about how many steps
  it deserves; each observation tells you which step you are on — check
  yourself against your plan and converge before the budget runs out.
- Never invent URLs, facts, numbers, or citations. Only cite pages you actually
  fetched or that appeared in search results.
- Prefer primary sources (papers, official docs, repos) over blog summaries.
- Use as many tool calls as the question deserves and no more. When you can
  answer with reasonable confidence, stop and answer.
- PLAN (optional, free — it costs no step): for a question with several parts,
  call plan so the user can see how you read it. `reading` = the question in
  one clean line; dictated questions arrive with misheard sound-alikes ("view
  con" = "view count", "nameberg" = "number"), so fix them there. `parts` = the
  question split by where each answer LIVES: know (what a word means, how to
  phrase something, settled facts — answer these yourself, never search them),
  search (world data, and only when the question says WHAT to look up), ask (it
  leaves its target open: which programme? which year? — do not search for
  random examples; answer the rest and end with one short question). Call plan
  again when a part is done or not found. A question you answer straight away
  needs no plan.
- Your answers to the 'know' parts stand: searching adds to them, it never
  replaces them.
- Two searches for the same thing that both come back without it are a
  finding, not a cue for a third rephrasing: it may not be published in that
  form. Mark the part not_found, say so plainly in the answer, and give the
  nearest thing that IS published — never fill the gap from memory.
- Most words have a main meaning: a plain topic or a famous name ("tell me
  about Prometheus", "what is Python") gets that meaning answered straight
  away. Do not search the other meanings; at most add one closing line naming
  them ("If you meant the software or the film, say so").
- Ask only when the user points at ONE specific thing they cannot name ("the
  Jeremy Clarkson show", "that Steve Jobs movie") and several real ones fit
  about equally. Then call ask_user: describe each option by what the user
  would recognise — the format, who was in it, the setting, the years — and
  give each its short answer if you have it. Never claim to know what they
  meant ("the show you're thinking of") when you are guessing.
- Running out of steps because your sources let you down is not a reason to give
  a thin answer. If the pages you fetched were paywalled, off-topic, or simply
  wrong, and you know what to try instead, call more_steps and say what you will
  try. That is what it is for. It is NOT for hunting more confirmation of
  something you have already established, or for putting off writing the answer.
"""


# ========================================================================
# OLLAMA (streaming, via stdlib — no extra dependency)
# ========================================================================

def chat_options(temperature=None):
    """Runner options sent with EVERY request — and the single source of truth for them.

    Ollama keys a loaded model on its options, so a request carrying different ones does
    not reuse the resident copy: it evicts it and loads a fresh runner (~30s on a 120B).
    Anything that talks to Ollama about a model we may already have loaded (re-arming
    keep_alive, unloading) must send exactly these, or what should be a free timer touch
    silently costs a full reload.

    temperature 1.0 / top_p 1.0 is OpenAI's recommended sampling for gpt-oss (top_p set
    explicitly because Ollama's default is 0.9). A reasoning model does its converging in
    the analysis channel, not via sampling clamps; the 0.4 used previously produced the
    classic too-cold signature — near-verbatim repeated thinking across steps and
    re-searching/re-fetching the same pages in a loop.

    `temperature` overrides that default for one request. Unlike num_ctx it is a
    per-request SAMPLING knob, not a runner setting: measured against a resident
    120B, load_duration stayed at 0.2s across 0.2/1.0/1.8, i.e. changing it does not
    evict or reload the model. That is why the UI can expose it as a live dial."""
    return {"temperature": TEMPERATURE if temperature is None else temperature,
            "top_p": 1.0, "num_ctx": NUM_CTX}


def ollama_chat_stream(model, messages, tools=None, effort=None, temperature=None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": chat_options(temperature),
    }
    ka = keep_alive_for(model)
    if ka is not None:          # else defer to the server's global OLLAMA_KEEP_ALIVE
        payload["keep_alive"] = ka
    if tools:
        payload["tools"] = tools
    if effort is not None:      # else the model's template default (medium); False = off
        payload["think"] = effort
    req = urllib.request.Request(
        OLLAMA_HOST + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # timeout is per socket read, not total generation time — a long answer keeps
    # streaming fine; only a wedged Ollama (no bytes at all for 5min) trips it.
    # Without this, a hung Ollama leaves the server thread blocked forever.
    with urllib.request.urlopen(req, timeout=300) as resp:
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
            # Native tool calls arrive already parsed — arguments is a dict, not
            # text to scrape. This only happens when `tools` was passed.
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                yield ("tool_call", (fn.get("name", ""), fn.get("arguments") or {}))
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
                block = f"[{i}] {item['title']}\n    {item['url']}\n    {item['snippet']}"
                # A previewed result carries the real page, so say so plainly —
                # otherwise the model treats it as another snippet and fetches the
                # page it has already been handed, which is the round trip we just
                # spent a concurrent fetch to avoid.
                if item.get("preview"):
                    block += ("\n    --- page text (already fetched, no need to "
                              f"web_fetch this one) ---\n{item['preview']}")
                lines.append(block)
        return "\n".join(lines) if lines else "(no results)"
    return str(result)

# ---------- final-answer cleanup ----------
# Some models decorate answers with notation a plain-markdown UI can't show:
# gpt-oss leaks its internal browsing-citation tokens (【2†L1-L4】), and models
# write formulas in LaTeX (\[ \text{Size} \approx 169\;\text{GB} \]) which
# renders as raw backslash soup. Rendering real TeX would mean shipping KaTeX —
# a heavy dependency for the occasional formula — so instead translate the
# common constructs into readable plain text and drop the rest. The system
# prompts also tell the model not to emit these; this pass is the guarantee.
_TIDY_RULES = [
    (re.compile(r"【[^】\n]{0,60}】"), ""),                 # 【2†L1-L4】 citation tokens
    (re.compile(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}"), r"\1/\2"),
    (re.compile(r"\\times\b"), "×"),
    (re.compile(r"\\approx\b"), "≈"),
    (re.compile(r"\\pm\b"), "±"),
    (re.compile(r"\\cdot\b"), "·"),
    (re.compile(r"\\(?:le|leq)\b"), "≤"),
    (re.compile(r"\\(?:ge|geq)\b"), "≥"),
    (re.compile(r"\\(?:ne|neq)\b"), "≠"),
    # logic symbols — proofs (which the model now writes itself) lean on these
    (re.compile(r"\\(?:lnot|neg)\b\s*"), "¬"),
    (re.compile(r"\\(?:land|wedge)\b"), "∧"),
    (re.compile(r"\\(?:lor|vee)\b"), "∨"),
    (re.compile(r"\\(?:Rightarrow|implies)\b"), "⇒"),
    (re.compile(r"\\(?:Leftrightarrow|iff)\b"), "⇔"),
    (re.compile(r"\\(?:to|rightarrow)\b"), "→"),
    (re.compile(r"\\(?:begin|end)\{[a-z*]+\}"), ""),       # \begin{aligned} ... \end{aligned}
    # aligned "&=" keeps its equals; bare & is left alone (R&D, AT&T are prose)
    (re.compile(r"[ \t]*&=[ \t]*"), " = "),
    (re.compile(r"\\\\(?=\s*$)", re.M), ""),               # trailing \\ line breaks
    (re.compile(r"\\(?:;|,|:|!)"), " "),                   # LaTeX spacing commands
    (re.compile(r"\\(?:quad|qquad)\b"), "  "),
    (re.compile(r"\\[\[\]()]"), ""),                       # \[ \] \( \) math fences
    (re.compile(r"\$\$"), ""),                             # $$ display fences ($ alone is money)
]

def tidy_answer(text):
    """Make a final answer displayable: strip citation tokens, de-TeX math."""
    for pat, rep in _TIDY_RULES:
        text = pat.sub(rep, text)
    return re.sub(r"\n{3,}", "\n\n", text)  # collapse holes left by removed fences

# ========================================================================
# AGENT LOOP
# ========================================================================

def _stream_turn(model, messages, tools=None, effort=None, temperature=None):
    """Stream one model completion, yielding {"type":"answer_chunk"|"thinking"} as
    tokens arrive, then a final {"_turn": (content, thinking, native_calls)}
    sentinel, where native_calls is a list of (tool, args) that arrived through
    Ollama's tool-call channel (empty unless `tools` was passed and used).
    Reasoning models split output across a content and a thinking channel and may
    hide the answer or the tool call in either, so we surface both and let the
    caller decide. This is the one streaming primitive behind both the CLI and web."""
    content, thinking, native_calls = "", "", []
    for kind, chunk in ollama_chat_stream(model, messages, tools, effort, temperature):
        if kind == "content":
            content += chunk
            yield {"type": "answer_chunk", "text": chunk}
        elif kind == "thinking":
            thinking += chunk
            yield {"type": "thinking", "text": chunk}
        else:
            native_calls.append(chunk)  # already-parsed (tool, args)
    if native_calls:
        # Echo the turn back in native form so the model's chat template renders
        # the calls the way it was trained to see them — reasoning included.
        #
        # The `thinking` is not decoration here. gpt-oss plans its research inside
        # the analysis channel ("search X, then open the spec page, then compare"),
        # and harmony requires that reasoning be fed back while a tool loop is still
        # running, or the model re-reads its own history with the plan amputated and
        # re-derives it every step. Ollama's template renders it back into the
        # analysis channel only for messages *after the last user message*, which is
        # exactly harmony's rule — reasoning survives the in-flight turn and is
        # dropped once the turn closes. commit_turn stores only the clean Q+A, so
        # nothing leaks into the next turn.
        messages.append({"role": "assistant", "content": content,
                         "thinking": thinking,
                         "tool_calls": [{"function": {"name": n, "arguments": a}}
                                        for n, a in native_calls]})
    else:
        # Keep the answer / in-band tool-call JSON in history; fall back to the
        # reasoning channel when content is empty so a hidden call isn't lost.
        messages.append({"role": "assistant", "content": content or thinking})
    yield {"_turn": (content, thinking, native_calls)}

SYNTH_FROM_EVIDENCE = (
    "Enough research — do NOT call any more tools. Using ALL the evidence you "
    "gathered above, write the FINAL ANSWER now as prose, with inline [n] "
    "citations and a 'Sources:' list of the URLs you actually used. If some "
    "details stayed uncertain, say so.")
SYNTH_FROM_KNOWLEDGE = (
    "You did not run any web search. Do NOT try to use tools now and do NOT say "
    "you need to search. Answer the question directly from your own knowledge, as "
    "prose. If you are not fully certain, say so in one short line at the end.")

def _option_text(opt):
    """One ask_user option as display text. The schema asks for strings, but gpt-oss
    often sends objects instead — {"label": ..., "description": ..., "short_answer": ...}
    — which used to reach the page as a Python dict repr. Label first, then the rest."""
    if isinstance(opt, dict):
        keys = ("label", "title", "name", "option")
        label = next((str(opt[k]).strip() for k in keys if opt.get(k)), "")
        rest = [str(v).strip() for k, v in opt.items() if v and k not in keys]
        opt = " — ".join(x for x in [label] + rest if x)
    return str(opt).strip(" -•\t")


SYNTH_RETRY = (
    "You wrote no answer — only private reasoning, which the user cannot see. No "
    "tools exist any more. Write the FINAL ANSWER now, as prose, from what you have.")
SYNTH_GAVE_UP = (
    "I ran out of research steps before I could write an answer. Try asking again "
    "— medium effort usually lands it — or narrow the question a little.")


def _synthesis(model, messages, instruction=SYNTH_FROM_EVIDENCE, effort=None, temperature=None):
    """Force a written answer: stream it as answer chunks, then end with a `final`
    event carrying the whole text. No `tools` are passed here, so in native mode
    the model cannot call anything even if it wants to — prose is the only exit.

    It can still leave the content channel EMPTY: at the end of a long high-effort
    run gpt-oss keeps planning its next search in its reasoning ("Let's find the Top
    Gear page."), finds no tool to call, and stops. This used to fall back to showing
    that reasoning as the answer — the user got a page of the model talking to itself,
    quoting the question stamp back as "a note the user included". Now: one retry at
    low effort (less room to plan, more pressure to write), then an honest one-liner.
    Raw reasoning is never the answer."""
    messages.append({"role": "user", "content": instruction})
    content = ""
    for attempt in range(2):
        for ev in _stream_turn(model, messages, effort=effort if attempt == 0 else "low",
                               temperature=temperature):
            if "_turn" in ev:
                content = ev["_turn"][0]
            else:
                yield ev
        if content.strip():
            break
        yield {"type": "notice", "text": "no answer written — asking it to write one"}
        messages.append({"role": "user", "content": SYNTH_RETRY})
    yield {"type": "final", "text": tidy_answer(content.strip()) or SYNTH_GAVE_UP}


def effort_for(min_sources):
    """Map the depth the user asked for onto gpt-oss's reasoning effort.

    Reasoning effort is this model's main dial and costs nothing to set: the same
    weights answer at three depths of deliberation. OpenAI's own numbers show the
    agentic benchmarks gain the most from it (Tau-Bench Retail 49->68, SWE-Bench
    48->62 going low->high), which is exactly the multi-step tool work this loop
    does — while `low` makes a quick lookup noticeably snappier. So the depth
    selector now governs how hard it thinks as well as how much it must read.
    Returns None for Auto, leaving the template's own default (medium)."""
    if min_sources <= 0:
        return None                       # Auto — the model matches effort to the question
    return "low" if min_sources < 3 else "high"   # Quick (1 source) / Deep (5 sources)


# ---------- the plan: how the model read the question ----------
# A checklist the user can read: the question as understood (dictation fixed) and its
# parts, each marked know/search/ask and ticked found/not found. It used to carry a step
# estimate per part, driving a progress bar, an ETA and "you are past your own estimate"
# nudges — and the first research call was held until a plan arrived. Dropped 2026-10-13:
# a model guessing its step count before reading any page is noise (planned ~1, used 5;
# planned ~2, used 1), and the forced plan cost a turn per question, two when gpt-oss
# sent it malformed. What stays is optional and free.
PLAN_FREE_CALLS = int(os.environ.get("RESEARCH_PLAN_FREE_CALLS", "4"))
PLAN_HOWS = ("know", "search", "ask")
PLAN_STATUSES = ("todo", "done", "not_found")


def _plan_parts(raw):
    """The model's parts list, normalised. gpt-oss sends dicts, strings or a mix (it did
    the same with ask_user options), and an in-band call may flatten the list to text."""
    if isinstance(raw, str):
        try:                                  # a JSON list passed as a string
            raw = json.loads(raw)
        except ValueError:
            raw = [ln.strip(" -•\t") for ln in re.split(r"[\n;]", raw)]
    if isinstance(raw, dict):                 # {"1": {...}, "2": {...}} or one part
        raw = list(raw.values()) if all(isinstance(v, (dict, str)) for v in raw.values()) \
            and "part" not in raw else [raw]
    parts = []
    flat = []
    for it in (raw if isinstance(raw, list) else []):
        # gpt-oss also packs several parts into one string: "know: X; search: Y"
        flat.extend(p.strip(" -•\t") for p in re.split(r"[\n;]", it)) \
            if isinstance(it, str) else flat.append(it)
    for it in [x for x in flat if x][:6]:
        if isinstance(it, str):
            m = re.match(r"(know|search|ask)\s*[:\-–]\s*(.+)", it, re.I)
            it = {"part": m.group(2), "how": m.group(1)} if m else {"part": it}
        if not isinstance(it, dict):
            continue
        # gpt-oss ignores the schema's field names for its own: asked for part/how it
        # sent {"description", "type", ...} — so read the synonyms too,
        # and {"search": "..."} as how + part.
        pick = lambda *keys: next((it[k] for k in keys if it.get(k) not in (None, "")), "")
        short = [k for k in PLAN_HOWS if isinstance(it.get(k), str)]
        name = str(pick("part", "name", "title", "task", "goal", "description", "what",
                        "text") or (it[short[0]] if short else "")).strip()
        name = re.split(r"(?<=[.;:])\s", name)[0].rstrip(".;:")   # one clause is enough
        if len(name) > 110:
            name = name[:110].rsplit(" ", 1)[0].rstrip(",") + " …"
        if not name or name.lower() in PLAN_HOWS:
            continue
        how = str(pick("how", "type", "method", "kind", "approach")
                  or (short[0] if short else "search")).strip().lower()
        how = how if how in PLAN_HOWS else "search"
        status = str(it.get("status", "todo")).strip().lower().replace(" ", "_")
        status = status if status in PLAN_STATUSES else "todo"
        parts.append({"part": name, "how": how, "status": status})
    return parts


_QUERY_STOP = {"the", "a", "an", "of", "for", "and", "in", "on", "to", "is", "what",
               "how", "many", "much", "vs", "by", "with", "news"}


def _query_words(q):
    """Content words of a search query, years dropped and view/views/viewing/viewership
    folded together — the rewording a model does when it re-asks a search that failed."""
    out = set()
    for w in re.findall(r"[a-z0-9]+", q.lower()):
        if w.isdigit() or w in _QUERY_STOP:
            continue
        out.add(re.sub(r"(ership|ers|ing|s)$", "", w) if len(w) > 4 else w)
    return out


def _repeats(query, earlier):
    """Earlier (step, query) pairs this query mostly repeats: 3 of 4 content words shared
    (overlap coefficient >= 0.75, at least two words). Measured on two real runs it caught
    4 of 9 searches in the BBC-ratings loop and the reshuffled 'Tornotti Nature Astronomy'
    one, and let the genuinely new 'author list' search (0.6) through."""
    a = _query_words(query)
    hits = []
    for n, q in earlier:
        b = _query_words(q)
        shared = len(a & b)
        if shared >= 2 and shared / max(1, min(len(a), len(b))) >= 0.75:
            hits.append((n, q))
    return hits


@contextlib.contextmanager
def _borrow_freely(model):
    """Default for `borrow` (the CLI): no admission control here, just try."""
    yield True, ""


def _research_core(model, messages, min_sources, effort, temperature, borrow, state):
    """The research loop as a STREAM OF EVENTS — the single source of truth behind
    both the terminal renderer (run_agent) and the web UI (webapp/engine.py). It
    drives the ReAct tool loop, mutating `messages` in place so follow-ups keep
    context. A depth gate (min_sources > 0) blocks a premature answer until enough
    distinct pages are read. Event types: step / thinking / answer_chunk /
    tool_call / observation / source / notice / final / error / done.

    Tool protocol is chosen per model: native Ollama tool calling when the model
    supports it (calls arrive parsed, in the channel the model was trained on),
    otherwise the original in-band JSON protocol. The in-band parser also stays on
    as a safety net in native mode, catching a model that writes JSON as text.

    borrow(name) is a context manager yielding (ok, why): it makes a helper model
    resident and holds it for view_image when the asking model has no vision. The web
    server passes one that goes through its RAM admission and per-model turn locks."""
    borrow = borrow or _borrow_freely
    native = model_supports_tools(model)
    # The web UI names the level outright; the CLI only says how many sources it
    # wants, so infer one from that.
    effort = effort or effort_for(min_sources)
    vision = model_has_vision(model)
    tools = (TOOL_SPECS if vision else
             [t for t in TOOL_SPECS if t["function"]["name"] != "view_image"]) if native else None
    # The two protocols need different instructions, the user can switch models
    # mid-session, and the date block must stay current in long-lived sessions —
    # so refresh the system prompt on every call. messages[0] is always the
    # system message (every working copy starts from it).
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_prompt(native, vision)
    question = messages[-1]["content"] if messages and messages[-1].get("role") == "user" else ""
    stamp_current_question(messages)  # date lands next to the question, where attention is
    # The lookup runs beside the turn. The model waits up to 4 s for it (its text is a free
    # source); a slower lookup still lands as the card, which is shown after the answer.
    import threading
    box = {}
    looker = threading.Thread(target=lambda: box.update(card=encyclopedia_card(question, box)),
                              daemon=True)
    if question:
        looker.start()
        looker.join(4)
    card = box.get("card")
    state.update(question=question, card=card, card_box=box,
                 card_thread=looker if question else None)
    if card:
        # The model gets the text the card will show, as a free source it may lean on:
        # a "What" question can then be answered without spending a research step.
        messages[-1]["content"] += (f"\n[encyclopedia, shown to the user beside your answer — "
                                    f"Wikipedia, \"{card['title']}\" ({card['url']}): "
                                    f"{card['extract']}]")
    if min_sources > 0:
        stamp_deep_brief(messages, min_sources)  # deep mode: collect first, conclude once
    read_domains = set()    # distinct sites actually fetched — this is our "depth"
    unread_urls = []        # urls seen in search results but not yet fetched
    nudges = 0
    stalls = 0              # turns that returned only reasoning — no answer, no call
    searches_in_a_row = 0   # list-tool calls since the last web_fetch (loop detector)
    plan = None             # the model's latest plan: {"reading", "parts"}
    plan_calls = 0          # plan calls so far; the first PLAN_FREE_CALLS cost no step
    queries = []            # (step, query) of every search run — the repeat detector
    fetch_parts = {}        # url -> last part read, so a re-fetch reads ON, not again
    page_figs = {}          # url -> [(alt, image url)] seen in its fetched parts
    page_titles = {}        # url -> its <title>, for the photo card
    state.update(page_figs=page_figs, page_titles=page_titles)
    viewed = set()          # image urls already looked at with view_image

    max_steps = steps_for(effort)
    extensions = 0          # how many times the model has bought itself more room
    step = 0
    wasted = 0              # steps refunded because every tool call in them failed
    try:
        while step < max_steps and wasted <= MAX_WASTED_STEPS:
            step += 1
            yield {"type": "step", "n": step, "max": max_steps}

            content = thinking = ""
            calls = []
            for ev in _stream_turn(model, messages, tools, effort, temperature):
                if "_turn" in ev:
                    content, thinking, calls = ev["_turn"]
                else:
                    yield ev

            came_natively = bool(calls)
            if not calls:
                # In-band protocol — or a native-capable model that wrote the JSON
                # into its text anyway. The lenient parser catches both.
                call = extract_tool_call(content)
                if not call and not content.strip():
                    # Content came back empty — a reasoning model (gpt-oss) may have
                    # hidden the call, or its whole answer, in the thinking channel.
                    call = extract_tool_call(thinking)
                if call:
                    calls = [call]

            if not calls:
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
                        f"{hint}\n" + ("Call the web_fetch tool now." if native else
                                       "Respond now with a web_fetch tool call (JSON only)."))})
                    continue
                if content.strip():
                    if plan:
                        # A 'know' part is answered by writing the answer. gpt-oss almost
                        # never ticks parts off, so an unmarked 'search' part claims only
                        # what the loop knows: sources were read ("answered", see them) or
                        # none were — then it came from memory, and the card says so.
                        searched = "answered" if read_domains else "unconfirmed"
                        settled = [dict(p, status=("done" if p["how"] == "know" else
                                                   searched if p["how"] == "search" else
                                                   p["status"]))
                                   if p["status"] == "todo" else p for p in plan["parts"]]
                        yield {"type": "plan", "reading": plan["reading"], "parts": settled,
                               "used": step - 1, "final": True}
                    yield {"type": "final", "text": tidy_answer(content.strip())}
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
                        "Nothing ran — you described the action but did not call a tool. "
                        "Actually invoke web_search, web_fetch or arxiv_search now, or write "
                        "the FINAL ANSWER as prose." if native else
                        "You described a tool call in words but did not emit it, so nothing ran. "
                        "Reply with ONLY the JSON object and nothing else, e.g.:\n"
                        '{"tool": "web_search", "args": {"query": "<what to look up>"}}\n'
                        "Fill in the query and send just that. If you truly don't need the web, "
                        "write the FINAL ANSWER as prose instead.")})
                    continue
                # Nudges exhausted: if it researched, synthesize from that; else answer
                # from its own knowledge (never surface raw "I need to search" reasoning).
                instr = SYNTH_FROM_EVIDENCE if read_domains else SYNTH_FROM_KNOWLEDGE
                for ev in _synthesis(model, messages, instr, effort, temperature):
                    yield ev
                yield {"type": "done"}
                return

            # --- a clarifying question ends the turn: the user answers it, not a tool ---
            # Like Claude Code's clarification prompt: when the words fit several things,
            # asking IS the action. The question becomes this turn's answer, so it is kept
            # in the conversation (and history) and the user's pick lands as a follow-up
            # to it; the `ask` event lets the page offer the options as buttons. Research
            # calls batched alongside it are dropped — the model has decided it needs the
            # user before going any further.
            ask = next((a for t, a in calls if t == "ask_user"), None)
            if ask is not None:
                state["asked"] = True
                question = str(ask.get("question") or "").strip() or "Which one did you mean?"
                options = ask.get("options") or []
                if isinstance(options, str):          # a lenient in-band call may flatten it
                    options = options.splitlines()
                options = [t for t in (_option_text(o) for o in options) if t][:6]
                yield {"type": "tool_call", "tool": "ask_user", "label": question}
                yield {"type": "ask", "question": question, "options": options}
                yield {"type": "final", "text": tidy_answer(
                    question + "".join(f"\n- {o}" for o in options))}
                yield {"type": "done"}
                return

            # --- tools were requested (a native model may batch several per turn) ---
            step_usable = False   # did anything in this step actually come back readable?
            planned_only = all(t == "plan" for t, _ in calls)
            for call_idx, (tool, args) in enumerate(calls):
                label = args.get("query") or args.get("url") or ""
                yield {"type": "tool_call", "tool": tool, "label": label}
                t0 = time.time()
                try:
                    if tool == "more_steps":
                        # Not a research tool — it edits this loop's own budget, so it is
                        # handled here rather than in TOOLS. Granting costs the step that
                        # asked, so the net gain is STEP_EXTENSION - 1; that is deliberate,
                        # since asking should never be entirely free.
                        why = str(args.get("reason", "")).strip()[:200]
                        if extensions < MAX_EXTENSIONS:
                            extensions += 1
                            max_steps += STEP_EXTENSION
                            obs = (f"Granted: +{STEP_EXTENSION} steps (now {max_steps} "
                                   f"total, extension {extensions} of {MAX_EXTENSIONS}). "
                                   f"Spend them on the plan you just described — a "
                                   f"different source or a different query, not the same "
                                   f"one again. Continue.")
                            yield {"type": "notice",
                                   "text": f"more steps requested — {why or 'no reason given'}"}
                            yield {"type": "observation", "tool": tool, "n": STEP_EXTENSION,
                                   "ok": True, "preview": f"+{STEP_EXTENSION} steps "
                                                          f"({extensions}/{MAX_EXTENSIONS})"}
                        else:
                            obs = (f"Refused: you have already extended {MAX_EXTENSIONS} "
                                   f"times, which is the limit. Write the best FINAL ANSWER "
                                   f"you can from what you have, and say plainly which "
                                   f"parts you could not confirm.")
                            yield {"type": "observation", "tool": tool, "n": 0, "ok": False,
                                   "preview": "extension limit reached"}
                        step_usable = True   # a decision was made; nothing to refund
                        if came_natively:
                            messages.append({"role": "tool", "tool_name": tool,
                                             "content": obs})
                        else:
                            messages.append({"role": "user", "content":
                                             f"OBSERVATION from {tool}:\n{obs}\n\n"
                                             f"Continue: call another tool (JSON only) or "
                                             f"write the FINAL ANSWER."})
                        continue
                    if tool == "plan":
                        # Bookkeeping, not research: free for the first PLAN_FREE_CALLS.
                        plan_calls += 1
                        parts = _plan_parts(args.get("parts") or args.get("steps")
                                            or args.get("plan") or args.get("items"))
                        reading = str(args.get("reading") or "").strip()[:300]
                        if not parts and not reading:
                            raise ValueError(
                                "plan needs `reading` and `parts`: a list of {part, how, "
                                "status}, e.g. [{\"part\": \"the term\", \"how\": \"know\"}] "
                                f"— you sent {json.dumps(args, ensure_ascii=False)[:300]}")
                        # A malformed parts list ("parts": ["search"]) keeps the earlier
                        # parts rather than costing a retry turn: the reading is the point.
                        parts = parts or (plan or {}).get("parts", [])
                        used = step - (1 if plan_calls <= PLAN_FREE_CALLS else 0)
                        plan = {"reading": reading or (plan or {}).get("reading", ""),
                                "parts": parts}
                        yield {"type": "plan", "reading": plan["reading"], "parts": parts,
                               "used": used}
                        yield {"type": "observation", "tool": tool, "n": len(parts),
                               "ok": True, "preview": f"{len(parts)} part(s)"}
                        open_know = [p["part"] for p in parts if p["how"] == "know"]
                        open_search = [p for p in parts
                                       if p["how"] == "search" and p["status"] == "todo"]
                        obs = f"Plan noted (free): {len(parts)} part(s). "
                        # Unviewed figures are named here because the plan call is where a
                        # model decides it is done (both muse runs: plan, then the answer).
                        unseen = [f for figs_ in page_figs.values() for f in figs_
                                  if f[1] not in viewed]
                        if unseen and vision:
                            obs += ("Pages you read have figures you have not looked at: " +
                                    "; ".join(f"'{a}' ({u})" for a, u in unseen[:4]) +
                                    ". If one carries a result your answer needs — for a "
                                    "page summary, the page's key chart — view_image it "
                                    "before writing. ")
                        elif not open_search:
                            obs += "Nothing left to research — write the FINAL ANSWER now. "
                        if open_know:
                            obs += ("Answer the 'know' parts yourself in the answer — no "
                                    "searching for them. ")
                        if any(p["how"] == "ask" for p in parts):
                            obs += ("For the 'ask' parts, end the answer with one short "
                                    "question for the missing detail. ")
                        obs += "Call plan again when a part is done or not found."
                        step_usable = True
                        if came_natively:
                            messages.append({"role": "tool", "tool_name": tool, "content": obs})
                        else:
                            messages.append({"role": "user", "content":
                                             f"OBSERVATION from {tool}:\n{obs}\n\n"
                                             f"Continue: call another tool (JSON only) or "
                                             f"write the FINAL ANSWER."})
                        continue
                    if tool in ("web_search", "arxiv_search"):
                        # Repeat-search discourager. One rewording of a search that came
                        # up dry is fair; a third go at the same thing is the "12-step
                        # search game" — refused, and the step is still spent, because
                        # the reply it needs is a decision, not another query.
                        query = str(args.get("query", ""))
                        again = _repeats(query, queries)
                        if len(again) >= 2:
                            prev = ", ".join(f"step {n}" for n, _ in again[:3])
                            obs = (f"NOT RUN: this is your third search for nearly the same "
                                   f"thing ({prev}). It has not turned up, and rewording it "
                                   f"again will not change that — it may simply not be "
                                   f"published in that form. Either web_fetch a specific "
                                   f"page you already found, or mark that part not_found "
                                   f"with plan and write the answer: say plainly what is "
                                   f"not published, and give the nearest thing that is.")
                            step_usable = True
                            yield {"type": "observation", "tool": tool, "n": 0, "ok": False,
                                   "preview": f"not run — repeats {prev}"}
                            yield {"type": "notice", "text": f"same search as {prev} — refused"}
                            if came_natively:
                                messages.append({"role": "tool", "tool_name": tool,
                                                 "content": obs})
                            else:
                                messages.append({"role": "user", "content":
                                                 f"OBSERVATION from {tool}:\n{obs}\n\n"
                                                 f"Continue: call another tool (JSON only) "
                                                 f"or write the FINAL ANSWER."})
                            continue
                        queries.append((step, query))
                    if tool not in TOOLS:
                        # a model inventing e.g. web_find gets a steer, not a bare KeyError
                        raise ValueError(f"unknown tool {tool!r} — the only tools are: "
                                         + ", ".join(list(TOOLS) + ["plan", "more_steps", "ask_user"]))
                    viewer = None
                    if tool == "web_fetch":
                        url = str(args.get("url", ""))
                        part = fetch_parts.get(url, 0) + 1
                        result = web_fetch(url, part)
                        fetch_parts[url] = part
                    elif tool == "view_image":
                        url = str(args.get("url", "")).strip()
                        question = str(args.get("question") or "")
                        if not vision:
                            # No helper mid-turn: a second model loading while this one
                            # works costs GPU for both (kong, 2026-10-13). Text-only models
                            # aren't offered the tool; this catches one that calls it anyway.
                            raise UnusableSource(
                                "view_image is not available to you. Work from the page "
                                f"text, and if the figure matters, link it ({url}) in the "
                                "answer and say it was not read.")
                        viewer = model            # it holds this turn already: no load
                        result = view_image(url, question, viewer)
                        viewed.add(url)
                    else:
                        result = TOOLS[tool](args)
                    if isinstance(result, list) and not result:
                        raise UnusableSource(
                            f"{tool} found no results for that query — it is too narrow "
                            "or misspelled. Rephrase it with different keywords.")
                    obs = format_observation(tool, result)
                    n = len(result) if isinstance(result, list) else 1
                    step_usable = True
                    # A long page is read in parts; say which one, or a model paging
                    # through a page looks like one re-reading it ("1 result(s)" x5).
                    pm = (re.search(r"\(part (\d+) of (\d+)\)", result[:600])
                          if tool == "web_fetch" else None)
                    what = (f"part {pm[1]} of {pm[2]}" if pm else
                            f"read by {viewer}" if viewer else f"{n} result(s)")
                    yield {"type": "observation", "tool": tool, "n": n, "ok": True,
                           "preview": f"{what} in {time.time() - t0:.1f}s"}
                    if tool == "web_fetch":  # count it as a real source read
                        searches_in_a_row = 0
                        # The system prompt alone did not do it: muse-glimmer summarised
                        # the HF page with the KLD chart marker right in front of it and
                        # never looked (2026-10-13). Say it where attention is.
                        # A paged read defers: it notes the figure on part 1 and has
                        # forgotten it by part 5 — so the last part repeats the page's
                        # figures not yet viewed.
                        figs = re.findall(r"\[image: (.+?) — (\S+?)(?:, \d+x\d+)?\]", result)
                        page_titles.setdefault(url, re.sub(
                            r"\s+[|\-–—:]\s+[^|\-–—:]*$", "",
                            result.split("\n", 1)[0].removeprefix("TITLE: "))[:90])
                        page_figs.setdefault(url, []).extend(f for f in figs
                                                             if f not in page_figs[url])
                        last = not pm or pm[1] == pm[2]     # single-part page, or its end
                        not_seen = [f for f in (page_figs[url] if last else figs)
                                if f[1] not in viewed]
                        if not_seen and vision:
                            obs += ("\n\nNOTE: " + ("this page has" if last else "this part has") +
                                    (f" {len(not_seen)} images" if len(not_seen) > 1 else " an image") +
                                    " the text cannot show, not viewed yet: " +
                                    "; ".join(f"'{a}' ({u})" for a, u in not_seen[:4]) +
                                    ". If one is a chart, plot, table or diagram that carries "
                                    "a result this question needs — for a page summary, the "
                                    "page's key figure — call view_image on it before "
                                    "answering.")
                        dom = urllib.parse.urlparse(args.get("url", "")).netloc
                        if dom and fetch_parts.get(url) == 1:   # list a page once, not per part
                            read_domains.add(dom)
                            yield {"type": "source", "url": args.get("url", "")}
                    elif isinstance(result, list):  # remember URLs we could fetch later
                        # A previewed result was really downloaded and read, so it counts
                        # as a source: it gets a source event (the UI's clickable list) and
                        # a read domain (the depth gate). Without this a run that answered
                        # straight from previews would show no sources at all.
                        for it in result:
                            if it.get("preview") and it.get("url"):
                                dom = urllib.parse.urlparse(it["url"]).netloc
                                if dom and dom not in read_domains:
                                    read_domains.add(dom)
                                    searches_in_a_row = 0
                                    yield {"type": "source", "url": it["url"]}
                        unread_urls.extend(it["url"] for it in result
                                           if it.get("url") and not it.get("preview"))
                        # Loop detector: a model re-searching the same question over and
                        # over (instead of reading or answering) burns minutes for nothing.
                        # After the 3rd search with no page read in between, say so to its
                        # face — right inside the observation it's about to act on.
                        again = (_repeats(args.get("query", ""), queries[:-1])
                                 if tool in ("web_search", "arxiv_search") else [])
                        if again:
                            obs += (f"\n\nNOTE: this search mostly repeats step "
                                    f"{again[0][0]}'s ({again[0][1]!r}). If what you need "
                                    f"is not in these results either, that is a finding — "
                                    f"it may not be published in that form. Say so in the "
                                    f"answer rather than searching a third way.")
                        searches_in_a_row += 1
                        if searches_in_a_row >= 3:
                            obs += (f"\n\nNOTE: that was search #{searches_in_a_row} in a row "
                                    "without reading a single page. Searching again will NOT "
                                    "surface anything new. Either web_fetch ONE of the URLs "
                                    "above, or — if you already know enough — write the FINAL "
                                    "ANSWER now.")
                except UnusableSource as e:
                    obs = str(e)          # already a plain steer; no traceback dressing
                    yield {"type": "observation", "tool": tool, "n": 0, "ok": False,
                           "preview": f"unusable: {str(e)[:80]}"}
                except Exception as e:
                    obs = f"ERROR running {tool}: {type(e).__name__}: {e}"
                    yield {"type": "observation", "tool": tool, "n": 0, "ok": False, "preview": obs}

                # Feed the result back in whichever protocol the call arrived by. A
                # natively-called turn gets a real `tool` message (the template
                # renders it the way the model was trained on, and no "Continue:"
                # coaching is needed); an in-band call gets the OBSERVATION message.
                if came_natively:
                    messages.append({"role": "tool", "tool_name": tool, "content": obs})
                else:
                    messages.append({"role": "user", "content":
                                     f"OBSERVATION from {tool}:\n{obs}\n\n"
                                     f"Continue: call another tool (JSON only) or write the FINAL ANSWER."})

            # --- refund a step that yielded nothing, then post the budget clock ---
            # A 403, a PDF, a consent wall or an empty search costs the user real time
            # but teaches the model nothing, so charging a step for it shortens the
            # research for no reason. Give the step back and let it try another source.
            # `wasted` is capped so a model stuck on broken links still terminates:
            # the run can never exceed max_steps + MAX_WASTED_STEPS + 1 iterations.
            refunded = bool(calls) and not step_usable
            if refunded:
                wasted += 1
                step -= 1
            elif calls and planned_only and plan_calls <= PLAN_FREE_CALLS:
                step -= 1         # planning is free — it is not research
            if calls:
                # Clock goes on the LAST message of the step, after the whole step's
                # outcome is known. It rides inside the tool/observation content on
                # purpose: a user-role note would move the chat template's
                # "last user message" marker and amputate the retained reasoning.
                if refunded:
                    clock = (f"\n\n[that source gave nothing usable, so step {step + 1} was "
                             f"NOT charged — you still have {max_steps - step} of "
                             f"{max_steps} research steps. Try a different source.]")
                else:
                    left = max_steps - step
                    clock = f"\n\n[progress: research step {step} of {max_steps}"
                    if plan:
                        open_n = sum(p["status"] == "todo" and p["how"] == "search"
                                     for p in plan["parts"])
                        if open_n:
                            clock += (f" · {open_n} search part(s) of your plan still open "
                                      "— mark each done or not_found with plan as you "
                                      "settle it")
                    if left <= 3:
                        # Two exits, stated together on purpose. Telling it only to
                        # converge is right when it HAS what it needs and wrong when its
                        # sources fell through — and only the model can tell those apart.
                        clock += (f" — only {left} step(s) remain before the final answer "
                                  "is forced. If you have enough, converge and land the "
                                  "answer yourself")
                        if extensions < MAX_EXTENSIONS:
                            clock += (". If instead the pages you drew were poor or "
                                      "off-topic and you know what to try next, call "
                                      "more_steps with a reason")
                    clock += "]"
                messages[-1]["content"] += clock

            # Once we've pushed it enough, stop looping and make it write the report.
            if nudges >= MAX_NUDGES and len(read_domains) >= 1:
                break

        instr = SYNTH_FROM_EVIDENCE if read_domains else SYNTH_FROM_KNOWLEDGE
        for ev in _synthesis(model, messages, instr, effort, temperature):
            yield ev
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "text": f"{type(e).__name__}: {e}"}
        yield {"type": "done"}


def research_events(model, messages, min_sources=MIN_SOURCES, effort=None, temperature=None,
                    borrow=None):
    """The research loop (see _research_core), then the card: text first, picture after.
    kong (2026-10-13): "Text first, then ... another image capable model to find an
    image, place the image onto the brief card, deliver it afterwards (so I can already
    read the text)". The card stage runs after the final answer and before `done`, still
    inside the turn, so a borrowed vision model is held under the same turn lock."""
    state = {}
    final = None
    for ev in _research_core(model, messages, min_sources, effort, temperature, borrow, state):
        if ev.get("type") == "final":
            final = ev.get("text")
        if ev.get("type") == "done" and final and not state.get("asked"):
            try:
                yield from card_stage(model, borrow or _borrow_freely, state)
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield {"type": "card_none", "why": f"{type(e).__name__}: {e}"}
        yield ev


CARD_PICK_MAX = 4        # candidate photos shown to the vision model at once
CARD_PICK_SIDE = 768     # px; enough to recognise a subject, cheap in tokens (~30 s pick)

def card_stage(model, borrow, state):
    """One card per answer, delivered after the text. A Wikipedia match is copied as is
    (the Quill case). Otherwise up to CARD_PICK_MAX photos from pages read — charts,
    screenshots, logos and anything sharing no key word with the question are filtered
    out first — are shown to a vision model, which picks the one that shows the subject,
    or none, and writes the card's lines. Tested 2026-10-13 on muse-glimmer: the Orange
    Pi board picked in 3/3 orders over a portrait and a chart, and "none" when it was
    absent."""
    if not state.get("card") and state.get("card_thread"):
        state["card_thread"].join(10)     # a lookup slower than the model's head start
        state["card"] = state["card_box"].get("card")
    if state.get("card"):
        yield {"type": "card", **state["card"]}
        return
    qwords = card_words(state.get("question", ""))
    seen, cands = set(), []
    for page, figs in state.get("page_figs", {}).items():
        for alt, src in figs:
            sc = photo_score(alt, src, qwords) if qwords else 0
            if sc and src not in seen:
                seen.add(src)
                cands.append((sc, alt, src, page))
    if not cands:
        why = state.get("card_box", {}).get("why")
        if why:
            yield {"type": "card_none", "why": why}
        return
    cands = sorted(cands, key=lambda c: -c[0])[:CARD_PICK_MAX]
    yield {"type": "card_wait"}
    import concurrent.futures as cf
    def grab(c):
        try:
            return c, _image_png_b64(c[2], CARD_PICK_SIDE)[0]
        except Exception:
            return c, None            # slow or blocked host: that candidate just drops out
    with cf.ThreadPoolExecutor(len(cands)) as pool:
        got = [(c, b) for c, b in pool.map(grab, cands) if b]
    if not got:
        yield {"type": "card_none", "why": f"none of {len(cands)} candidate pictures downloaded"}
        return
    prompt = (f"A user asked: {state.get('question', '').strip()[:500]}\n"
              f"The {len(got)} image(s) above are numbered 1 to {len(got)} in order. Pick the "
              "ONE that best shows what the question is about — a photo of the real thing, "
              "not a chart, screenshot, logo or unrelated person. If none plainly shows it, "
              "pick 0.\nReply exactly:\nPICK: <number>\nCAPTION: <one or two short "
              "sentences describing what the picked image shows>")
    msgs = [{"role": "user", "content": prompt, "images": [b for _, b in got]}]
    viewer = model if model_has_vision(model) else VISION_MODEL
    out = ""
    with (contextlib.nullcontext((True, "")) if viewer == model else borrow(viewer)) as (ok, _):
        if not ok:
            yield {"type": "card_none", "why": f"{viewer} not available: {_}"}
            return
        for kind, chunk in ollama_chat_stream(viewer, msgs, effort=False, temperature=0.3):
            if kind == "content":
                out += chunk
    m = re.search(r"PICK:\s*(\d+)", out)
    pick = int(m.group(1)) if m else 0
    if not 1 <= pick <= len(got):
        yield {"type": "card_none", "why": f"{viewer} found no fitting picture among "
                                          f"{len(got)}: {out.strip()[:200]}"}
        return
    (_, alt, src, page), _ = got[pick - 1]
    cap = re.search(r"CAPTION:\s*(.+)", out, re.S)
    yield {"type": "card", "title": state.get("page_titles", {}).get(page) or alt[:90],
           "extract": (cap.group(1).strip() if cap else alt)[:300], "image": src,
           "url": page, "source": urllib.parse.urlparse(page).netloc.removeprefix("www."),
           "photo": True, "by": viewer}


def run_agent(model, messages, min_sources=MIN_SOURCES, temperature=None):
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

    for ev in research_events(model, messages, min_sources, temperature=temperature):
        t = ev.get("type")
        if t == "step":
            newline()
            gate = f", read {read}/{min_sources}" if min_sources > 0 else ""
            sys.stdout.write(c(f"\n  ┄ thinking (step {ev['n']}/{ev.get('max', MAX_STEPS)}"
                               f"{gate}) ┄\n", C.dim))
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
        elif t == "card":
            newline()
            sys.stdout.write(c(f"  ▣ {ev['source']}: {ev['title']} — {ev['url']}\n", C.cyan))
            at_line_start = True
        elif t == "plan":
            newline()
            marks = {"done": "✓", "not_found": "✗"}
            lines = [f"  ▤ plan · {ev.get('reading', '')}"]
            lines += [f"     {marks.get(p['status'], '·')} {p['part']} ({p['how']})"
                      for p in ev.get("parts", [])]
            sys.stdout.write(c("\n".join(lines) + "\n", C.cyan))
            at_line_start = True
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
    # Protocol variant and date are both refreshed per-call in research_events;
    # this just seeds the slot so messages[0] is always the system message.
    return [{"role": "system", "content": system_prompt(False)}]

def check_model(model):
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            tags = json.load(r)
        names = [m["name"] for m in tags.get("models", [])]
        if model not in names and names:
            print(c(f"note: '{model}' not in `ollama list`. Available: {', '.join(names)}", C.yellow))
    except Exception:
        print(c(f"warning: can't reach Ollama at {OLLAMA_HOST} — is it running?", C.red))

def repl(model, min_sources=MIN_SOURCES, temperature=None):
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
            answer = run_agent(model, messages, min_sources, temperature)
        except KeyboardInterrupt:
            print(c("\n(interrupted)", C.yellow)); continue
        print(c("\n─── answer ───", C.bold))
        print(answer.strip(), "\n")

def one_shot(model, question, min_sources=MIN_SOURCES, temperature=None):
    messages = new_conversation()
    messages.append({"role": "user", "content": question})
    answer = run_agent(model, messages, min_sources, temperature)
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
  --temp <t>           sampling temperature {TEMP_MIN}-{TEMP_MAX} (default {TEMPERATURE}:
                       OpenAI's recommendation for gpt-oss — lower makes it repeat itself)
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
    temp_opt = _take_opt(args, "--temp")
    temperature = None
    if temp_opt:
        try:
            temperature = min(TEMP_MAX, max(TEMP_MIN, float(temp_opt)))
        except ValueError:
            print(c(f"ignoring --temp {temp_opt!r}: not a number", C.yellow))
    question = " ".join(args).strip()

    check_model(model)
    if question:
        one_shot(model, question, min_sources, temperature)
    else:
        repl(model, min_sources, temperature)

if __name__ == "__main__":
    main()
