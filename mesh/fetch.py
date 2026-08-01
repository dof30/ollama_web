#!/usr/bin/env python3
"""
fetch — parallel page retrieval and text extraction.

This is where the wall-clock win lives. Eight pages come from eight different
hosts, so fetching them concurrently costs about as long as the slowest single
one instead of the sum of all eight. There is no rate limit to respect here — one
request each to eight unrelated servers is not a burst, it is a browser opening
eight tabs.

Contrast with search.py, where queries all hit *one* endpoint and so must be
serialised. Same word, "parallel"; opposite conclusion. The distinction is
whether the load lands on one shoulder or many.
"""

import concurrent.futures
import re
import threading
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from . import config

_local = threading.local()

# Sites that never yield usable text to a plain HTTP GET: the page is a JS shell
# and the content arrives later over XHR. Fetching them wastes a slot and, worse,
# hands the reader 200 characters of boilerplate to hallucinate from.
SKIP_DOMAINS = {
    "youtube.com", "youtu.be", "m.youtube.com",
    "twitter.com", "x.com", "instagram.com", "facebook.com", "tiktok.com",
    "linkedin.com", "pinterest.com",
}

# Same content, static markup. old.reddit.com serves real HTML to a plain GET while
# www.reddit.com returns a JS shell — a one-line rewrite that turns 37 characters of
# nothing into an actual thread. Forum answers are often the best source for
# "how do I do X in <application>", so this matters more than it looks.
REWRITE_HOSTS = {
    "www.reddit.com": "old.reddit.com",
    "reddit.com": "old.reddit.com",
    "np.reddit.com": "old.reddit.com",
}

# Below this, a "successful" fetch is really a cookie wall or a JS shell. Treated as
# a failure so the caller can fall back to the search snippet instead.
MIN_USEFUL_CHARS = 250

# Elements that are furniture, not content. Stripping them is the cheapest
# quality win available: it is most of what used to fill the big model's context.
_STRIP_TAGS = (
    "script", "style", "noscript", "nav", "footer", "header",
    "form", "aside", "iframe", "svg", "button",
)

_WS = re.compile(r"\n{3,}")


def _session():
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        # A fuller header set than just User-Agent. Several sites (Autodesk's forums
        # among them) return 403 to a request that claims to be Firefox but omits the
        # headers every real Firefox sends — the mismatch is the tell, not the UA.
        s.headers.update(
            {
                "User-Agent": config.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Connection": "keep-alive",
            }
        )
        _local.session = s
    return s


def host_of(url):
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


def prepare(urls):
    """Drop URLs that can never yield text, and rewrite those with a static twin."""
    out = []
    for u in urls:
        host = host_of(u)
        bare = host[4:] if host.startswith("www.") else host
        if bare in SKIP_DOMAINS:
            continue
        target = REWRITE_HOSTS.get(host)
        out.append(u.replace(host, target, 1) if target else u)
    return out


def extract(html):
    """(title, text) from an HTML document, furniture removed."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    text = soup.get_text("\n")
    lines = (ln.strip() for ln in text.splitlines())
    text = "\n".join(ln for ln in lines if ln)
    return title, _WS.sub("\n\n", text)


def fetch_one(url, chars=None):
    """Retrieve and clean one page.

    Returns {"url","title","text","ok"} — never raises. A dead link is normal
    during research and must not take down the round; the caller sees ok=False
    and moves on to the next candidate.
    """
    chars = chars or config.PAGE_CHARS
    try:
        r = _session().get(url, timeout=config.FETCH_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "html" not in ctype and "xml" not in ctype and ctype:
            return {"url": url, "title": "", "text": "", "ok": False,
                    "error": f"not a page ({ctype.split(';')[0]})"}
        title, text = extract(r.text)
        if len(text) > chars:
            text = text[:chars].rstrip() + "\n…[truncated]"
        if len(text.strip()) < MIN_USEFUL_CHARS:
            # A cookie wall or JS shell. Call it a failure so the caller falls back
            # to the search snippet rather than feeding the reader boilerplate.
            return {"url": url, "title": title, "text": "", "ok": False,
                    "error": f"only {len(text.strip())} chars (JS-only or blocked)"}
        return {"url": r.url, "title": title, "text": text, "ok": True, "error": ""}
    except requests.RequestException as e:
        return {"url": url, "title": "", "text": "", "ok": False, "error": str(e)[:160]}
    except Exception as e:  # malformed HTML, decoding failures, etc.
        return {"url": url, "title": "", "text": "", "ok": False, "error": str(e)[:160]}


def fetch_many(urls, workers=None):
    """Fetch concurrently. Yields results as they land, fastest first.

    Yielding rather than returning a list matters: the caller can start handing
    pages to the reader while slower ones are still in flight, so fetching and
    distilling overlap instead of queueing behind one another.
    """
    urls = list(dict.fromkeys(u for u in urls if u))  # dedupe, keep order
    if not urls:
        return
    workers = min(workers or config.FETCH_WORKERS, len(urls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, u): u for u in urls}
        for fut in concurrent.futures.as_completed(futures):
            try:
                yield fut.result()
            except Exception as e:
                yield {"url": futures[fut], "title": "", "text": "",
                       "ok": False, "error": str(e)[:160]}
