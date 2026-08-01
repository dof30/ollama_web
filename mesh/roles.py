#!/usr/bin/env python3
"""
roles — the three jobs, and which model does each.

    router   (small)  Does this even need the web? If so, which queries?
    reader   (small)  One page in, ~150 words of evidence out. Runs N at a time.
    assess   (small)  Is the evidence enough yet, or what's still missing?
    synth    (big)    Evidence in, cited answer out. Streams to the user.

The split is not "easy questions to the small model, hard ones to the big model"
— that would just divide the queue. It is **volume vs judgment**. Reading ten
pages is volume: mechanical, parallel, and the answer is on the page. Deciding
what is actually true is judgment, and that stays with the 120b.

The payoff is that the big model is called once or twice per question instead of
twelve times, and every one of those calls sees clean distilled evidence rather
than 60 000 characters of nav bars.
"""

import datetime
import json
import re

from . import config, ollama

# --- date awareness ----------------------------------------------------------
# A local model's weights are frozen at training time, so "latest" and "current"
# silently mean "as of training" unless you say otherwise. Carried over from the
# old agent because it fixed a real and very confusing class of wrong answer.


def date_block():
    today = datetime.date.today()
    return (
        f"Today's date is {today:%A, %d %B %Y}. Your training data is older than "
        f"this. Treat 'current', 'latest', 'recent' and 'now' as meaning "
        f"{today:%B %Y}, and prefer what the sources say over what you remember."
    )


# --- JSON coaxing ------------------------------------------------------------


def _first_json_object(text):
    """Pull the first balanced {...} out of a model's reply.

    Small models wrap JSON in prose or fences no matter how firmly you ask them
    not to. Brace-matching is duller than a parser but survives all of it.
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _assist(messages, timeout=180):
    """qwen — classification, JSON, screening. Cheap and structured."""
    return ollama.complete(
        config.ASSIST_MODEL,
        messages,
        config.SMALL_OPTIONS,
        timeout=timeout,
        think=config.SMALL_THINK,
    )


def _read(messages, timeout=420):
    """gpt-oss at low effort — reading page prose. The primary model, always."""
    return ollama.complete(
        config.PRIMARY_MODEL,
        messages,
        config.BIG_OPTIONS,
        timeout=timeout,
        think=config.READ_THINK,
    )


# Kept so existing call sites keep working; the helper jobs all run on qwen.
_small = _assist


# --- resolve -----------------------------------------------------------------

RESOLVE_SYSTEM = """You rewrite a follow-up question so it stands on its own.

{date}

The user is mid-conversation and speaks naturally: "it", "that one", "the second \
one", "compared to the other". You have the earlier turns. Replace every such \
reference with the thing it actually names, so the question can be understood by \
someone who has not read the conversation.

Reply with ONLY: {{"question": "the rewritten question"}}

Rules:
- Change nothing else. Keep the user's wording, scope and tone; you are resolving \
references, not improving the question.
- Never answer it, never expand it, never add detail the user did not ask for.
- If it already stands alone, return it unchanged.

Getting this wrong is expensive: everything downstream — the searches, the pages \
read, the final answer — uses your version. Resolve "it" to the wrong subject and \
the whole run researches the wrong thing."""


def resolve(question, history_text):
    """Rewrite a follow-up into a standalone question.

    Without this, "can it handle CSI cameras better than the S3?" loses what "it"
    was: the searches, the reader and the synthesiser all receive the bare text, so
    the run silently researches whatever nouns happen to be present.
    """
    if not history_text:
        return question
    try:
        raw = _assist(
            [
                {"role": "system", "content": RESOLVE_SYSTEM.format(date=date_block())},
                {"role": "user", "content": (
                    f"Earlier turns:\n{history_text}\n\n"
                    f"Follow-up question: {question}"
                )},
            ],
            timeout=120,
        )
    except ollama.OllamaError:
        return question
    data = _first_json_object(raw) or {}
    out = str(data.get("question", "")).strip()
    # Guard against the model returning something wild: a resolved question should
    # be a modest expansion, never a rewrite into something unrecognisable.
    if not out or len(out) > max(400, len(question) * 4):
        return question
    return out


# --- router ------------------------------------------------------------------

ROUTER_SYSTEM = """You are the triage step of a research system. You do NOT answer \
the question. You decide how much work it deserves, and you are the reason the \
system does not waste a minute on questions that need a second.

{date}

Reply with ONLY a JSON object:

{{"mode": "direct" | "quick" | "deep",
  "queries": ["search query", ...],
  "reason": "one short sentence"}}

The question you are answering is NOT "could I answer this myself?" — you almost \
always could, and that is the trap. It is "**would a source make this answer more \
correct or more trustworthy?**" Searching is the whole point of this tool; a \
confident answer from memory that is two years stale is the exact failure it \
exists to prevent.

Choosing the mode:

- "quick" — a SINGLE fact settles it, and you would know it when you saw it: who \
holds a post, the price of a thing, which version is current, a menu path, whether \
something has shipped. One clean source ends the matter. Give 1-2 queries.
- "deep" — the answer has parts, or judgment, or could be wrong in an interesting \
way. Give 3-4 queries covering genuinely different angles. Choose deep when the \
question:
  * compares things ("X vs Y", "is A better than B") — a comparison needs both sides;
  * asks **why** something is the case — causes are contested and one source gives \
you one opinion;
  * contains more than one sub-question (kong often asks two or three at once);
  * rests on a premise that might be wrong — you need sources that could contradict it;
  * concerns how good, popular, reliable or worthwhile something is, which is \
opinion aggregated across many voices, not a fact you can look up once.
- "direct" — no search, and you must be able to justify it. Only for things no \
source could settle better than you: arithmetic and logic, grammar and translation, \
the meaning of a word, established maths and physics, writing or explaining code \
whose language semantics do not change, opinions and preferences, and anything \
about this conversation itself. Use an empty queries list.

When torn between "direct" and "quick", choose **quick**. A wasted search costs a \
few seconds. A stale answer delivered confidently costs the user their trust in \
every other answer.

When torn between "quick" and "deep", choose **deep**. "Quick" is for questions with \
one right answer sitting on one page; nearly everything else benefits from a second \
source. The cost of deep is a few minutes, which the user has explicitly said they \
are happy to spend on a good answer.

What they are NOT happy with is going through the motions: rounds that re-run a \
search already run, pages skimmed for the sake of a counter. Every query you give \
should be after something different from the others.

Write queries as a person would type them into a search box: keywords, no \
punctuation, no "site:" operators. Include the year only when recency is the point."""


def route(question, history_text=""):
    """Decide how much research the question deserves. One small-model call."""
    ctx = f"Earlier in this conversation:\n{history_text}\n\n" if history_text else ""
    try:
        raw = _small(
            [
                {"role": "system", "content": ROUTER_SYSTEM.format(date=date_block())},
                {"role": "user", "content": f"{ctx}Question: {question}"},
            ],
            timeout=120,
        )
    except ollama.OllamaError:
        # If triage fails, assume the middle path rather than guessing wrong in
        # either direction: answering blind, or grinding through a deep search.
        return {"mode": "quick", "queries": [question], "reason": "router unavailable"}

    data = _first_json_object(raw) or {}
    mode = str(data.get("mode", "")).lower().strip()
    if mode not in ("direct", "quick", "deep"):
        mode = "quick"
    queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
    if mode == "direct":
        queries = []
    elif not queries:
        queries = [question]
    cap = 2 if mode == "quick" else 4
    return {
        "mode": mode,
        "queries": queries[:cap],
        "reason": str(data.get("reason", "")).strip()[:200],
    }


# --- reader ------------------------------------------------------------------

READER_SYSTEM = """You are reading web pages and taking notes for yourself. Later, \
with these notes in front of you and the pages gone, you will write the full answer. \
Write down what you will wish you had kept.

{date}

You will be given several numbered pages. Write one block per page, in order, each \
starting with its marker on its own line:

===PAGE 1===
<notes, or NOT_RELEVANT>
===PAGE 2===
<notes, or NOT_RELEVANT>

For each page, up to 350 words:
- Keep the page's own language. Quote its actual sentences, in "quotation marks", \
wherever the phrasing carries meaning — an explanation, a caveat, a vivid detail, a \
strong opinion. A stripped-down list of facts is NOT what you want later: you cannot \
write a good paragraph from a bad summary, and the wording is half the evidence.
- Keep every concrete specific: numbers, dates, versions, names, prices, model \
numbers, benchmark figures.
- Keep the reasoning, not just the conclusion. If the page explains WHY, write the \
why down; that is usually the part worth reading and the first thing a summary loses.
- Note the page's date if visible, and say if it looks outdated.
- Write down what cuts AGAINST the question's assumption too. If a question rests on \
a wrong premise, the evidence that says so is the most valuable thing on the page.
- Never add knowledge of your own here. This page only — your own knowledge is \
welcome later, when writing, but must not be mixed into the record of what a source said.

Write NOT_RELEVANT for a paywall, cookie notice, error page, navigation stub, or a \
page simply off-topic. Emit a block for every page, including those."""

NOT_RELEVANT = "NOT_RELEVANT"
_PAGE_MARK = re.compile(r"^=+\s*PAGE\s*(\d+)\s*=+\s*$", re.I | re.M)


def read_page(question, page):
    """Read one page into one note. Returns note text, or None if nothing useful.

    One page per call, deliberately. Batching three pages per call looked like a 3x
    saving, but on a single-GPU box Ollama serialises anyway — so the batch bought
    no speed and cost the things that actually matter: visible per-page progress, a
    chance to re-decide after each source, and notes that stay about one page.
    """
    return read_pages(question, [page])[0]


def read_pages(question, pages):
    """Read several pages in one call. Returns a list aligned with `pages`.

    Kept for the case where a caller genuinely has pages in hand at once; the
    pipeline reads one at a time via read_page().
    """
    pages = [p for p in pages]
    if not pages:
        return []

    parts = []
    for i, p in enumerate(pages, 1):
        parts.append(
            f"===PAGE {i}===\n"
            f"Title: {p.get('title') or '(untitled)'}\nURL: {p.get('url')}\n\n"
            f"{p.get('text', '')}"
        )
    user = (
        f"Question: {question}\n\n"
        f"Write one evidence block per page, using the ===PAGE n=== markers.\n\n"
        + "\n\n".join(parts)
    )

    try:
        # gpt-oss, not the assistant. This is the whole point: the model that writes
        # is the model that reads, so it keeps the source's own language to write from.
        out = _read(
            [
                {"role": "system", "content": READER_SYSTEM.format(date=date_block())},
                {"role": "user", "content": user},
            ]
        )
    except ollama.OllamaError:
        return [None] * len(pages)

    # Split on the markers. Anything before the first marker is preamble we ignore.
    notes = [None] * len(pages)
    chunks = _PAGE_MARK.split(out)
    for idx in range(1, len(chunks) - 1, 2):
        try:
            n = int(chunks[idx]) - 1
        except ValueError:
            continue
        if not 0 <= n < len(pages):
            continue
        body = chunks[idx + 1].strip()
        body = re.sub(r"^\s*(?:#+\s*)?(?:summary|evidence|notes?)\s*:?\s*", "",
                      body, flags=re.I)
        if body and NOT_RELEVANT not in body.upper()[:60]:
            notes[n] = body.strip()

    # If the model ignored the markers entirely and there was only one page, take
    # the whole reply rather than throwing away a perfectly good note.
    if len(pages) == 1 and notes[0] is None:
        body = out.strip()
        if body and NOT_RELEVANT not in body.upper()[:60]:
            notes[0] = body
    return notes


# --- assess ------------------------------------------------------------------

ASSESS_SYSTEM = """You check whether gathered evidence is enough to answer a \
question, and you are the brake on a system that would otherwise keep searching \
forever.

{date}

Reply with ONLY a JSON object:

{{"sufficient": true | false,
  "queries": ["follow-up search", ...],
  "missing": "one short sentence"}}

Say true when a careful person could now write a solid, honest answer — even if \
more reading would add nuance. Perfect coverage is not the bar; "can we answer \
this well" is.

Say false only when something the question directly asks about is genuinely \
absent or the sources contradict each other on a key point. Then give 1-3 \
queries aimed squarely at the gap — different wording from what was already \
tried, not a rephrasing of it."""


def assess(question, evidence_text, tried):
    """Decide whether to run another round. One small-model call."""
    try:
        raw = _small(
            [
                {"role": "system", "content": ASSESS_SYSTEM.format(date=date_block())},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Searches already tried: {', '.join(sorted(tried)) or '(none)'}\n\n"
                        f"Evidence gathered so far:\n{evidence_text}"
                    ),
                },
            ],
            timeout=150,
        )
    except ollama.OllamaError:
        return {"sufficient": True, "queries": [], "missing": ""}

    data = _first_json_object(raw) or {}
    queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
    sufficient = bool(data.get("sufficient", True))
    if not queries:
        sufficient = True  # nothing actionable to search for; stop rather than spin
    return {
        "sufficient": sufficient,
        "queries": queries[:3],
        "missing": str(data.get("missing", "")).strip()[:200],
    }


SCREEN_SYSTEM = """You screen search results before an expensive reader spends time \
on them. You are the filter that decides what is worth reading, and reading a junk \
page costs far more than skipping a decent one.

{date}

You get a numbered list of results (title, site, snippet). Reply with ONLY:

{{"keep": [1, 4, 5], "why": "one short sentence"}}

Keep a result when the title or snippet suggests it genuinely addresses the question: \
documentation, a primary source or spec, a substantive article, a forum thread with \
real answers, a review with actual measurements.

Drop: shopping and price-comparison pages, ad farms and SEO listicles, pure video \
pages, login or paywall stubs, category and tag index pages, sites in a language \
other than the question's, and anything whose snippet shows it only mentions the \
topic in passing.

Prefer variety of sites over several pages from one. Keep the order given. If \
almost everything looks weak, still keep the best few — an empty list wastes the \
search entirely."""


def screen(question, results, limit):
    """Pick which search results are worth fetching. Runs on the assistant model.

    Classifying a list of titles and snippets is exactly qwen's kind of job, and it
    protects the expensive reader from spending a 10 s call on a shopping page. On
    failure it degrades to "keep the first `limit`", which is the old behaviour.
    """
    if not results:
        return []
    listing = "\n".join(
        f"{i}. {r.get('title', '')[:110]} [{r.get('url', '')[:70]}]\n   "
        f"{(r.get('snippet') or '')[:220]}"
        for i, r in enumerate(results, 1)
    )
    try:
        raw = _assist(
            [
                {"role": "system", "content": SCREEN_SYSTEM.format(date=date_block())},
                {"role": "user", "content": f"Question: {question}\n\nResults:\n{listing}"},
            ],
            timeout=150,
        )
    except ollama.OllamaError:
        return results[:limit]

    data = _first_json_object(raw) or {}
    keep = []
    for n in (data.get("keep") or []):
        try:
            i = int(n) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(results) and results[i] not in keep:
            keep.append(results[i])
    return (keep or results)[:limit]


WIDEN_SYSTEM = """You are given a research question and the searches already run. \
The evidence so far looks adequate, but we want a genuinely different angle before \
writing — a second opinion, not a second phrasing.

{date}

Reply with ONLY: {{"queries": ["...", "..."]}}

Give 2-3 searches that would surface something the existing ones would have missed: \
a critic or a downside, a competing product or method, a primary source or spec \
sheet, a practitioner's account, or a more recent development. Do not reword the \
searches already tried."""


def widen(question, tried):
    """Find a different angle when the floor says keep going but evidence looks fine.

    Without this, a forced extra round just re-runs a near-identical search and adds
    nothing — depth for its own sake, which is the exact waste we are avoiding.
    """
    try:
        raw = _small(
            [
                {"role": "system", "content": WIDEN_SYSTEM.format(date=date_block())},
                {"role": "user", "content": (
                    f"Question: {question}\n\n"
                    f"Already tried: {', '.join(sorted(tried)) or '(none)'}"
                )},
            ],
            timeout=150,
        )
    except ollama.OllamaError:
        return []
    data = _first_json_object(raw) or {}
    return [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()][:3]


# --- synthesiser -------------------------------------------------------------

SYNTH_SYSTEM = """You are a careful research assistant with a local web-research \
system working for you. Junior readers have already been through the pages and \
written up what each one says; you are reading their notes, not the raw web.

{date}

Write the answer the user actually asked for:
- Lead with the answer. No throat-clearing, no restating the question.
- Then actually develop it. Give the reasons behind the answer, the numbers and \
specifics that support it, and the caveats that would change it — a reader should \
finish understanding *why*, not just *what*. Aim a little fuller than feels \
necessary: several substantial paragraphs for a real question. Stop when you have \
said what matters; padding and restating are worse than being brief.
- Match the question's size. A one-line factual question still gets a short answer; \
a question with genuine parts gets the room to cover them.
- Where the question rests on a premise the sources contradict, say so early and \
explain the discrepancy rather than answering as if the premise held.
- Cite with markdown links, [like this](https://example.com), on the specific \
claims that came from a source.
- Every source URL below is a real page that was really read. Never invent a URL, \
and never cite one that is not listed.
- If the notes disagree, say so and say which you find more credible and why. \
That is more useful than a confident average of the two.
- Where the notes fall short, **answer anyway from your own knowledge** and label \
that part plainly — a line like "Not from the sources, from general knowledge:" is \
enough. Searching can come back thin because pages were blocked or empty, and that \
is not a reason to withhold an answer you actually know. Refusing to answer a \
question you could have answered is the one outcome worse than an unsourced answer.
- Flag the parts most worth double-checking, especially anything version-specific \
like a menu path, since those move between releases.
- Prefer specifics from the notes — numbers, dates, versions — over generalities.
- Write plain markdown. No LaTeX or \\( \\) math delimiters: the renderer does not \
understand them and shows them raw."""

DIRECT_SYSTEM = """You are a knowledgeable, direct assistant running locally.

{date}

Answer from what you know. Be conversational and get to the point — match the \
length of the answer to the size of the question, and skip the preamble.

You have no sources in front of you for this one, because it was judged not to \
need any. If the honest answer turns out to depend on something current that you \
cannot be sure of, say which part you are unsure about and suggest the user ask \
again with "search" in the question.

Write plain markdown. No LaTeX or \\( \\) math delimiters — this is rendered by a \
small markdown renderer that does not understand them, so \\(17 \\times 23 = 391\\) \
reaches the reader exactly like that. Write 17 x 23 = 391."""


def synth_messages(question, evidence_text, history_text, mode):
    """Build the big model's prompt: system + bounded history + question + evidence.

    Note what is NOT here: no tool-call transcript, no raw pages, no record of the
    rounds it took. The prompt is rendered fresh from the ledger every time, so it
    is the same size after three rounds as after one.
    """
    system = (DIRECT_SYSTEM if mode == "direct" else SYNTH_SYSTEM).format(
        date=date_block()
    )
    msgs = [{"role": "system", "content": system}]
    if history_text:
        msgs.append(
            {
                "role": "user",
                "content": f"Earlier in this conversation:\n{history_text}",
            }
        )
        msgs.append({"role": "assistant", "content": "Understood — I have the context."})
    if mode == "direct":
        msgs.append({"role": "user", "content": question})
    else:
        msgs.append(
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Notes from the pages that were read:\n\n{evidence_text}\n\n"
                    f"Answer the question using these notes."
                ),
            }
        )
    return msgs


# gpt-oss emits citations as 【https://…】 regardless of how firmly the prompt asks
# for markdown, so we convert rather than argue. Doing it in code also means a model
# swap can't silently reintroduce the raw brackets.
_BRACKET_CITE = re.compile(r"【\s*([^】]*?)\s*】")
_URL = re.compile(r"https?://[^\s,;、】]+")


def normalise_citations(text, allowed_urls):
    """Turn 【url】 citations into [domain](url) markdown links.

    Only URLs actually read are linked — a bracket citing anything else is dropped
    rather than rendered, so a hallucinated link can never become a clickable one.
    """
    allowed = set(allowed_urls or [])

    def repl(m):
        urls = [u.rstrip(".,;)") for u in _URL.findall(m.group(1))]
        urls = [u for u in urls if u in allowed]
        if not urls:
            return ""
        seen, links = set(), []
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            host = u.split("//", 1)[-1].split("/", 1)[0]
            links.append(f"[{host[4:] if host.startswith('www.') else host}]({u})")
        return " (" + ", ".join(links) + ")"

    text = _BRACKET_CITE.sub(repl, text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", text)


def synth_stream(messages):
    """Stream the final answer from the big model."""
    return ollama.stream(
        config.BIG_MODEL, messages, config.BIG_OPTIONS, think=config.BIG_THINK
    )
