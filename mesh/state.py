#!/usr/bin/env python3
"""
state — the evidence ledger. This is the heart of the redesign.

The old loop used the **chat transcript as its memory**: every search result and
every fetched page was appended to `messages`, so round 5's prompt contained all
of rounds 1-4. Context grew without bound, decoding slowed down with it, and the
model — swimming in nav bars and cookie banners — kept losing the thread and
searching again. Bloated context was not a side effect of over-searching; it was
the *cause* of it.

Here, memory is a **structured object** instead. The ledger accumulates facts;
every model call renders a *fresh, bounded* prompt from it. Round 5's prompt is
the same size as round 2's. That single change turns O(n^2) into O(n), and it is
what makes extra rounds cheap enough to be worth having.

Two invariants keep it honest:

  1. `render()` never exceeds `max_chars`. Not "usually" — the budget is enforced
     by construction, so no sequence of rounds can overflow the window.
  2. Nothing raw ever lands here. A Note is what the small model *distilled* from
     a page (~700 chars), never the page itself (~8000).
"""

from dataclasses import dataclass, field
from urllib.parse import urlparse

# How much of the big model's window the evidence block may occupy. At ~4 chars
# per token this is roughly 3k tokens, leaving plenty of room in a 16k window for
# the system prompt, the question, past turns, and a long answer.
DEFAULT_EVIDENCE_BUDGET = 12000

# Past (question, answer) pairs retained across turns, and how much of each answer.
MAX_HISTORY_TURNS = 6
HISTORY_ANSWER_CHARS = 700


def domain_of(url):
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


# eq=False so Notes compare by identity. Two pages can legitimately distil to the
# same text, and value-equality would make `in chosen` / `.index()` below confuse
# them for each other.
@dataclass(eq=False)
class Note:
    """One page, after the small model has read it. Never the page itself."""

    url: str
    title: str
    text: str
    round: int = 0

    @property
    def domain(self):
        return domain_of(self.url)

    def render(self):
        return f"[{self.domain}] {self.title}\n{self.text}\nSource: {self.url}"


@dataclass
class Ledger:
    question: str
    notes: list = field(default_factory=list)
    queries_tried: set = field(default_factory=set)
    seen_urls: set = field(default_factory=set)
    gaps: list = field(default_factory=list)
    rounds: int = 0

    # --- accumulating -------------------------------------------------------

    def record_query(self, query):
        """True if this query is new. Stops the loop re-running the same search."""
        key = " ".join(query.lower().split())
        if key in self.queries_tried:
            return False
        self.queries_tried.add(key)
        return True

    def add_note(self, url, title, text):
        self.notes.append(
            Note(url=url, title=title, text=text.strip(), round=self.rounds)
        )

    def filter_unseen(self, urls, per_domain=2):
        """Drop URLs we've already handled, and cap how many come from one site.

        The per-domain cap is a quality control, not just politeness: eight pages
        from one domain is one source wearing eight hats, and it makes an answer
        look well-corroborated when it isn't.
        """
        counts = {}
        for n in self.notes:
            counts[n.domain] = counts.get(n.domain, 0) + 1
        out = []
        for u in urls:
            if not u or u in self.seen_urls:
                continue
            d = domain_of(u)
            if counts.get(d, 0) >= per_domain:
                continue
            counts[d] = counts.get(d, 0) + 1
            out.append(u)
        return out

    def mark_seen(self, url):
        """Record that a URL was attempted, so a failed fetch is never retried."""
        self.seen_urls.add(url)

    # --- reporting ----------------------------------------------------------

    @property
    def domains(self):
        return {n.domain for n in self.notes if n.domain}

    def __len__(self):
        return len(self.notes)

    # --- rendering (the bounded part) ---------------------------------------

    def render(self, max_chars=DEFAULT_EVIDENCE_BUDGET):
        """The evidence block handed to the big model. Never exceeds max_chars.

        When the budget binds we keep the *newest* notes and spread across domains,
        because later rounds exist precisely to fill gaps the earlier ones left.
        """
        if not self.notes:
            return "(no sources read yet)"

        chosen, used, per_domain = [], 0, {}
        # Two passes: first take at most one note per domain (guarantees breadth
        # even under a tight budget), then backfill with the rest.
        for limit in (1, 99):
            for n in reversed(self.notes):
                if n in chosen:
                    continue
                if per_domain.get(n.domain, 0) >= limit:
                    continue
                block = n.render()
                if used + len(block) + 2 > max_chars:
                    continue
                chosen.append(n)
                used += len(block) + 2
                per_domain[n.domain] = per_domain.get(n.domain, 0) + 1

        # Restore reading order so citation numbering is stable across rounds.
        chosen.sort(key=lambda n: self.notes.index(n))
        return "\n\n".join(n.render() for n in chosen)

    def sources(self):
        """Unique URLs actually distilled, in reading order."""
        seen, out = set(), []
        for n in self.notes:
            if n.url not in seen:
                seen.add(n.url)
                out.append(n.url)
        return out


def render_history(turns):
    """Bounded rendering of past conversation turns.

    Same discipline as the ledger: old answers are truncated rather than carried
    whole, so a long session cannot slowly refill the context window.
    """
    if not turns:
        return ""
    kept = turns[-MAX_HISTORY_TURNS:]
    parts = []
    for q, a in kept:
        a = (a or "").strip()
        if len(a) > HISTORY_ANSWER_CHARS:
            a = a[:HISTORY_ANSWER_CHARS].rstrip() + " …"
        parts.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(parts)
