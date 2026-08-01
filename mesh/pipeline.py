#!/usr/bin/env python3
"""
pipeline — resolve, triage, then read one page at a time until the budget runs out.

Shape of a turn:

    resolve()      follow-up -> standalone question        assist, ~1s
    route()        how much does this deserve, and queries  assist, ~2s
      |
      +-- "direct" ------------------------------> answer, no web
      |
      +-- otherwise, until the page budget is spent:
              search when the queue of candidates runs dry   (network)
              fetch the next candidate                       (network)
              read it -> one note                            PRIMARY, ~10s
              every few pages, assess: enough? what's missing?
      |
      synthesise from the ledger                             PRIMARY, streamed

**One page is one step.** An earlier version batched eight pages per "round" and
read three per call, copying how cloud platforms fan out. That was wrong for this
machine: there is one GPU and Ollama serialises, so a wide step is not faster — it
just becomes a long opaque stall, and coarser control over when to stop. Narrow
steps give steady visible progress, a natural place to re-decide after every page,
and a budget that means something concrete.

The budget is **pages read**, not rounds. That is the thing the user actually cares
about ("how much should it search and read"), it maps directly to time, and it does
not need explaining.
"""

import os
import time

from . import config, events, fetch, roles, search, state

# Effort levels: how many pages to read. "auto" defers to the router.
EFFORT = {
    "light": 3,
    "normal": 6,
    "thorough": 12,
    "exhaustive": 20,
}
# What the router's own verdict is worth in pages, when effort is "auto".
ROUTER_PAGES = {"quick": 3, "deep": 8}

# Re-assess after this many pages. Often enough to change course, rarely enough
# not to spend the whole turn deliberating about whether to keep going.
ASSESS_EVERY = 3

# Wall-clock ceiling on RESEARCH — checked between pages, never mid-page. Writing
# the answer still follows (~70s on a large ledger), so this sits below the real
# limit: 420 + writing keeps the worst case under the 9 minutes kong asked for.
MAX_RESEARCH_SECONDS = int(os.environ.get("MESH_MAX_SECONDS", "420"))


def _stopped(should_stop):
    return bool(should_stop and should_stop())


def research(question, history=None, effort="auto", should_stop=None):
    """Run one question end to end, yielding events.

    history: list of (question, answer) pairs from earlier in the session.
    effort:  "auto" | "light" | "normal" | "thorough" | "exhaustive"
    should_stop: zero-arg predicate polled between pages — the "Answer now" button.
    """
    t0 = time.monotonic()
    history_text = state.render_history(history or [])

    # --- resolve ------------------------------------------------------------
    # Before anything else, because everything else depends on it.
    asked = question
    if history_text:
        question = roles.resolve(question, history_text)
        if question != asked:
            yield events.notice(f"reading that as: {question}")

    ledger = state.Ledger(question=question)

    # --- triage -------------------------------------------------------------
    yield events.phase("triage", "deciding how much this needs")
    decision = roles.route(question, history_text)
    mode = decision["mode"]
    yield events.phase("triage", f"{mode} — {decision['reason']}")

    if mode == "direct" and effort == "auto":
        yield events.step(1)
        yield events.notice("answered from knowledge — no search needed")
        yield from _synthesise(question, ledger, history_text, "direct", t0)
        return

    budget = EFFORT.get(effort) or ROUTER_PAGES.get(mode, 6)
    yield events.notice(
        f"{'you asked for' if effort != 'auto' else 'planning'} "
        f"up to {budget} page{'' if budget == 1 else 's'}"
    )

    queries = decision["queries"] or [question]
    candidates = []          # search results not yet fetched
    snippets = {}            # url -> search snippet, for the blocked-page fallback
    step = 0
    since_assess = 0

    while step < budget:
        if _stopped(should_stop):
            yield events.notice("stopping early — writing up what we have")
            break
        if time.monotonic() - t0 > MAX_RESEARCH_SECONDS:
            yield events.notice(
                f"{time.monotonic() - t0:.0f}s spent — at the time limit, "
                f"writing up {len(ledger)} source(s)"
            )
            break

        # --- restock the queue when it runs dry -----------------------------
        if not candidates:
            fresh = [q for q in queries if ledger.record_query(q)]
            if not fresh:
                yield events.notice("no new angles left to search")
                break
            for q in fresh:
                yield events.tool_call("web_search", q)
            try:
                results = search.search_many(fresh)
            except search.SearchError as e:
                yield events.observation("web_search", 0, False, str(e))
                break

            for r in results:
                for u in fetch.prepare([r["url"]]):
                    snippets[u] = r.get("snippet", "")
            unseen = [r for r in results
                      if (p := fetch.prepare([r["url"]])) and p[0] not in ledger.seen_urls]
            # qwen screens the list so the expensive reader never opens a shop page.
            picked = roles.screen(question, unseen, budget * 2)
            dropped = len(unseen) - len(picked)
            candidates = [u for r in (picked + [x for x in unseen if x not in picked])
                          for u in fetch.prepare([r["url"]])]
            candidates = ledger.filter_unseen(candidates)
            yield events.observation(
                "web_search", len(results), True,
                f"{len(results)} results"
                + (f", skipped {dropped} weak" if dropped > 0 else "")
                + f", {len(candidates)} to read",
            )
            if not candidates:
                yield events.notice("nothing new worth reading")
                break

        # --- one page, one step --------------------------------------------
        url = candidates.pop(0)
        step += 1
        yield events.step(step)
        yield events.phase("reading", f"page {step} of up to {budget}")
        yield events.tool_call("web_fetch", url)

        page = fetch.fetch_one(url)
        ledger.mark_seen(url)
        if not page["ok"]:
            yield events.observation(
                "fetch", 0, False,
                f"{state.domain_of(url)}: {page.get('error', 'failed')}",
            )
            snip = (snippets.get(url) or "").strip()
            if len(snip) >= 80:
                ledger.add_note(
                    url, page.get("title", ""),
                    f"(page could not be read; search-result summary only, so "
                    f"weaker evidence) {snip}",
                )
                yield events.source(url)
            step -= 1        # a page we could not read should not spend the budget
            continue

        note = roles.read_page(question, page)
        if not note:
            yield events.observation(
                "read", 0, False, f"{state.domain_of(url)}: nothing relevant"
            )
            continue
        ledger.add_note(url, page.get("title", ""), note)
        yield events.note(url, page.get("title", ""), note)
        yield events.source(url)
        since_assess += 1

        # --- re-decide, every few pages -------------------------------------
        if since_assess >= ASSESS_EVERY and step < budget:
            since_assess = 0
            yield events.phase("checking", "is this enough to answer?")
            verdict = roles.assess(question, ledger.render(), ledger.queries_tried)
            if verdict["sufficient"]:
                yield events.notice(
                    f"enough to answer after {step} page(s) — writing"
                )
                break
            if verdict["queries"]:
                yield events.notice(f"gap: {verdict['missing'] or 'needs more'}")
                queries = verdict["queries"]
                candidates = []      # prefer the new angle over the stale queue

    if not len(ledger):
        yield events.notice(
            "no usable sources — answering from knowledge, so treat anything "
            "time-sensitive with suspicion"
        )
        yield from _synthesise(question, ledger, history_text, "direct", t0)
        return

    yield from _synthesise(question, ledger, history_text, mode, t0)


def _synthesise(question, ledger, history_text, mode, t0):
    """The one big-model call. Streams the answer, then a final event."""
    yield events.phase(
        "writing",
        f"{len(ledger)} sources from {len(ledger.domains)} sites"
        if len(ledger) else "from knowledge",
    )
    msgs = roles.synth_messages(question, ledger.render(), history_text, mode)
    buf = []
    # The synthesiser reasons at high effort — thousands of characters of
    # deliberation nobody wants scrolling past between the last source and the
    # answer. The research trace above is the part worth watching; this is the wait.
    yield events.notice("preparing the answer…")
    try:
        for channel, text in roles.synth_stream(msgs):
            if channel == "thinking":
                continue  # deliberately dropped — see note above
            buf.append(text)
            yield events.answer_chunk(text)
    except Exception as e:
        yield events.error(f"synthesis failed: {e}")
        yield events.done()
        return

    answer = roles.normalise_citations("".join(buf).strip(), ledger.sources())
    yield events.final(answer)
    yield events.notice(
        f"{time.monotonic() - t0:.0f}s · {len(ledger)} sources · "
        f"1 call to {config.PRIMARY_MODEL}"
    )
    yield events.done()
