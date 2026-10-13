#!/usr/bin/env python3
"""
history — a locked-away session recorder for the research web UI.

Every finished research turn (question, answer, sources, model, timing) is
appended to a local SQLite database. It stays out of the page you work in: the
web UI shows none of it until you ask, with a shortcut (Alt+H), for a read-only
browser over it — kong, 2026-10-13, reversing the July "terminal only" rule while
keeping its point, a focused page. That browser reads through browse()/fetch()
below on a read-only connection; deleting stays terminal-only. From the terminal:

    python3 history.py list [N]        # the last N turns, one line each (default 20)
    python3 history.py show <id>       # one full record: question, answer, sources
    python3 history.py search <term>   # find past turns by question/answer text
    python3 history.py stats           # where the DB lives, size, counts, perms
    python3 history.py purge --days N  # delete records older than N days
    python3 history.py purge --all     # wipe the whole history

Security posture (single local user, stdlib only):
  - DB lives OUTSIDE the repo and outside the served static/ dir, under
    ~/.local/share/research-web/ — it can't be committed or served by accident.
  - Directory is chmod 0700 and the DB file 0600: only your user can read it.
  - The web app writes; it reads only through a read-only connection, behind a
    Host check and a custom header another site's script cannot send. Purging is
    never offered over HTTP — only this CLI, run as your user, deletes.
  - RESEARCH_HISTORY=0 disables recording entirely.
  - Recording is best-effort: any failure here is swallowed so it can never
    take down a research run.
"""

import json
import os
import sqlite3
import sys
import time

ENABLED = os.environ.get("RESEARCH_HISTORY", "1") != "0"
DATA_DIR = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "research-web")
DB_PATH = os.environ.get("RESEARCH_HISTORY_DB",
                         os.path.join(DATA_DIR, "history.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT    NOT NULL,   -- ISO-8601 local time, e.g. 2026-07-15 14:03:22
    sid      TEXT    NOT NULL,   -- browser session id (groups a conversation)
    model    TEXT    NOT NULL,
    depth    INTEGER NOT NULL,   -- min_sources gate the user asked for (0 = casual)
    question TEXT    NOT NULL,
    answer   TEXT    NOT NULL,   -- empty if the run was aborted mid-stream
    sources  TEXT    NOT NULL,   -- JSON array of URLs actually fetched
    seconds  REAL    NOT NULL,   -- wall time of the turn
    ok       INTEGER NOT NULL    -- 1 = a final answer was produced
);
CREATE INDEX IF NOT EXISTS turns_ts  ON turns(ts);
CREATE INDEX IF NOT EXISTS turns_sid ON turns(sid);
"""


def _connect():
    """Open (creating if needed) the DB with owner-only permissions."""
    os.makedirs(DATA_DIR, mode=0o700, exist_ok=True)
    os.chmod(DATA_DIR, 0o700)  # tighten even if the dir pre-existed looser
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.executescript(_SCHEMA)
    # sqlite creates the file with the process umask; clamp it (and any WAL/SHM
    # siblings) down to owner-only after the fact.
    for p in (DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm", DB_PATH + "-journal"):
        try:
            os.chmod(p, 0o600)
        except FileNotFoundError:
            pass
    return conn


def record(sid, model, depth, question, answer, sources, seconds):
    """Append one finished turn. Never raises — history must not break research."""
    if not ENABLED:
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = _connect()
        try:
            with conn:  # commit/rollback; close below so the server never leaks handles
                conn.execute(
                    "INSERT INTO turns (ts, sid, model, depth, question, answer,"
                    " sources, seconds, ok) VALUES (?,?,?,?,?,?,?,?,?)",
                    (ts, sid, model, int(depth), question, answer or "",
                     json.dumps(list(sources)), round(float(seconds), 1),
                     1 if answer else 0))
        finally:
            conn.close()
    except Exception as e:
        # Log to the server console and move on; the answer already streamed.
        print(f"history: failed to record turn: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


# ======================= web reader (read-only) ========================
# For the history window (Alt+H). A `mode=ro` connection: even a bug in a caller
# cannot write or delete through it, and a missing DB reads as empty instead of
# being created by a read.

def _connect_ro():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)


def _snip(text, width):
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[:width - 1] + "…"


def browse(term="", before=None, limit=50):
    """Newest turns first, optionally matching `term` in question or answer.
    `before` pages backwards by id. -> {"items": [...], "more": bool}"""
    conn = _connect_ro()
    if conn is None:
        return {"items": [], "more": False}
    limit = max(1, min(int(limit), 200))
    where, args = ["1=1"], []
    if term:
        where.append("(question LIKE ? OR answer LIKE ?)")
        args += [f"%{term}%"] * 2
    if before:
        where.append("id < ?")
        args.append(int(before))
    try:
        rows = conn.execute(
            "SELECT id, ts, model, question, answer, ok, seconds FROM turns WHERE "
            + " AND ".join(where) + " ORDER BY id DESC LIMIT ?", args + [limit + 1]).fetchall()
    finally:
        conn.close()
    items = [{"id": r[0], "ts": r[1], "model": r[2], "q": _snip(r[3], 200),
              "a": _snip(r[4], 220), "ok": bool(r[5]), "seconds": r[6]} for r in rows[:limit]]
    return {"items": items, "more": len(rows) > limit}


def fetch(id_):
    """One full turn, plus the other turns of its conversation (same sid) for context."""
    conn = _connect_ro()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT id, ts, sid, model, question, answer, sources, seconds, ok"
            " FROM turns WHERE id = ?", (int(id_),)).fetchone()
        if not row:
            return None
        thread = conn.execute(
            "SELECT id, question FROM turns WHERE sid = ? ORDER BY id LIMIT 60",
            (row[2],)).fetchall()
    finally:
        conn.close()
    try:
        sources = json.loads(row[6])
    except ValueError:
        sources = []
    return {"id": row[0], "ts": row[1], "model": row[3], "question": row[4],
            "answer": row[5], "sources": sources, "seconds": row[7], "ok": bool(row[8]),
            "thread": [{"id": t[0], "q": _snip(t[1], 90)} for t in thread]}


# ======================= terminal reader (CLI) =========================
# Kept in the same file so there is exactly one place that knows the schema.

def _dim(s):
    return f"\033[2m{s}\033[0m" if sys.stdout.isatty() else s


def _bold(s):
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


def _one_line(text, width=100):
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 1] + "…"


def _cli_list(limit):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, ts, model, depth, question, answer, ok FROM turns"
            " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    if not rows:
        print("history is empty")
        return
    for id_, ts, model, depth, q, a, ok in rows:
        flag = "" if ok else "  [aborted]"
        gate = f" depth={depth}" if depth else ""
        print(f"{_bold(f'#{id_}')}  {_dim(ts)}  {_dim(model + gate)}{flag}")
        print(f"  Q: {_one_line(q)}")
        if a:
            print(f"  A: {_dim(_one_line(a))}")
    print(_dim(f"\n({len(rows)} shown — `show <id>` for full text)"))


def _cli_show(id_):
    with _connect() as conn:
        row = conn.execute(
            "SELECT ts, sid, model, depth, question, answer, sources, seconds,"
            " ok FROM turns WHERE id = ?", (id_,)).fetchone()
    if not row:
        print(f"no record #{id_}")
        return
    ts, sid, model, depth, q, a, sources, seconds, ok = row
    print(f"{_bold(f'#{id_}')}  {ts}  ·  {model}  ·  session {sid}"
          f"  ·  {seconds}s" + ("" if ok else "  ·  ABORTED"))
    if depth:
        print(_dim(f"depth gate: {depth} sources required"))
    print(f"\n{_bold('QUESTION')}\n{q}\n\n{_bold('ANSWER')}\n{a or '(none — run aborted)'}")
    urls = json.loads(sources)
    if urls:
        print(f"\n{_bold('SOURCES FETCHED')}")
        for u in urls:
            print(f"  - {u}")


def _cli_search(term):
    like = f"%{term}%"
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, ts, question FROM turns WHERE question LIKE ? OR"
            " answer LIKE ? ORDER BY id DESC LIMIT 50", (like, like)).fetchall()
    if not rows:
        print(f"no match for {term!r}")
        return
    for id_, ts, q in rows:
        print(f"{_bold(f'#{id_}')}  {_dim(ts)}  {_one_line(q)}")


def _cli_stats():
    with _connect() as conn:
        n, first, last = conn.execute(
            "SELECT COUNT(*), MIN(ts), MAX(ts) FROM turns").fetchone()
    size = os.path.getsize(DB_PATH)
    mode = oct(os.stat(DB_PATH).st_mode & 0o777)
    print(f"db:       {DB_PATH}")
    print(f"perms:    {mode} (dir {oct(os.stat(DATA_DIR).st_mode & 0o777)})")
    print(f"records:  {n}")
    if n:
        print(f"range:    {first}  →  {last}")
    print(f"size:     {size / 1024:.1f} KiB")
    print(f"recording {'DISABLED (RESEARCH_HISTORY=0)' if not ENABLED else 'enabled'}")


def _cli_purge(argv):
    conn = _connect()
    if argv[:1] == ["--all"]:
        n = conn.execute("DELETE FROM turns").rowcount
    elif argv[:1] == ["--days"] and len(argv) > 1:
        cutoff = time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(time.time() - float(argv[1]) * 86400))
        n = conn.execute("DELETE FROM turns WHERE ts < ?", (cutoff,)).rowcount
    else:
        print("usage: history.py purge --days N | --all")
        return
    conn.commit()
    conn.execute("VACUUM")  # actually shrink the file so purged text is gone
    conn.close()
    print(f"deleted {n} record(s)")


def main(argv):
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        _cli_list(int(argv[1]) if len(argv) > 1 else 20)
    elif cmd == "show" and len(argv) > 1:
        _cli_show(int(argv[1]))
    elif cmd == "search" and len(argv) > 1:
        _cli_search(" ".join(argv[1:]))
    elif cmd == "stats":
        _cli_stats()
    elif cmd == "purge":
        _cli_purge(argv[1:])
    else:
        print(__doc__.strip())


if __name__ == "__main__":
    main(sys.argv[1:])
