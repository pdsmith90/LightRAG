#!/usr/bin/env python3
"""
s2_citation_sweep.py - citation contexts from the public Semantic Scholar Graph
API (no key), for the builder's S2 store.

For each cited library paper, GET /paper/{paperId}/citations is paged (limit 1000,
following ``next``) and the rows whose citing paper is in the library are kept.
They are written to a SQLite store with the same schema as s2_citations_sync.py,
so build_struct_layer.read_s2_store reads it unchanged (release_id "graph-api",
mode "api"). A citationid is a negative surrogate, -(citing * 2**31 + cited), so it
never collides with a Datasets API citationid.

No key is ever read or sent. Requests go through s2_api.S2Client (one pacer: 2 s,
slowing x1.5 per throttle up to 10 s); a 429 or 5xx backs off exponentially from
4 s, doubling to 300 s, up to 8 retries.

Usage:
  python3 s2_citation_sweep.py --out out --store s2store/api_citations.sqlite
  python3 s2_citation_sweep.py --store s2store/api_citations.sqlite --report

``library_seen`` (sweep-only) records when each library corpus id first appeared in L,
and never changes it: a swept paper is due again when one of its in-library citers
(cites.csv) first appeared after the paper was fetched.

Each cited paper is one transaction (rows deleted and inserted, ``swept`` row
upserted): a paper that fails midway leaves the store as it was and is recorded as
``failed``. Resume skips papers swept ``ok`` / ``truncated`` / ``notfound`` within
--max-age-days that have no newer citer (--refresh ignores both).

Exit status: 0 on success or a clean budget stop (the summary says "complete":
false), 1 on unusable input, 3 after 10 failed papers in a row ("throttled; resume
later"), everything already committed is kept.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import math
import os
import re
import signal
import sqlite3
import sys
import time
import urllib.error

import s2_api
import s2_citations_sync as sync

FIELDS = "corpusId,contexts,intents,contextsWithIntent,isInfluential"
LIMIT = 1000
RETRIES = 8
BACKOFF_MAX = 300.0
MAX_FAILED_IN_A_ROW = 10
PROGRESS_EVERY = 50
MAX_AGE_DAYS = 180  # was 30 (operator decision 2026-11-25): a new library citer still triggers a re-sweep at once
TRUNC_RATIO = 0.9  # fewer rows than this share of s2_citation_count: next stopped early
DONE = ("ok", "truncated", "notfound")  # statuses that resume skips

SWEPT_DDL = """
CREATE TABLE IF NOT EXISTS swept (
    cited INTEGER PRIMARY KEY,
    s2_paper_id TEXT,
    fetched_at TEXT,
    rows_seen INTEGER,
    rows_kept INTEGER,
    pages INTEGER,
    truncated INTEGER,
    status TEXT
);
CREATE TABLE IF NOT EXISTS library_seen (
    corpusid INTEGER PRIMARY KEY,
    first_seen TEXT
);
"""
REFUSED = re.compile(r"^HTTP 4\d\d$")  # a non-retried 4xx (an offset the API refuses)


class Throttled(Exception):
    """10 papers failed in a row."""


# --------------------------------------------------------------------------
# inputs and targets
# --------------------------------------------------------------------------


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def select_targets(
    papers: list[dict], cited_ids, citers=None
) -> tuple[set, list[dict], dict]:
    """(L, targets, info). L = every non-empty s2_corpus_id. A target has an
    s2_paper_id, s2_citation_count > 0 and a corpus id, and is either cited by a
    library paper (``cited_ids``: cited_paper_id of cites.csv) or has no openalex_id.
    Ordered by s2_citation_count ascending, then paper_id. ``citers`` ({cited
    paper_id: [citing paper_id]}, from cites.csv) fills each target's "citers": the
    sorted corpus ids of its in-library citers."""
    library, targets = set(), []
    info = {
        "papers": len(papers),
        "no_s2_id": 0,
        "zero_citations": 0,
        "no_corpus_id": 0,
        "not_needed": 0,
    }
    cited_ids = set(cited_ids)
    citers = citers or {}
    corpus = {p["paper_id"]: _int(p.get("s2_corpus_id")) for p in papers}
    for p in papers:
        cid = _int(p.get("s2_corpus_id"))
        if cid is not None:
            library.add(cid)
        pid = (p.get("s2_paper_id") or "").strip()
        count = _int(p.get("s2_citation_count")) or 0
        if not pid:
            info["no_s2_id"] += 1
        elif count <= 0:
            info["zero_citations"] += 1
        elif cid is None:
            info["no_corpus_id"] += 1
        elif p["paper_id"] not in cited_ids and (p.get("openalex_id") or "").strip():
            info["not_needed"] += 1
        else:
            ids = {corpus.get(c) for c in citers.get(p["paper_id"], ())} - {None, cid}
            targets.append(
                {
                    "paper_id": p["paper_id"],
                    "s2_paper_id": pid,
                    "corpus_id": cid,
                    "count": count,
                    "citers": sorted(ids),
                }
            )
    targets.sort(key=lambda t: (t["count"], t["paper_id"]))
    info["targets"] = len(targets)
    return library, targets, info


def load_inputs(out: str):
    papers = read_csv(os.path.join(out, "papers.csv"))
    citers = collections.defaultdict(list)
    for r in read_csv(os.path.join(out, "cites.csv")):
        citers[r["cited_paper_id"]].append(r["citing_paper_id"])
    return select_targets(papers, set(citers), citers)


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------


def surrogate_id(citing: int, cited: int) -> int:
    """Negative, deterministic; corpus ids are below 2**31, so it cannot collide
    with a (positive) dataset citationid."""
    return -(citing * 2**31 + cited)


def _strs(x) -> list:
    return [s for s in x if isinstance(s, str)] if isinstance(x, list) else []


def convert(rec, cited: int, library) -> dict | None:
    """One Graph API citation record -> {citing, infl, contexts, intents, kind}, or
    None when the citer is unknown, outside the library or the cited paper itself.
    ``kind`` is "cwi" (built from contextsWithIntent), "flat" (contexts only, empty
    intents) or "none" (no context at all)."""
    if not isinstance(rec, dict):
        return None
    citing = _int((rec.get("citingPaper") or {}).get("corpusId"))
    if citing is None or citing == cited or citing not in library:
        return None
    cwi = rec.get("contextsWithIntent")
    contexts, intents = [], []
    if isinstance(cwi, list) and cwi:
        kind = "cwi"
        for item in cwi:
            if isinstance(item, dict) and isinstance(item.get("context"), str):
                contexts.append(item["context"])
                intents.append(_strs(item.get("intents")))
    else:
        contexts = _strs(rec.get("contexts"))
        intents = [
            [] for _ in contexts
        ]  # the flat intents list is not aligned with contexts
        kind = "flat" if contexts else "none"
    if kind == "cwi" and not contexts:
        kind = "none"
    return {
        "citing": citing,
        "infl": 1 if rec.get("isInfluential") else 0,
        "contexts": contexts,
        "intents": intents,
        "kind": kind,
    }


def _merge(a: dict, b: dict) -> dict:
    """The same citer twice in one answer: one row, contexts concatenated."""
    a["infl"] = a["infl"] or b["infl"]
    a["contexts"] += b["contexts"]
    a["intents"] += b["intents"]
    return a


def read_paper(client, target: dict, library) -> dict:
    """Page through the citations of one target. Returns {"rows": {citing: row},
    "seen", "pages", "truncated", "kinds"}. Raises NotFound (first page), S2Error or
    BudgetExhausted; the store is not touched here."""
    path = f"/paper/{target['s2_paper_id']}/citations"
    rows, kinds = {}, collections.Counter()
    seen = pages = offset = 0
    truncated = False
    while True:
        try:
            resp = client.get(
                path, {"fields": FIELDS, "limit": LIMIT, "offset": offset}
            )
        except s2_api.NotFound:
            if pages == 0:
                raise
            truncated = True
            break
        except s2_api.S2Error as e:
            if pages > 0 and REFUSED.match(str(e)):  # the API refuses a deeper offset
                truncated = True
                break
            raise
        data = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data, list):
            raise s2_api.S2Error("answer without a data list")
        pages += 1
        for rec in data:
            seen += 1
            row = convert(rec, target["corpus_id"], library)
            if row is None:
                continue
            kinds[row["kind"]] += 1
            rows[row["citing"]] = (
                _merge(rows[row["citing"]], row) if row["citing"] in rows else row
            )
        nxt = resp.get("next")
        if nxt is None:
            break
        if not isinstance(nxt, int) or nxt <= offset:
            truncated = True
            break
        offset = nxt
    if seen < TRUNC_RATIO * target["count"]:
        truncated = True
    return {
        "rows": rows,
        "seen": seen,
        "pages": pages,
        "truncated": truncated,
        "kinds": kinds,
    }


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


def _now() -> str:
    return sync._now()


def open_store(path: str) -> sqlite3.Connection:
    con = sync.open_store(path)
    con.executescript(SWEPT_DDL)
    return con


def init_store(con, library) -> None:
    """Meta, library_ids (replaced by this run's L) and library_seen (ids new to it get
    first_seen = now; existing rows never change) in one transaction."""
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute("DELETE FROM library_ids")
        con.executemany(
            "INSERT INTO library_ids VALUES (?)", [(i,) for i in sorted(library)]
        )
        con.executemany(
            "INSERT OR IGNORE INTO library_seen VALUES (?, ?)",
            [(i, _now()) for i in sorted(library)],
        )
        sync._set_meta(con, release_id="graph-api", mode="api", updated_at=_now())
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def commit_paper(con, target: dict, res: dict) -> int:
    """One transaction: this paper's rows replaced, its ``swept`` row upserted.
    Returns the rows kept."""
    cited = target["corpus_id"]
    rows = [
        (
            surrogate_id(r["citing"], cited),
            r["citing"],
            cited,
            r["infl"],
            json.dumps(r["contexts"], ensure_ascii=False),
            json.dumps(r["intents"], ensure_ascii=False),
        )
        for r in res["rows"].values()
    ]
    now = _now()
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(
            "DELETE FROM citations WHERE cited = ? AND citationid < 0", (cited,)
        )
        con.executemany("INSERT OR REPLACE INTO citations VALUES (?,?,?,?,?,?)", rows)
        con.execute(
            "INSERT OR REPLACE INTO swept VALUES (?,?,?,?,?,?,?,?)",
            (
                cited,
                target["s2_paper_id"],
                now,
                res["seen"],
                len(rows),
                res["pages"],
                int(res["truncated"]),
                res["status"],
            ),
        )
        sync._set_meta(con, updated_at=now)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def mark_failed(con, target: dict) -> None:
    """Record a failed paper; the paper's rows and counts stay as they were."""
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(
            "INSERT INTO swept VALUES (?,?,?,0,0,0,0,'failed') ON CONFLICT(cited) DO UPDATE SET "
            "fetched_at = excluded.fetched_at, status = 'failed'",
            (target["corpus_id"], target["s2_paper_id"], _now()),
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def _fresh(fetched_at, now: datetime.datetime, max_age_days: float) -> bool:
    try:
        then = datetime.datetime.strptime(fetched_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc
        )
    except (TypeError, ValueError):
        return False
    return (now - then).total_seconds() <= max_age_days * 86400


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def counting_opener(opener, stats: dict):
    """Wrap the client's opener to count 429 answers (the client does not)."""
    opener = opener or s2_api._urlopen_json

    def call(method, url, headers, body):
        try:
            return opener(method, url, headers, body)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                stats["429"] += 1
            raise

    return call


def _eta(todo_targets, window) -> str:
    """Remaining calls (pages) x the recent seconds per call."""
    if len(window) < 2 or window[-1][1] <= window[0][1]:
        return "?"
    secs_per_call = (window[-1][0] - window[0][0]) / (window[-1][1] - window[0][1])
    left_calls = sum(max(1, math.ceil(t["count"] / LIMIT)) for t in todo_targets)
    s = int(left_calls * secs_per_call)
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def due_reason(t, status, fetched, seen, now, max_age_days, refresh=False):
    """Why target ``t`` is swept in this run, or None when it is fresh: "refresh",
    "never", "failed", "age" (older than max_age_days, or not a finished status) or
    "new_citer" (an in-library citer in cites.csv first appeared after ``fetched``;
    ``seen`` is {corpusid: first_seen}). ISO UTC strings compare in time order."""
    if refresh:
        return "refresh"
    if status is None:
        return "never"
    if status == "failed":
        return "failed"
    if status not in DONE or not _fresh(fetched, now, max_age_days):
        return "age"
    if any((seen.get(c) or "") > fetched for c in t.get("citers", ())):
        return "new_citer"
    return None


def read_seen(con) -> dict:
    try:
        return dict(
            con.execute("SELECT corpusid, first_seen FROM library_seen").fetchall()
        )
    except sqlite3.OperationalError:  # a store from before library_seen
        return {}


def run_sweep(
    con,
    client,
    targets,
    library,
    *,
    stats=None,
    refresh=False,
    max_age_days=MAX_AGE_DAYS,
    max_hours=None,
    clock=time.monotonic,
    now=None,
    stop=None,
    log=None,
) -> dict:
    """Sweep ``targets`` (see select_targets). ``stop`` is a dict whose "flag" a
    signal handler sets; it ends the sweep after the current paper's commit.
    Returns the summary (``exit``: 0, or 3 after 10 failed papers in a row)."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    stats = stats if stats is not None else collections.Counter()
    stop = stop if stop is not None else {"flag": False}
    now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
    known = {
        r[0]: (r[1], r[2])
        for r in con.execute("SELECT cited, status, fetched_at FROM swept")
    }
    t0 = clock()
    sm = collections.Counter()
    sm["targets"] = len(targets)
    kinds = collections.Counter()
    in_a_row = 0
    stopped = None
    window = collections.deque(maxlen=PROGRESS_EVERY + 1)
    window.append((t0, client.calls))
    seen = read_seen(con)
    todo, due = [], collections.Counter()
    for t in targets:
        status, fetched = known.get(t["corpus_id"], (None, None))
        why = due_reason(t, status, fetched, seen, now(), max_age_days, refresh)
        if why:
            todo.append(t)
            due[why] += 1
    sm["skipped"] = len(targets) - len(todo)
    for i, t in enumerate(todo):
        if stop["flag"]:
            stopped = "sigterm"
            break
        if max_hours is not None and clock() - t0 >= max_hours * 3600:
            stopped = "max-hours"
            break
        try:
            res = read_paper(client, t, library)
            res["status"] = "truncated" if res["truncated"] else "ok"
        except s2_api.BudgetExhausted:
            stopped = "max-calls"
            break
        except s2_api.NotFound:
            res = {
                "rows": {},
                "seen": 0,
                "pages": 1,
                "truncated": False,
                "kinds": {},
                "status": "notfound",
            }
        except s2_api.S2Error as e:
            mark_failed(con, t)
            sm["failed"] += 1
            in_a_row += 1
            log(f"WARN: {t['paper_id']} failed: {e}")
            if in_a_row >= MAX_FAILED_IN_A_ROW:
                stopped = "throttled"
                break
            continue
        in_a_row = 0
        sm["rows_kept"] += commit_paper(con, t, res)
        sm[res["status"]] += 1
        kinds.update(res["kinds"])
        done = sm["ok"] + sm["truncated"] + sm["notfound"] + sm["failed"]
        if done % PROGRESS_EVERY == 0:
            window.append((clock(), client.calls))
            log(
                f"sweep: {done} done / {len(todo) - i - 1} left, calls {client.calls}, "
                f"429s {stats['429']}, pace {client.pacer.interval:.1f}s, rows kept {sm['rows_kept']}, "
                f"ETA {_eta(todo[i + 1 :], window)}"
            )
    summary = {
        "complete": stopped is None and sm["failed"] == 0,
        "stopped": stopped,
        "targets": sm["targets"],
        "swept_ok": sm["ok"],
        "swept_truncated": sm["truncated"],
        "swept_notfound": sm["notfound"],
        "failed": sm["failed"],
        "skipped_fresh": sm["skipped"],
        "due": dict(due),
        "rows_kept": sm["rows_kept"],
        "calls": client.calls,
        "throttled_429": stats["429"],
        "rows_from_contextsWithIntent": kinds["cwi"],
        "rows_from_flat_contexts": kinds["flat"],
        "rows_without_context": kinds["none"],
        "exit": 3 if stopped == "throttled" else 0,
    }
    return summary


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def report(path: str, targets=None, max_age_days=MAX_AGE_DAYS, now=None) -> dict:
    """``targets`` (from load_inputs), when given, adds "due": how many are due by reason."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        due = None
        if targets is not None:
            known = {
                r[0]: (r[1], r[2])
                for r in con.execute("SELECT cited, status, fetched_at FROM swept")
            }
            seen = read_seen(con)
            now = now or datetime.datetime.now(datetime.timezone.utc)
            due = collections.Counter(
                why
                for t in targets
                if (
                    why := due_reason(
                        t,
                        *known.get(t["corpus_id"], (None, None)),
                        seen,
                        now,
                        max_age_days,
                    )
                )
            )
            due = {k: due.get(k, 0) for k in ("never", "failed", "age", "new_citer")}
        by_status = dict(
            con.execute("SELECT status, COUNT(*) FROM swept GROUP BY status").fetchall()
        )
        rows = con.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
        pairs = con.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM citations GROUP BY citing, cited)"
        ).fetchone()[0]
        with_ctx = con.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM citations GROUP BY citing, cited "
            "HAVING SUM(contexts IS NOT NULL AND contexts <> '[]') > 0)"
        ).fetchone()[0]
        listing = {
            s: [
                r[0]
                for r in con.execute(
                    "SELECT cited FROM swept WHERE status = ? ORDER BY cited", (s,)
                )
            ]
            for s in ("truncated", "failed")
        }
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    finally:
        con.close()
    return {
        "release_id": meta.get("release_id"),
        "updated_at": meta.get("updated_at"),
        "papers_by_status": by_status,
        "rows": rows,
        "pairs": pairs,
        "pairs_with_context": with_ctx,
        "truncated_papers": len(listing["truncated"]),
        "truncated_cited": listing["truncated"],
        "failed_papers": len(listing["failed"]),
        "failed_cited": listing["failed"],
        "targets_due": due,
    }


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def main(argv=None, opener=None, clock=time.monotonic, sleep=None) -> int:
    ap = argparse.ArgumentParser(
        description="Citation contexts from the public Semantic Scholar Graph API."
    )
    ap.add_argument(
        "--out", default="out", help="the builder's output dir (papers.csv, cites.csv)"
    )
    ap.add_argument("--store", default="s2store/api_citations.sqlite")
    ap.add_argument("--max-age-days", type=float, default=MAX_AGE_DAYS)
    ap.add_argument("--refresh", action="store_true", help="ignore --max-age-days")
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--max-hours", type=float)
    ap.add_argument(
        "--report",
        action="store_true",
        help="print a JSON summary of the store and exit",
    )
    args = ap.parse_args(argv)

    if args.report:
        if not os.path.exists(args.store):
            print(f"ERROR: no store at {args.store}", file=sys.stderr)
            return 1
        try:
            targets = load_inputs(args.out)[1]
        except (
            OSError,
            KeyError,
        ):  # no builder output to compare with: no "due" counts
            targets = None
        print(json.dumps(report(args.store, targets, args.max_age_days), indent=2))
        return 0
    try:
        library, targets, info = load_inputs(args.out)
    except (OSError, KeyError) as e:
        print(
            f"ERROR: cannot read {args.out}/papers.csv and cites.csv: {type(e).__name__}",
            file=sys.stderr,
        )
        return 1
    print(f"sweep: {info}", file=sys.stderr)

    stats = collections.Counter()
    client = s2_api.S2Client(
        "",
        opener=counting_opener(opener, stats),
        clock=clock,
        sleep=sleep,
        max_calls=args.max_calls,
        retries=RETRIES,
        backoff0=s2_api.BACKOFF0,
        backoff_max=BACKOFF_MAX,
    )
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    con = open_store(args.store)
    try:
        init_store(con, library)
        summary = run_sweep(
            con,
            client,
            targets,
            library,
            stats=stats,
            refresh=args.refresh,
            max_age_days=args.max_age_days,
            max_hours=args.max_hours,
            clock=clock,
            stop=stop,
        )
    finally:
        con.close()
    code = summary.pop("exit")
    print(json.dumps(summary, indent=2))
    if code == 3:
        print("throttled; resume later", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
