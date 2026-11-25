#!/usr/bin/env python3
"""
s2_citations_sync.py - keep a local superset of the Semantic Scholar citation
dataset: the rows whose citing OR cited paper is in the library.

The Datasets API (key required, sent as the header "x-api-key" to
api.semanticscholar.org only) lists pre-signed S3 URLs. Those URLs are fetched
WITHOUT the key. A full pass streams every shard (HTTP -> gzip -> lines, never
written to disk whole), prefilters each line with a byte regex on the two corpus
ids and keeps the rows that touch the library. Later runs apply the weekly diffs.

Usage:
  python3 s2_citations_sync.py --store s2store/citations.sqlite --measure
  python3 s2_citations_sync.py --store s2store/citations.sqlite \\
      --library-ids out/library_corpus_ids.txt --full [--parallel 3]
  python3 s2_citations_sync.py --store ... --library-ids ... --diff [--backfill-new-ids]

The store is SQLite (WAL): citations (one row per citationid), meta
(release_id, mode, library_corpus_ids_hash, updated_at, and full_complete),
shards (a full pass resumes per shard) and library_ids (the ids the filter was
built from, so a grown library can be told apart from a changed one).

Exit status: 0 on success; 1 on a failure (a shard that failed after its
retries, an unusable diff, a missing key).
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import s2_api

DATASETS = f"https://{s2_api.HOST}/datasets/v1"
TRIES = 3  # per shard or diff file
CHUNK = 5000  # rows per executemany

SCHEMA = """
CREATE TABLE IF NOT EXISTS citations (
    citationid INTEGER PRIMARY KEY,
    citing INTEGER,
    cited INTEGER,
    isinfluential INTEGER,
    contexts TEXT,
    intents TEXT
);
CREATE INDEX IF NOT EXISTS citations_citing_idx ON citations (citing);
CREATE INDEX IF NOT EXISTS citations_cited_idx ON citations (cited);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS shards (
    release_id TEXT,
    url_basename TEXT,
    rows_seen INTEGER,
    rows_kept INTEGER,
    done_at TEXT,
    PRIMARY KEY (release_id, url_basename)
);
CREATE TABLE IF NOT EXISTS library_ids (corpusid INTEGER PRIMARY KEY);
"""

# A corpus id is null when S2 could not resolve that end of the citation.
PAIR_RE = re.compile(
    rb'"citingcorpusid"\s*:\s*(\d+|null)\s*,\s*"citedcorpusid"\s*:\s*(\d+|null)'
)
CITING_RE = re.compile(rb'"citingcorpusid"\s*:\s*(\d+|null)')
CITED_RE = re.compile(rb'"citedcorpusid"\s*:\s*(\d+|null)')


class SyncError(Exception):
    """A failure that stops the run (the message carries no URL and no key)."""


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


def open_store(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path, timeout=600, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def get_meta(con) -> dict:
    return dict(con.execute("SELECT key, value FROM meta").fetchall())


def _set_meta(con, **kv) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [(k, str(v)) for k, v in kv.items()],
    )


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(str(i) for i in sorted(ids)).encode()).hexdigest()[
        :16
    ]


def read_ids(path: str) -> set[int]:
    with open(path, encoding="utf-8") as f:
        return {int(x) for x in f.read().split()}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def commit_batch(con, rows, deletes=(), shard=None, meta=None, library_ids=()) -> None:
    """One transaction: delete, upsert (``rows`` is an iterable of 6-tuples), record
    the shard (release_id, basename, seen, kept), update meta, add library ids."""
    con.execute("BEGIN IMMEDIATE")
    try:
        deletes = list(deletes)
        for i in range(0, len(deletes), CHUNK):
            con.executemany(
                "DELETE FROM citations WHERE citationid = ?",
                [(d,) for d in deletes[i : i + CHUNK]],
            )
        buf = []
        for r in rows:
            buf.append(r)
            if len(buf) >= CHUNK:
                con.executemany(
                    "INSERT OR REPLACE INTO citations VALUES (?,?,?,?,?,?)", buf
                )
                buf = []
        if buf:
            con.executemany(
                "INSERT OR REPLACE INTO citations VALUES (?,?,?,?,?,?)", buf
            )
        if shard:
            con.execute(
                "INSERT OR REPLACE INTO shards VALUES (?,?,?,?,?)", (*shard, _now())
            )
        if library_ids:
            con.executemany(
                "INSERT OR IGNORE INTO library_ids VALUES (?)",
                [(i,) for i in sorted(library_ids)],
            )
        if meta:
            _set_meta(con, updated_at=_now(), **meta)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


# --------------------------------------------------------------------------
# reading a stream of records
# --------------------------------------------------------------------------


def _id(v):
    return None if v is None else int(v)


def _row(rec: dict) -> tuple:
    ctx, ints = rec.get("contexts"), rec.get("intents")
    return (
        int(rec["citationid"]),
        _id(rec["citingcorpusid"]),
        _id(rec["citedcorpusid"]),
        1 if rec.get("isinfluential") else 0,
        None if ctx is None else json.dumps(ctx, ensure_ascii=False),
        None if ints is None else json.dumps(ints, ensure_ascii=False),
    )


def scan_lines(lines, wanted, sink) -> tuple[int, int]:
    """Feed ``sink`` the row of every record whose citing or cited corpus id is in
    ``wanted``. The ids are found with a byte regex first; only the lines that
    match are parsed as JSON. Returns (lines seen, rows kept)."""
    seen = kept = 0
    for line in lines:
        if not line.strip():
            continue
        seen += 1
        m = PAIR_RE.search(line)
        if m:
            a, b = m.group(1), m.group(2)
        else:
            m1, m2 = CITING_RE.search(line), CITED_RE.search(line)
            if not (m1 and m2):
                raise SyncError(
                    f"record {seen} has no citingcorpusid/citedcorpusid; stopping"
                )
            a, b = m1.group(1), m2.group(1)
        a = None if a == b"null" else int(a)
        b = None if b == b"null" else int(b)
        if a in wanted or b in wanted:
            sink(_row(json.loads(line)))
            kept += 1
    return seen, kept


def scan_deletes(lines) -> list[int]:
    """The citationids of a delete file. A record without one stops the run: no
    other key is guessed."""
    out = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        cid = json.loads(line).get("citationid")
        if not isinstance(cid, int) or isinstance(cid, bool):
            raise SyncError(
                f"delete record {n} has no citationid; stopping (not guessing another key)"
            )
        out.append(cid)
    return out


def _gz_lines(stream):
    return gzip.GzipFile(fileobj=stream)


def open_stream(url: str):
    """Binary file object of a pre-signed S3 URL. No key is sent (s2_api.s2_headers
    adds it only for api.semanticscholar.org, and none is passed here anyway)."""
    req = urllib.request.Request(url, headers=s2_api.s2_headers(url, ""))
    return urllib.request.urlopen(req, timeout=120)


def head_size(url: str):
    """Size in bytes of a pre-signed file, or None. A pre-signed URL is signed for
    one method, so when HEAD is refused a one-byte ranged GET reads Content-Range."""
    headers = s2_api.s2_headers(url, "")
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers, method="HEAD"), timeout=60
        ) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except urllib.error.HTTPError:
        pass
    try:
        req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            m = re.search(r"/(\d+)$", r.headers.get("Content-Range") or "")
            return int(m.group(1)) if m else None
    except (urllib.error.URLError, OSError):
        return None


def process_shard(
    store_path, release_id, basename, stream, wanted, spool_dir=None
) -> dict:
    """Read one shard stream, spool the kept rows to a temporary file, then commit
    them and the shard record in ONE transaction. A shard that fails before the
    commit leaves the store untouched."""
    spool_dir = spool_dir or os.path.dirname(os.path.abspath(store_path))
    os.makedirs(spool_dir, exist_ok=True)
    fd, spool = tempfile.mkstemp(suffix=".spool", dir=spool_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            seen, kept = scan_lines(
                _gz_lines(stream), wanted, lambda row: out.write(json.dumps(row) + "\n")
            )

        def rows():
            with open(spool, encoding="utf-8") as f:
                for line in f:
                    yield tuple(json.loads(line))

        con = open_store(store_path)
        try:
            commit_batch(con, rows(), shard=(release_id, basename, seen, kept))
        finally:
            con.close()
    finally:
        os.unlink(spool)
    return {"basename": basename, "rows_seen": seen, "rows_kept": kept}


# --------------------------------------------------------------------------
# the Datasets API
# --------------------------------------------------------------------------


def basename(url: str) -> str:
    return urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]


class Api:
    """Datasets API calls through one S2Client (one pacer). Needs the key."""

    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()

    def get(self, path: str):
        with self.lock:
            try:
                data = self.client.request("GET", f"{DATASETS}{path}")
            except s2_api.S2Error as e:
                raise SyncError(f"Datasets API {path.split('/')[1]}: {e}") from None
            if self.client.key_rejected:
                raise SyncError("the Datasets API rejected the API key")
        return data

    def latest(self) -> str:
        return self.get("/release/latest")["release_id"]

    def files(self, release: str) -> list[str]:
        return list(self.get(f"/release/{release}/dataset/citations")["files"])

    def diffs(self, frm: str, to: str) -> list[dict]:
        return list(self.get(f"/diffs/{frm}/to/{to}/citations")["diffs"])


# --------------------------------------------------------------------------
# full pass
# --------------------------------------------------------------------------


def worker_cmd() -> list[str]:
    """The shard worker's command line: under nice 19 and ionice idle when present."""
    cmd = []
    if shutil.which("nice"):
        cmd += ["nice", "-n", "19"]
    if shutil.which("ionice"):
        cmd += ["ionice", "-c3"]
    return cmd + [sys.executable, os.path.abspath(__file__), "--worker"]


def spawn_worker(job: dict):
    """Run one shard in a subprocess. The job (with the pre-signed URL) goes in on
    stdin, never on the command line. Returns (ok, info dict or error text)."""
    p = subprocess.run(
        worker_cmd(), input=json.dumps(job), capture_output=True, text=True
    )
    if p.returncode == 0:
        try:
            return True, json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return True, {}
    return False, (p.stderr.strip().splitlines() or ["worker failed"])[-1][:300]


def worker_main() -> int:
    job = json.load(sys.stdin)
    try:
        wanted = read_ids(job["ids_file"])
        with open_stream(job["url"]) as stream:
            info = process_shard(
                job["store"], job["release_id"], job["basename"], stream, wanted
            )
    except Exception as e:  # the message never holds the URL (urllib's errors do not)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(info))
    return 0


def stream_shards(
    api, release, store_path, ids_file, parallel, spawn, done_key
) -> dict:
    """Stream every shard of ``release`` not yet recorded under ``done_key`` (each
    in its own process, ``parallel`` at a time). A failed shard is retried from the
    start, up to TRIES times, with a fresh URL listing first. Returns
    {"done", "failed": [basenames], "rows_kept"}."""
    con = open_store(store_path)
    try:
        finished = {
            r[0]
            for r in con.execute(
                "SELECT url_basename FROM shards WHERE release_id = ?", (done_key,)
            )
        }
    finally:
        con.close()
    urls = {basename(u): u for u in api.files(release)}
    pending = sorted(b for b in urls if b not in finished)
    print(
        f"{len(urls)} shards, {len(finished & set(urls))} already done, {len(pending)} to stream"
    )

    def one(b):
        for attempt in range(TRIES):
            if attempt:
                try:
                    urls.update({basename(u): u for u in api.files(release)})
                except SyncError as e:
                    print(
                        f"shard {b}: cannot refresh the URL listing: {e}",
                        file=sys.stderr,
                    )
            ok, info = spawn(
                {
                    "store": store_path,
                    "release_id": done_key,
                    "basename": b,
                    "url": urls[b],
                    "ids_file": ids_file,
                }
            )
            if ok:
                print(
                    f"shard {b}: {info.get('rows_seen', '?')} rows seen, {info.get('rows_kept', '?')} kept"
                )
                return b, info
            print(f"shard {b}: attempt {attempt + 1} failed: {info}", file=sys.stderr)
        return b, None

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        results = list(ex.map(one, pending))
    return {
        "done": [b for b, i in results if i is not None],
        "failed": [b for b, i in results if i is None],
        "rows_kept": sum((i or {}).get("rows_kept", 0) for _, i in results),
    }


def run_full(api, store_path, ids_file, parallel=3, spawn=spawn_worker) -> int:
    wanted = read_ids(ids_file)
    h = ids_hash(wanted)
    release = api.latest()
    con = open_store(store_path)
    try:
        meta = get_meta(con)
        if meta.get("release_id") and meta["release_id"] != release:
            raise SyncError(
                f"the store holds release {meta['release_id']} and the latest is {release}: "
                "use --diff to move forward, or start a new store for a full pass"
            )
        if meta.get("full_complete") == "1":
            print(f"the full pass of release {release} is already complete")
            return 0
        if meta.get("library_corpus_ids_hash") not in (None, h):
            raise SyncError(
                "the library ids changed since this full pass started; remove the store and start again"
            )
        _set_meta(
            con,
            release_id=release,
            mode="full",
            library_corpus_ids_hash=h,
            full_complete=0,
            updated_at=_now(),
        )
    finally:
        con.close()
    res = stream_shards(api, release, store_path, ids_file, parallel, spawn, release)
    if res["failed"]:
        print(
            f"{len(res['failed'])} shard(s) failed after {TRIES} tries: {', '.join(res['failed'])}",
            file=sys.stderr,
        )
        return 1
    con = open_store(store_path)
    try:
        commit_batch(con, (), meta={"full_complete": 1}, library_ids=wanted)
    finally:
        con.close()
    print(
        f"full pass of release {release} complete: {res['rows_kept']} rows kept in this run"
    )
    return 0


# --------------------------------------------------------------------------
# diffs
# --------------------------------------------------------------------------


def _read_with_tries(url_for, fn):
    """fn(gzip stream) with TRIES attempts from the start; ``url_for(attempt)`` gives
    the URL (a fresh listing after the first failure)."""
    for attempt in range(TRIES):
        try:
            with open_stream(url_for(attempt)) as stream:
                return fn(_gz_lines(stream))
        except SyncError:
            raise
        except Exception as e:
            if attempt == TRIES - 1:
                raise SyncError(
                    f"a diff file failed after {TRIES} tries ({type(e).__name__})"
                ) from None
            print(
                f"diff file: attempt {attempt + 1} failed ({type(e).__name__}); retrying",
                file=sys.stderr,
            )


def apply_diff(api, con, store_path, d, wanted) -> dict:
    """Apply one diff in one transaction: its deletes, its upserts (filtered by
    ``wanted``), and the release advance."""
    frm, to = d["from_release"], d["to_release"]

    def fresh(kind, i):
        def url_for(attempt):
            if attempt == 0:
                return d[kind][i]
            for x in api.diffs(frm, to):
                if x["from_release"] == frm and x["to_release"] == to:
                    return x[kind][i]
            raise SyncError(f"diff {frm} -> {to} is gone from the listing")

        return url_for

    spool_dir = os.path.dirname(os.path.abspath(store_path))
    fd, spool = tempfile.mkstemp(suffix=".spool", dir=spool_dir)
    deletes, seen, kept = [], 0, 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for i in range(len(d.get("delete_files") or [])):
                deletes += _read_with_tries(fresh("delete_files", i), scan_deletes)
            for i in range(len(d.get("update_files") or [])):
                out.flush()
                start = out.tell()

                def into_spool(lines):
                    out.flush()
                    out.seek(
                        start
                    )  # a retry starts the file again: drop what the failed try wrote
                    out.truncate()
                    return scan_lines(
                        lines, wanted, lambda row: out.write(json.dumps(row) + "\n")
                    )

                s, k = _read_with_tries(fresh("update_files", i), into_spool)
                seen, kept = seen + s, kept + k

        def rows():
            with open(spool, encoding="utf-8") as f:
                for line in f:
                    yield tuple(json.loads(line))

        commit_batch(
            con, rows(), deletes=deletes, meta={"release_id": to, "mode": "diff"}
        )
    finally:
        os.unlink(spool)
    return {
        "from": frm,
        "to": to,
        "deleted_ids": len(deletes),
        "rows_seen": seen,
        "rows_kept": kept,
    }


def run_diff(api, store_path, ids_file) -> int:
    wanted = read_ids(ids_file)
    con = open_store(store_path)
    try:
        meta = get_meta(con)
        if meta.get("full_complete") != "1":
            raise SyncError("the store has no complete full pass: run --full first")
        have = {r[0] for r in con.execute("SELECT corpusid FROM library_ids")}
        new = wanted - have
        print(f"{len(new)} library corpus id(s) are not in the store's filter")
        if new:
            print(
                f"WARNING: {len(new)} new library id(s): rows touching only them are in the store only if "
                "kept for another reason; --backfill-new-ids re-streams the release for them",
                file=sys.stderr,
            )
        frm, to = meta["release_id"], api.latest()
        if frm == to:
            print(f"the store is at the latest release ({to})")
            return 0
        chain = api.diffs(frm, to)
        pos = frm
        for d in chain:
            if d["from_release"] != pos:
                raise SyncError(f"the diff chain {frm} -> {to} has a gap at {pos}")
            pos = d["to_release"]
        if pos != to or not chain:
            raise SyncError(f"the diff chain {frm} -> {to} does not reach {to}")
        for d in chain:
            r = apply_diff(api, con, store_path, d, wanted)
            print(
                f"diff {r['from']} -> {r['to']}: {r['deleted_ids']} deletes, {r['rows_seen']} update rows "
                f"seen, {r['rows_kept']} kept"
            )
    finally:
        con.close()
    return 0


def run_backfill(api, store_path, ids_file, parallel=3, spawn=spawn_worker) -> int:
    """Re-stream the store's release for the library ids its filter lacks."""
    wanted = read_ids(ids_file)
    con = open_store(store_path)
    try:
        meta = get_meta(con)
        if meta.get("full_complete") != "1":
            raise SyncError("the store has no complete full pass: run --full first")
        new = sorted(
            wanted - {r[0] for r in con.execute("SELECT corpusid FROM library_ids")}
        )
    finally:
        con.close()
    if not new:
        print("nothing to backfill: every library id is in the store's filter")
        return 0
    release = meta["release_id"]
    key = f"{release}+backfill-{ids_hash(new)[:12]}"
    fd, tmp = tempfile.mkstemp(
        suffix=".ids", dir=os.path.dirname(os.path.abspath(store_path))
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write("".join(f"{i}\n" for i in new))
        print(f"backfilling {len(new)} new id(s) from release {release}")
        res = stream_shards(api, release, store_path, tmp, parallel, spawn, key)
    finally:
        os.unlink(tmp)
    if res["failed"]:
        print(
            f"{len(res['failed'])} shard(s) failed after {TRIES} tries; rerun to resume",
            file=sys.stderr,
        )
        return 1
    con = open_store(store_path)
    try:
        commit_batch(
            con, (), meta={"library_corpus_ids_hash": ids_hash(wanted)}, library_ids=new
        )
    finally:
        con.close()
    print(f"backfill complete: {res['rows_kept']} rows kept")
    return 0


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------


def run_measure(api, store_path, head=head_size) -> dict:
    """Print the sizes of the shard files (and of the diffs from the store's release,
    when it has one). Downloads nothing."""
    release = api.latest()
    sizes = {basename(u): head(u) for u in api.files(release)}
    total = sum(v or 0 for v in sizes.values())
    print(f"release {release}: {len(sizes)} shard files")
    for b, n in sorted(sizes.items()):
        print(f"  {b}: {'unknown' if n is None else f'{n / 2**30:.2f} GiB'}")
    print(
        f"  total: {total / 2**30:.1f} GiB ({sum(1 for v in sizes.values() if v is None)} unknown)"
    )
    out = {"release": release, "shards": sizes, "diffs": None}
    stored = None
    if store_path and os.path.exists(store_path):
        con = open_store(store_path)
        try:
            stored = get_meta(con).get("release_id")
        finally:
            con.close()
    if stored and stored != release:
        diff_sizes = {}
        for d in api.diffs(stored, release):
            for kind in ("update_files", "delete_files"):
                for u in d.get(kind) or []:
                    diff_sizes[
                        f"{d['from_release']}->{d['to_release']} {kind} {basename(u)}"
                    ] = head(u)
        print(
            f"diffs {stored} -> {release}: {len(diff_sizes)} files, {sum(v or 0 for v in diff_sizes.values()) / 2**20:.0f} MiB"
        )
        out["diffs"] = diff_sizes
    else:
        print("no stored release older than the latest: diff sizes not measured")
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None, client=None, spawn=spawn_worker) -> int:
    ap = argparse.ArgumentParser(
        description="Sync the library's slice of the S2 citation dataset."
    )
    ap.add_argument("--store", help="SQLite store (s2store/citations.sqlite)")
    ap.add_argument(
        "--library-ids",
        help="file of library corpus ids, one per line (out/library_corpus_ids.txt)",
    )
    ap.add_argument(
        "--measure", action="store_true", help="list file sizes; download nothing"
    )
    ap.add_argument("--full", action="store_true", help="stream the whole release once")
    ap.add_argument(
        "--diff",
        action="store_true",
        help="apply the diffs from the stored release to the latest",
    )
    ap.add_argument(
        "--backfill-new-ids",
        action="store_true",
        help="re-stream the release for ids new to the library",
    )
    ap.add_argument(
        "--parallel", type=int, default=3, help="shards streamed at once (default 3)"
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        return worker_main()
    modes = [a.measure, a.full, a.diff]
    if sum(modes) > 1 or not (any(modes) or a.backfill_new_ids):
        ap.error(
            "give one of --measure, --full, --diff (--backfill-new-ids may follow --diff or stand alone)"
        )
    if a.full and a.backfill_new_ids:
        ap.error("--backfill-new-ids goes with --diff or stands alone")
    if not a.store:
        ap.error("--store is required")
    if not a.measure and not a.library_ids:
        ap.error("--library-ids is required")
    key = os.environ.get("S2_API_KEY", "")
    if not key and client is None:
        print(
            "The Datasets API needs a key: set S2_API_KEY (sent only as the x-api-key header).",
            file=sys.stderr,
        )
        return 1
    api = Api(client or s2_api.S2Client(key))
    try:
        if a.measure:
            run_measure(api, a.store)
            return 0
        if a.full:
            return run_full(api, a.store, a.library_ids, a.parallel, spawn)
        status = run_diff(api, a.store, a.library_ids) if a.diff else 0
        if a.backfill_new_ids and status == 0:
            status = run_backfill(api, a.store, a.library_ids, a.parallel, spawn)
        return status
    except SyncError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
