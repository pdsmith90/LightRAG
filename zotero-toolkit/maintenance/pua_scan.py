#!/usr/bin/env python3
"""pua_scan.py — list documents whose stored text is glyph codes (Unicode Private Use Area).

A scanned PDF whose fonts map every glyph into the Private Use Area (U+E000-F8FF and
planes 15-16) passes every text extractor as "text": the corpus builder's empty-text
check never sends it to OCR, and the ingest turns it into thousands of chunks of glyph
codes that extract nothing and keep the pipeline busy for hours. corpus/build_corpus.py
(looks_garbled -> OCR) and corpus/quarantine_junk.py (the glyph rule) stop new arrivals;
this is the audit for documents that were ingested before those checks existed.

Read-only. One pass over lightrag_doc_full of the workspace (a regular expression per
row; about 15 s for 3 GB of text), joined to lightrag_doc_status for the status, chunk
count and length. A document is reported when PUA code points / content length is at
least --min-ratio (default 0.30; garbled scans score 0.70-0.95, genuine documents with a
symbol font stay under 0.05). Exit 0 = none, 1 = documents reported, 2 = database error.

To remove a reported document, use LightRAG's own delete path (DELETE
/documents/delete_document with its id; it rebuilds the entities the document shared),
then add its storage key to the corpus denylist so the next build does not regenerate it,
or leave it off the denylist and remove its parsed output so the build re-converts it
through OCR.

Configuration follows clean_dangling_refs.py: command-line flag, then the process
environment, then --env-file (LightRAG's own .env works unchanged).
"""

import argparse
import asyncio
import os
import sys

import asyncpg

ENV = {}

SQL = r"""
SELECT s.file_path, s.status, s.chunks_count, s.content_length,
       length(regexp_replace(f.content, U&'[^\E000-\F8FF]', '', 'g')) AS pua
FROM lightrag_doc_full f
JOIN lightrag_doc_status s ON s.id = f.id AND s.workspace = f.workspace
WHERE f.workspace = $1 AND f.content ~ U&'[\E000-\F8FF]'
"""


def load_env(path):
    """Parse a dotenv file the way LightRAG's server reads its .env (python-dotenv)."""
    env = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            elif " #" in v:
                v = v.split(" #", 1)[0].rstrip()
            env[k.strip()] = v
    return env


def setting(name, default=None):
    """Process environment first, then --env-file (LightRAG loads .env with override=False)."""
    v = os.environ.get(name)
    if v is None:
        v = ENV.get(name)
    return default if v in (None, "") else v


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="List documents whose stored text is Private Use Area glyph codes."
    )
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=0.30,
        help="report a document when PUA code points / content length is at least this "
        "(default: %(default)s)",
    )
    ap.add_argument(
        "--env-file",
        help="dotenv file to read settings from, e.g. LightRAG's .env; "
        "the process environment takes precedence",
    )
    ap.add_argument(
        "--workspace",
        help="default: POSTGRES_WORKSPACE, else WORKSPACE, else 'default'",
    )
    return ap.parse_args(argv)


def configure(a):
    global ENV
    ENV = load_env(a.env_file) if a.env_file else {}
    return (
        a.workspace
        or setting("POSTGRES_WORKSPACE")
        or setting("WORKSPACE")
        or "default"
    )


async def connect():
    return await asyncpg.connect(
        host=setting("POSTGRES_HOST", "localhost"),
        port=int(setting("POSTGRES_PORT", "5432")),
        user=setting("POSTGRES_USER", "postgres"),
        password=setting("POSTGRES_PASSWORD"),
        database=setting("POSTGRES_DATABASE", "postgres"),
    )


def classify(rows, min_ratio):
    """(ratio, file_path, status, chunks, length, pua) for every row at or above the bar."""
    hits = []
    for r in rows:
        ratio = (r["pua"] or 0) / max(r["content_length"] or 0, 1)
        if ratio >= min_ratio:
            hits.append(
                (
                    ratio,
                    r["file_path"],
                    r["status"],
                    r["chunks_count"],
                    r["content_length"],
                    r["pua"],
                )
            )
    hits.sort(reverse=True)
    return hits


async def scan(workspace, min_ratio):
    con = await connect()
    try:
        rows = await con.fetch(SQL, workspace)
    finally:
        await con.close()
    hits = classify(rows, min_ratio)
    for ratio, fp, status, chunks, length, pua in hits:
        print(
            f"pua_scan: WARN {ratio:5.1%} PUA  {chunks:>6} chunks  {length:>9} chars  "
            f"[{status}]  {fp}"
        )
    print(
        f"pua_scan: {len(hits)} glyph-code document(s) at >= {min_ratio:.0%} "
        f"({len(rows)} documents contain any PUA code point) in workspace {workspace!r}"
    )
    return 1 if hits else 0


def main(argv=None):
    a = parse_args(argv)
    workspace = configure(a)
    try:
        return asyncio.run(scan(workspace, a.min_ratio))
    except (asyncpg.PostgresError, OSError) as e:
        print(f"pua_scan: ERROR {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
