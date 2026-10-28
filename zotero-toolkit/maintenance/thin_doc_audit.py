#!/usr/bin/env python3
"""thin_doc_audit.py — list processed documents whose stored text is a sliver of what their source PDF holds.

Two shapes turn up in a Zotero → LightRAG workspace: a header-only stub ingested before
the real PDF reached the storage folder, and a thin conversion made by an older converter
(metadata header, abstract and a watermark line per page, little else). Both are processed
documents that retrieval can barely use. This tool finds them again after every build.

Read-only: it opens the database in a read-only session and only reads the Zotero storage
folder; it never writes the database, the corpus, the hold or storage. Its only outputs are
the report on stdout and, with --out, a targets TSV for a requeue script.

For every processed document of the workspace whose stored chunk text is below
--kb-max-chars it measures the text layer of the largest PDF in `<zotero>/storage/<KEY>/`
(first 40 pages, scaled by page count, 20 s cap per document) and keeps the document when
the PDF holds more than max(--min-gain x stored chars, 20000) characters, or when the stored
body (text after the metadata header) is under 200 characters and the PDF holds at least
2000. A candidate is "arrived after ingest" when the PDF reached the disk (ctime, immune to
an old mtime from a sync) after the document was created, else "thin first conversion".

--dry-convert BUILD_CORPUS_DIR confirms candidates: it imports build_corpus.py from that
directory with every OCR server address pointed at a dead port, converts each candidate
into a throwaway directory (OCR off, text layer only) and keeps those whose body is at
least --min-dry-gain x the stored characters. Without it candidates are "unconfirmed".

Needs pymupdf; asyncpg only for the database query. Exit 0 = done, 1 = database
unreachable or query failed, 2 = bad arguments or input.
"""

import argparse
import contextlib
import datetime
import json
import os
import shutil
import sys
import tempfile
import time

OCR_GUARD = {
    "OCR_VLM_REMOTE_URL": "http://127.0.0.1:9",
    "OCR_VLM_LOCAL_URL": "http://127.0.0.1:9",
    "OCR_VLM_LOCAL_START": "",
}
PDF_PAGES = 40
PDF_SECONDS = 20.0
PDF_FLOOR = 20000
STUB_BODY = 200
STUB_PDF_MIN = 2000

SQL = """
select d.id, d.file_path, d.chunks_count, to_char(d.created_at, 'YYYY-MM-DD'),
       coalesce(string_agg(c.content, E'\\n' order by c.chunk_order_index), '')
from lightrag_doc_status d
left join lightrag_doc_chunks c on c.workspace = d.workspace and c.full_doc_id = d.id
where d.workspace = $1 and d.status = 'processed'
group by d.id, d.file_path, d.chunks_count, d.created_at
having coalesce(sum(length(c.content)), 0) < $2
order by d.file_path
"""


def load_env(path):
    """Parse a dotenv file the way LightRAG's server reads its .env."""
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


def body_of(text):
    """Text after the metadata header: the first `---` line whose preceding text has `**Zotero:**`."""
    i = text.find("\n---\n")
    while i >= 0:
        if "**Zotero:**" in text[:i]:
            return text[i + 5 :].strip()
        i = text.find("\n---\n", i + 1)
    return text.strip()


def kb_body_chars(text):
    return len(body_of(text))


def pdf_text(folder, max_pages=PDF_PAGES, seconds=PDF_SECONDS):
    """Text-layer estimate of the largest PDF in folder, or None when there is no usable PDF.

    Returns {pages, chars, ctime, mtime, path}; chars is -1 when the file will not open.
    """
    if not os.path.isdir(folder):
        return None
    pdfs = [
        (os.path.getsize(p), p)
        for p in (
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".pdf")
        )
        if os.path.isfile(p)
    ]
    if not pdfs or max(pdfs)[0] == 0:
        return None
    path = max(pdfs)[1]
    st = os.stat(path)
    info = {
        "pages": 0,
        "chars": -1,
        "ctime": st.st_ctime,
        "mtime": st.st_mtime,
        "path": path,
    }
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    try:
        doc = pymupdf.open(path)
    except Exception:
        return info
    n = doc.page_count
    chars = seen = 0
    t0 = time.monotonic()
    for i in range(min(n, max_pages)):
        try:
            chars += len(doc[i].get_text().strip())
        except Exception:
            pass
        seen += 1
        if time.monotonic() - t0 > seconds:
            break
    doc.close()
    info.update(pages=n, chars=int(chars * n / max(seen, 1)))
    return info


def day(ts):
    return datetime.datetime.fromtimestamp(ts).date().isoformat()


def classify(created, ctime):
    """'arrived after ingest' when the PDF reached the disk on a later day than the document's creation."""
    return (
        "arrived after ingest"
        if day(ctime) > str(created)[:10]
        else "thin first conversion"
    )


def measure(rows, zotero):
    """Add key, kb_chars, kb_body and pdf (see pdf_text) to each fetched row."""
    out = []
    for r in rows:
        key = r["file_path"].split("__")[0]
        out.append(
            dict(
                r,
                key=key,
                kb_chars=len(r["text"]),
                kb_body=kb_body_chars(r["text"]),
                pdf=pdf_text(os.path.join(zotero, "storage", key)),
            )
        )
    return out


def select_candidates(rows, min_gain=10, exclude=()):
    """Rows (as from measure) whose PDF text layer dwarfs the stored text; adds `class`."""
    skip = set(exclude)
    out = []
    for r in rows:
        pdf = r.get("pdf")
        if not pdf or pdf["chars"] < 0 or r["key"] in skip:
            continue
        gain = pdf["chars"] > max(min_gain * r["kb_chars"], PDF_FLOOR)
        stub = r["kb_body"] < STUB_BODY and pdf["chars"] >= STUB_PDF_MIN
        if gain or stub:
            out.append(dict(r, **{"class": classify(r["created"], pdf["ctime"])}))
    return out


def dry_convert(keys, bundle_dir, meta_path, zotero=None):
    """Body characters build_corpus.convert() would produce per key, with OCR unreachable.

    Returns {key: {"status", "body"}}. Nothing is written outside a temp dir that is
    always removed; build_corpus log output goes to stderr.
    """
    os.environ.update(
        OCR_GUARD
    )  # before the import: build_corpus reads these at import time
    sys.modules.pop("build_corpus", None)
    sys.path.insert(0, bundle_dir)
    out = tempfile.mkdtemp(prefix="thin_doc_audit_")
    res = {}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import build_corpus as bc

            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            bc.CFG.update(
                zotero=zotero or os.path.expanduser("~/Zotero"),
                out=out,
                held=None,
                deny=set(),
                force=True,
                ocr="off",
                ocr_vlm="off",
                min_chars=200,
                oom_skip=set(),
                meta=meta,
            )
            for key in keys:
                try:
                    r = bc.convert(key)
                    path = r[2]
                    text = ""
                    if path and os.path.exists(path):
                        with open(path, encoding="utf-8") as fh:
                            text = fh.read()
                    res[key] = {"status": r[1], "body": len(body_of(text))}
                except Exception as e:
                    res[key] = {"status": f"error:{type(e).__name__}", "body": 0}
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(bundle_dir)
        shutil.rmtree(out, ignore_errors=True)
    return res


def fetch_rows(workspace, kb_max_chars, env=None):
    """Processed documents with < kb_max_chars stored chunk text, in a read-only session."""
    import asyncio

    import asyncpg

    env = env or {}

    def setting(name, default=None):
        v = os.environ.get(name)
        if v is None:
            v = env.get(name)
        return default if v in (None, "") else v

    async def run():
        con = await asyncpg.connect(
            host=setting("POSTGRES_HOST", "localhost"),
            port=int(setting("POSTGRES_PORT", "5432")),
            user=setting("POSTGRES_USER", "postgres"),
            password=setting("POSTGRES_PASSWORD"),
            database=setting("POSTGRES_DATABASE", "postgres"),
            timeout=15,
            server_settings={
                "default_transaction_read_only": "on",
                "statement_timeout": "300000",
            },
        )
        try:
            return await con.fetch(SQL, workspace, kb_max_chars)
        finally:
            await con.close()

    return [
        {
            "doc_id": r[0],
            "file_path": r[1],
            "chunks": int(r[2] or 0),
            "created": r[3] or "",
            "text": r[4] or "",
        }
        for r in asyncio.run(run())
    ]


def reason(c, min_dry_gain=None):
    dry = c.get("dry")
    if c.get("class") == "arrived after ingest":
        s = f"stub ingested before the full PDF arrived (KB {c['kb_chars']} chars)"
    else:
        s = f"thin conversion (KB {c['kb_chars']} chars)"
    if dry is not None:
        s += f"; dry convert gives {dry['body']}"
    else:
        s += "; unconfirmed"
    return s


def write_targets(path, candidates, how="", thresholds=""):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# generated {time.strftime('%Y-%m-%d %H:%M:%S')} by thin_doc_audit.py"
            f" ({how or 'no dry convert'}; {thresholds})\n"
        )
        f.write("# key\treason\tchunks\tfile_path\n")
        for c in candidates:
            f.write(f"{c['key']}\t{reason(c)}\t{c['chunks']}\t{c['file_path']}\n")


def print_report(cands, dry_run):
    cols = "key kb_chars/body chunks created pdf_pages/chars pdf_ctime class" + (
        " dry_body" if dry_run else ""
    )
    print("\t".join(cols.split()))
    for c in cands:
        p = c["pdf"]
        row = [
            c["key"],
            f"{c['kb_chars']}/{c['kb_body']}",
            str(c["chunks"]),
            str(c["created"]),
            f"{p['pages']}/{p['chars']}",
            day(p["ctime"]),
            c["class"],
        ]
        if dry_run:
            row.append(str(c["dry"]["body"]) if c.get("dry") else "-")
        print("\t".join(row))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--env-file",
        help="dotenv file with POSTGRES_* settings; the process environment wins",
    )
    ap.add_argument(
        "--workspace",
        help="default: POSTGRES_WORKSPACE, else WORKSPACE, else 'default'",
    )
    ap.add_argument(
        "--zotero",
        default=os.path.expanduser("~/Zotero"),
        help="Zotero data dir (contains storage/)",
    )
    ap.add_argument(
        "--kb-max-chars",
        type=int,
        default=4000,
        help="only documents with less stored text (default %(default)s)",
    )
    ap.add_argument(
        "--min-gain",
        type=float,
        default=10,
        help="PDF chars must exceed this x stored chars (default %(default)s)",
    )
    ap.add_argument(
        "--min-dry-gain",
        type=float,
        default=5,
        help="with --dry-convert: dry body must reach this x stored chars (default %(default)s)",
    )
    ap.add_argument(
        "--dry-convert",
        metavar="BUILD_CORPUS_DIR",
        help="directory holding build_corpus.py; confirms candidates by converting them without OCR",
    )
    ap.add_argument(
        "--meta",
        help="zotero_metadata.json for --dry-convert (default: beside build_corpus.py)",
    )
    ap.add_argument("--exclude", default="", help="comma-separated keys to drop")
    ap.add_argument("--out", help="write a requeue-targets TSV here")
    a = ap.parse_args(argv)

    try:
        env = load_env(a.env_file) if a.env_file else {}
    except OSError as e:
        print(f"thin_doc_audit: ERROR cannot read --env-file: {e}", file=sys.stderr)
        return 2
    ws = (
        a.workspace
        or os.environ.get("POSTGRES_WORKSPACE")
        or env.get("POSTGRES_WORKSPACE")
        or os.environ.get("WORKSPACE")
        or env.get("WORKSPACE")
        or "default"
    )
    try:
        rows = fetch_rows(ws, a.kb_max_chars, env)
    except Exception as e:
        print(
            f"thin_doc_audit: ERROR database unreachable or query failed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1
    exclude = [k.strip() for k in a.exclude.split(",") if k.strip()]
    cands = select_candidates(measure(rows, a.zotero), a.min_gain, exclude)
    how = "unconfirmed, no dry convert"
    if a.dry_convert:
        meta = a.meta or os.path.join(a.dry_convert, "zotero_metadata.json")
        try:
            res = dry_convert([c["key"] for c in cands], a.dry_convert, meta, a.zotero)
        except (OSError, ImportError, ValueError) as e:
            print(
                f"thin_doc_audit: ERROR dry convert failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return 2
        for c in cands:
            c["dry"] = res.get(c["key"])
        cands = [
            c
            for c in cands
            if c["dry"] and c["dry"]["body"] >= a.min_dry_gain * c["kb_chars"]
        ]
        how = "confirmed by dry convert"
    thresholds = f"kb-max-chars {a.kb_max_chars}, min-gain {a.min_gain:g}, min-dry-gain {a.min_dry_gain:g}"
    print(
        f"thin_doc_audit: workspace {ws!r}: {len(rows)} short document(s), {len(cands)} candidate(s) ({how})"
    )
    print_report(cands, bool(a.dry_convert))
    if a.out:
        write_targets(a.out, cands, how, thresholds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
