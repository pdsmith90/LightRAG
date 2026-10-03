#!/usr/bin/env python3
"""textlayer_audit.py — list corpus documents whose markdown kept far less text than the PDF's own text layer.

A conversion that lost most of the text layer is the signature of pages pymupdf4llm
dropped: scans whose OCR text sits under the page image, or figure pages of born-digital
papers that its built-in OCR replaced with a worse pass (corpus/build_corpus.py repairs
both since the text-layer route). It also catches extractor failures that left a
header-only file. This is the audit for files converted before that repair, and the
backstop to run after every build.

Read-only; needs pymupdf. For every `<KEY>__*.md` in the corpus directories it opens
`<zotero>/storage/<KEY>/*.pdf` (the largest when there are several), estimates the text
layer from up to six sampled pages times the page count, and compares it with the
markdown body (everything after the metadata header). A document is reported when
kept / layer is below --max-ratio (default 0.35; the lossy conversions measured so far
kept 0.01-0.16, clean ones 0.6-1.1) and the layer holds at least --min-layer non-space
characters (default 3,000, so abstracts and one-page notes are left alone). "scan" marks
documents whose sampled pages are at least 70 % image. Exit 0 = none, 1 = documents
reported, 2 = a directory could not be read.

To fix a reported document that is not yet ingested, remove its markdown file and run
build_corpus.py again (a missing output is rebuilt). For one already in the workspace,
delete it through LightRAG's own delete path first, then remove the parsed copy, so the
rebuild is ingested under the same id.
"""

import argparse
import glob
import json
import os
import re
import sys
import time


def layer_stats(pdf: str) -> tuple[int, float, int]:
    """(estimated text-layer chars, mean image cover of the sampled pages, pages)."""
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    doc = pymupdf.open(pdf)
    n = doc.page_count
    idx = []
    if n:
        idx = sorted(
            {min(1, n - 1)} | {round((i + 1) * (n - 1) / 6) for i in range(5)}
        )[:6]
    chars, covers = [], []
    for i in idx:
        page = doc[i]
        area = abs(page.rect) or 1.0
        chars.append(len(re.sub(r"\s+", "", page.get_text("text"))))
        try:
            imgs = page.get_image_info()
        except Exception:
            imgs = []
        covers.append(
            min(1.0, sum(abs(pymupdf.Rect(im["bbox"])) for im in imgs) / area)
            if imgs
            else 0.0
        )
    doc.close()
    pymupdf.TOOLS.mupdf_warnings(reset=True)
    if not idx:
        return 0, 0.0, 0
    return int(sum(chars) / len(chars) * n), sum(covers) / len(covers), n


def audit(dirs, zotero, since_hours, max_ratio, min_layer):
    now = time.time()
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(d)
        for p in glob.glob(os.path.join(d, "*.md")):
            if since_hours and now - os.stat(p).st_ctime > since_hours * 3600:
                continue
            files.append(p)
    rows, offenders = [], []
    for p in sorted(files):
        base = os.path.basename(p)
        key = base.split("__", 1)[0]
        pdfs = glob.glob(os.path.join(zotero, "storage", key, "*.pdf"))
        if not pdfs:
            continue
        pdf = max(pdfs, key=os.path.getsize)
        txt = open(p, encoding="utf-8", errors="replace").read()
        body = txt.split("\n---\n", 1)[1] if "\n---\n" in txt else txt
        md_chars = len(re.sub(r"\s+", "", body))
        try:
            layer, cover, pages = layer_stats(pdf)
        except Exception as e:
            rows.append({"key": key, "file": base, "error": type(e).__name__})
            continue
        ratio = md_chars / layer if layer else None
        row = {
            "key": key,
            "file": base,
            "where": os.path.basename(os.path.dirname(p)),
            "md_chars": md_chars,
            "layer_est": layer,
            "pages": pages,
            "cover": round(cover, 2),
            "ratio": round(ratio, 3) if ratio is not None else None,
            "scan": cover >= 0.7,
        }
        rows.append(row)
        if ratio is not None and layer >= min_layer and ratio < max_ratio:
            offenders.append(row)
    return rows, offenders


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--zotero",
        default=os.path.expanduser("~/Zotero"),
        help="Zotero data dir (contains storage/)",
    )
    ap.add_argument(
        "--corpus",
        nargs="+",
        default=None,
        help="directories of <KEY>__*.md files to audit (default: <zotero>/rag_corpus)",
    )
    ap.add_argument(
        "--since-hours",
        type=float,
        default=0.0,
        help="only files changed this recently (0 = all)",
    )
    ap.add_argument("--max-ratio", type=float, default=0.35)
    ap.add_argument("--min-layer", type=int, default=3000)
    ap.add_argument(
        "--keys-out",
        default=None,
        help="write the offending storage keys, one per line",
    )
    ap.add_argument("--json-out", default=None, help="write every audited row as JSON")
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    a = ap.parse_args(argv)
    dirs = a.corpus or [os.path.join(a.zotero, "rag_corpus")]
    t0 = time.time()
    try:
        rows, offenders = audit(dirs, a.zotero, a.since_hours, a.max_ratio, a.min_layer)
    except FileNotFoundError as e:
        print(f"textlayer_audit: ERROR not a directory: {e}", file=sys.stderr)
        return 2
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=0)
    if a.keys_out:
        with open(a.keys_out, "w", encoding="utf-8") as f:
            for r in offenders:
                f.write(r["key"] + "\n")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{stamp} textlayer_audit: {len(offenders)} document(s) kept < {a.max_ratio:.0%} of a "
        f">= {a.min_layer}-char text layer, out of {len(rows)} audited in {' '.join(dirs)} "
        f"({time.time() - t0:.0f} s)"
    )
    if not a.quiet:
        for r in sorted(offenders, key=lambda r: r["ratio"]):
            print(
                f"  {r['ratio']:.3f} kept  md {r['md_chars']:>7} / layer {r['layer_est']:>8}  "
                f"{r['pages']:>4} p  {'scan' if r['scan'] else 'text'}  {r['where'][:8]:8}  "
                f"{r['file'][:70]}"
            )
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main())
