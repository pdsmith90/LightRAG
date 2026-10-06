# corpus: Zotero library → clean Markdown for LightRAG

Three scripts that turn a Zotero data directory into a Markdown corpus that LightRAG
(or any RAG indexer) can ingest. Each document gets a metadata header, and each file
name starts with its Zotero storage key, so answers can cite back to Zotero.

```
zotero.sqlite ──build_metadata.py──▶ zotero_metadata.json ─┐
                                                           ├─build_corpus.py──▶ rag_corpus/<KEY>__<slug>.md ──quarantine_junk.py
storage/<KEY>/<attachment> ────────────────────────────────┘                    + .manifest.jsonl
```

| Script | What it does |
|---|---|
| `build_metadata.py` | Reads `zotero.sqlite` and writes `zotero_metadata.json`, which maps every attachment's storage key to its bibliographic metadata. A rebuild never forgets a key: entries for attachments that Zotero no longer has are carried forward and flagged `"orphan": true`. |
| `build_corpus.py` | Converts each `storage/<KEY>/` attachment into cleaned Markdown with a metadata header. It is resumable (only missing or outdated outputs are rebuilt), runs in parallel, and records what it did in `.manifest.jsonl`. |
| `quarantine_junk.py` | Moves webpage-snapshot junk (a high markup/base64 ratio, or a denylist entry) and glyph-garbage files (a text layer that is mostly Private Use Area code points, U+FFFD replacement characters, control codes or letter soup — `build_corpus.garble_reason`'s rules) out of the corpus root into `<corpus>_junk_quarantine/`. It never deletes anything and always exits 0. |

## Install

Python 3.10 or newer.

```bash
python3 -m pip install -r corpus/requirements.txt
```

`build_metadata.py` and `quarantine_junk.py` need only the standard library. For
`build_corpus.py`, `pymupdf4llm` is the PDF engine. The other packages are fallbacks
for other formats. `build_corpus.py` also uses these tools automatically when they
are on `PATH`:

- `pandoc` for docx, epub and html.
- `djvutxt` (from djvulibre) for djvu.
- `ocrmypdf`, which needs tesseract and ghostscript, to OCR scanned PDFs.

## Quick start

```bash
# 1. metadata. Zotero usually locks zotero.sqlite while it runs. If you see
#    "database is locked", close Zotero or point --zotero at a copy of the data dir.
python3 corpus/build_metadata.py --zotero ~/Zotero --out ~/Zotero/zotero_metadata.json

# 2. pilot a few documents, then drop --limit for the full run
python3 corpus/build_corpus.py --zotero ~/Zotero --out ~/Zotero/rag_corpus \
        --meta ~/Zotero/zotero_metadata.json --jobs 8 --ocr auto --limit 25

# 3. optional: keep saved-webpage junk out of the index
python3 corpus/quarantine_junk.py --corpus ~/Zotero/rag_corpus
```

Then point LightRAG's `INPUT_DIR` at the corpus directory. When your library changes,
re-run steps 1 to 3. Only new or changed attachments are converted again. LightRAG
indexes the new ones on its next scan, but not a changed one until its old document is
deleted: see [Re-indexing a changed attachment](#re-indexing-a-changed-attachment).

## Settings

All paths come from flags. The single environment variable is `ZOTERO_METADATA`,
which the MCP server in this toolkit also reads to resolve citations. Set it to
one absolute path, and the metadata builder, the corpus builder and the MCP server
all use the same file.

### build_metadata.py

| Flag | Default | Meaning |
|---|---|---|
| `--zotero DIR` | `~/Zotero` | Zotero data directory (contains `zotero.sqlite`). |
| `--out FILE` | `$ZOTERO_METADATA`, else `<zotero>/zotero_metadata.json` | Output JSON. The previous file at this path is the source of carried-forward orphans. It is replaced atomically. |

### build_corpus.py

| Flag | Default | Meaning |
|---|---|---|
| `--zotero DIR` | `~/Zotero` | Zotero data directory (contains `storage/`). |
| `--out DIR` | `<zotero>/rag_corpus` | Output corpus directory. |
| `--meta FILE` | `$ZOTERO_METADATA`, else `<zotero>/zotero_metadata.json` | Metadata from `build_metadata.py`. If the file is missing, headers carry only the title `(untitled)` and the Zotero link. |
| `--jobs N` | CPU count − 2 | Worker processes. Each worker can itself use several cores, because recent pymupdf4llm and Tesseract OCR are multi-threaded, so a few workers can keep many cores busy. Lower N on a shared machine. |
| `--ocr auto\|off\|force` | `auto` | `auto` and `force` both run `ocrmypdf`, if installed, on PDFs with no usable text layer: text shorter than `--min-chars`, or a garbled text layer (see *Converting each format*). `off` never runs it. |
| `--min-chars N` | `200` | A PDF whose extracted text is shorter than this (whitespace excluded) is treated as image-only. |
| `--limit N` | `0` (all) | Process only the first N storage folders, sorted by key. |
| `--force` | off | Rebuild even when the output is up to date. |
| `--denylist FILES` | none | Comma-separated denylist files (see below). Pass the same list to `quarantine_junk.py`. |
| `--held-dir DIR` | none | Optional. A directory where an external ingest scheduler parks built but not yet ingested files. An up-to-date file there counts as built. |
| `--ocr-cache-dir DIR` | none | Optional. Text you have already OCR'd, named `<pdf stem>.pdf.txt`. It is reused for image-only PDFs before `ocrmypdf` runs. |
| `--fallback-md-dir DIR` | none | Optional. Text extracted earlier (for example, old `pdftotext` output), named `<pdf stem>.md`. It is the last resort for image-only PDFs. |
| `--engine` | `pymupdf4llm` | The only PDF engine. |

### quarantine_junk.py

| Flag | Default | Meaning |
|---|---|---|
| `--corpus DIR` | `~/Zotero/rag_corpus` | Only the top level is scanned. Files that LightRAG has already moved into `__parsed__/` are never touched. |
| `--quarantine DIR` | `<corpus>_junk_quarantine` | Where files are moved. Each move is appended to `quarantined.jsonl` in this directory. |
| `--denylist FILES` | none | Comma-separated denylist files. |
| `--min-junk R` | `0.90` | Move files whose junk ratio is at or above R. The ratio is `1 − prose/total`, where prose is the text left after removing base64 runs and HTML tags. |
| `--min-chars N` | `500` | The ratio rule ignores files smaller than N. |
| `--min-pua R` | `0.30` | Move files whose non-whitespace text is at least R Private Use Area code points (glyph codes from a font without a Unicode mapping, see *Converting each format*). Files with fewer than 500 non-whitespace characters are left alone. |
| `--dry-run` | off | Report only. |

**Denylist files.** Put one entry per line. `#` starts a comment, and only the first
token on a line counts. An entry is one of two things:

- A bare storage key, which blocks every output from that attachment.
- An exact output file name ending in `.md`, which blocks only that file. This lets
  a hand-recovered sibling with the same key through.

`build_corpus.py` skips denylisted outputs, and `quarantine_junk.py` moves them out.
Both scripts use the same loader.

## Output

### File names

```
<KEY>__<slug>.md
```

- `<KEY>` is the 8-character Zotero storage key of the attachment, which is also the
  folder name under `storage/`.
- `<slug>` is the attachment's file name without its extension. Each run of
  characters outside `[A-Za-z0-9._-]` becomes `_`, and the result is cut to 80
  characters (`untitled` if nothing is left).

The key prefix makes names unique and lets a consumer resolve citations: the MCP
server matches `^[A-Z0-9]{8}__` and looks the key up in `zotero_metadata.json`.
**Keep this naming stable.**

### Choosing the file in each folder

The attachment's own file name from the metadata wins. Otherwise the choice goes by
format priority, `pdf > epub > docx > pptx/ppt > djvu > html/htm > txt`, and then by
the largest file.

Only `storage/` folders are converted. Linked files that live outside `storage/` are
not.

### Metadata header

Every document starts with this header:

```markdown
# <title>

**Authors:** Doe Jane; Roe Richard            (at most 12, then "et al.")
**Year:** 2019  **Published in:** <publication>  **DOI:** <doi>
**Tags:** tag1, tag2                           (at most 20)
**Zotero:** zotero://select/library/items/<KEY>

> **Abstract.** <abstract, whitespace collapsed>

---
```

Lines whose fields are empty are left out.

### Converting each format

- **PDF.** The file goes through pymupdf4llm, one page at a time with the engine's
  own OCR off, and every page is then judged. A page the engine emptied — it kept less
  than 40 % of a text layer of at least 400 characters, which happens to scans whose
  OCR text is drawn under the page image (the page comes back as its download watermark
  only) and to figure pages of born-digital papers — or turned into **glyph garbage** is
  replaced by that page's text layer as paragraphs, provided the layer itself reads as
  text; the other pages keep their markdown. Glyph garbage is what `garble_reason`
  recognises: Private Use Area code points (at least 30 % of the non-space characters),
  U+FFFD replacement characters (5 %, what the engine emits for a font it cannot decode
  while the layer itself is fine), C0 control codes (5 % on a page with under 25 % of
  its characters inside words — TeX symbol fonts leave a few per cent on equation pages
  of real prose), letter soup (35 % of the tokens are single non-digit characters, under
  15 % of the characters are in words and under 30 % are digits — numeric tables are
  mostly digits) or letters that form no words at all. A page whose layer is garbage too
  is dropped, and convert() sends those pages to `ocrmypdf --pages` (all pages once they
  are the majority): such files are labelled `pdf+ocr_pages` (or `pdf+ocr`), or
  `pdf_garbled_pages_dropped` when OCR is off or fails, and the log says `GARBLED pages
  <file>: k/n page(s) ...`. Repaired files are labelled `pdf+textlayer` and the log says
  `TEXTLAYER <file>: pymupdf4llm kept N of M text-layer chars; k/n page(s) replaced`. A
  PDF whose pages are full-page images with no text layer at all (`IMAGE-ONLY` in the
  log) goes straight to the OCR fallbacks below. If the whole text is shorter than
  `--min-chars`, or garbled by the same rules (the log says `GARBLED text layer`), the
  fallbacks are tried in order: `--ocr-cache-dir`, then `ocrmypdf` (unless `--ocr off`;
  its time limit is 600 s or 5 s per page OCR'd, whichever is longer), then
  `--fallback-md-dir`. A garbled layer whose fallbacks are glyph codes too is written as
  `pdf_needs_ocr` with the header only, never the garbage. Control codes are stripped
  from every conversion in any case (PostgreSQL refuses NUL in text). The bars were
  measured on a corpus of several thousand academic PDFs: prose has 50-70 % of its
  characters inside words of four or more letters and every garbage shape under 10 %;
  equation-heavy pages sit at 2-26 % but are mostly digits.
- **HTML snapshot.** Zotero's own `.zotero-ft-cache` in the same folder is used when
  it is non-empty. Otherwise the page is converted with pandoc, or with
  BeautifulSoup if pandoc is missing.
- **docx, epub.** pandoc, falling back to python-docx or ebooklib.
- **pptx.** python-pptx. A legacy binary `.ppt` is attempted too, but python-pptx
  cannot read it, so its body comes out empty.
- **djvu.** `djvutxt`.
- **txt.** Read as is.

### Cleaning

Every body passes through `clean_md`, in this order:

1. OCR'd picture-text blocks and `data:` image links are removed.
2. Unbroken base64 runs of 200 characters or more are removed.
3. `<script>`/`<style>` blocks are removed together with their content.
4. `<br>` soup becomes newlines, and HTML tag markup is stripped. Only real tag
   shapes are stripped, and never the element text, so math like `a<b, c>d`
   survives.
5. **Back-of-book indexes are removed.** A subject, name or author index (a heading
   such as "Index", "Subject Index" or "Index of Programs") in the last quarter of a
   document is dropped from that heading to the end of the document, or to the first
   heading that is not part of the index, or to a table: back matter such as a table
   of constants or a publisher's series page stays. The region goes only when at
   least 60% of its lines read as index entries (page numbers, or a "see"
   cross-reference). Index chunks carry no content, and small extraction models tend
   to loop on them until the request times out. Each decision is printed to the
   build log:

   ```
   [build_corpus] INDEX stripped|kept <file>: ...
   ```
6. **Reference-list sections are removed.** Detection works line by line. A run of
   at least 5 citation-shaped entries is dropped, together with a
   "References"-style heading directly above it. The detector tolerates wrapped
   entries, running headers and many citation styles. Tables and code fences are
   never touched.

   A **safety valve** keeps the document whole when stripping would remove more
   than 60% of it. The exception is when at least 10,000 characters of body would
   remain: then up to 75% may go. Each valve decision is printed to the build log:

   ```
   [build_corpus] VALVE relaxed|kept whole <file>: references N% of the text, M chars of body -- ...
   ```
7. **Repeated page furniture is removed.** A line of 40 or more characters that
   repeats 5 or more times (digits normalised) keeps its first occurrence, and the
   later copies are dropped. Markdown structure (tables, lists, headings, quotes)
   is never a candidate. If this would shrink the document by more than 25%, the
   document is left unchanged.
8. Runs of blank lines are collapsed.

### When an output counts as up to date

A document is skipped, unless `--force` is given, when its output has an mtime at or
after the source attachment's. The first one found is checked, in this order:

1. `<out>/<name>`
2. `<out>/__parsed__/<name>`, where LightRAG moves inputs after ingesting them. If that
   copy is older than the source, the newest of it and the numbered copies LightRAG
   filed beside it (`<stem>_001.md`, `<stem>_002.md`, …) counts.
3. `<held-dir>/<name>`, only if `--held-dir` is given.

A changed source is rebuilt at the corpus root. When an earlier copy is in
`__parsed__/`, the build log says so:

```
[build_corpus] REBUILT <file>: an earlier copy is in __parsed__/; a scan indexes this one only if LightRAG's document with this name was deleted first
```

### Re-indexing a changed attachment

LightRAG identifies a document by its file name (checked against LightRAG 1.5.7). A
scan that finds a file whose name belongs to a document LightRAG has already processed
does not index it. It moves the file into `__parsed__/`, as `<stem>_001.md` or the next
free number when the earlier copy is still there, and the knowledge base keeps the old
text. A changed attachment keeps its file name, so its rebuild is indexed only once the
old document is gone. The same holds for every file that `--force` rebuilds.

To re-index a changed attachment, in this order:

1. Delete its document in LightRAG: in the WebUI with **Also delete uploaded files**
   checked, or with `DELETE /documents/delete_document` and
   `{"doc_ids": ["<doc id>"], "delete_file": true}`. The file option makes LightRAG
   remove the files with that name from the corpus directory and from `__parsed__/`,
   numbered copies included. If LightRAG keeps them instead (its pipeline messages then
   say `Source file preserved`), remove them by hand.
2. If the workspace is in PostgreSQL, sweep now, before the document is added again
   ([maintenance/README.md](../maintenance/README.md#safety) explains why the order matters).
3. Run `build_corpus.py`. With no copy left, it rebuilds the document at the corpus root.
4. Scan.

A numbered copy counts as built whether LightRAG indexed it or only set it aside, so
each rebuild reaches LightRAG once instead of coming back to the corpus root on every
run.

### Manifest

`<out>/.manifest.jsonl` is rewritten on every run. It holds one line per storage
folder:

```json
{"key": "ABCD2345", "method": "pdf", "out": "ABCD2345__Doe_2020_paper.md", "chars": 48213}
```

| `method` | Meaning |
|---|---|
| `pdf`, `epub`, `docx`, `pptx`, `djvu`, `html`, `txt` | Converted with that format's extractor. |
| `html+ftcache` | Built from Zotero's `.zotero-ft-cache`. |
| `pdf+ocr` | OCR'd by `ocrmypdf`. With `--ocr off` and no cached or fallback text, the result is `pdf+ocr_empty`. |
| `pdf+ocr_cache` | Taken from `--ocr-cache-dir`. |
| `pdf_legacy_md` | Taken from `--fallback-md-dir`. |
| `pdf_needs_ocr` | Image-only PDF, or one whose text layer is glyph codes, and OCR was unavailable, off or failed. It is indexed by its header only. |
| `…_empty` suffix | The body came out shorter than `--min-chars`. |
| `skip` | Up to date. |
| `denylist` | Blocked by `--denylist`. |
| `no_source` | No convertible file in the folder. |
| `error:<Type>` | The extractor raised an exception. |

## zotero_metadata.json schema

The file is a single JSON object. Each key is the **attachment item key**, which is
the `storage/<KEY>/` folder name and the `<KEY>` prefix of a corpus file.

```json
{
  "ABCD2345": {
    "filename": "Doe 2020 paper.pdf",
    "title": "A paper title",
    "authors": ["Doe Jane", "Roe Richard"],
    "year": "2020",
    "doi": "10.1234/example",
    "publication": "Journal of Examples",
    "tags": ["tag1", "tag2"],
    "abstract": "First 1500 characters of the abstract",
    "source": "parent",
    "orphan": true
  }
}
```

| Field | Type | Content |
|---|---|---|
| `filename` | string | Attachment file name with Zotero's `storage:` prefix removed. A linked file keeps its stored path. A linked URL has `""`. |
| `title` | string | The parent item's title, else the attachment's own, else the file name without its extension. |
| `authors` | list of strings | `"lastName firstName"` for every creator (authors, editors, …) in Zotero order. Single-field names are just the name. |
| `year` | string | The first 4-digit run in the `date` field, or `""`. |
| `doi` | string | `DOI` field, or `""`. |
| `publication` | string | The first non-empty of `publicationTitle`, `bookTitle`, `proceedingsTitle`, `conferenceName`, or `""`. |
| `tags` | list of strings | Tag names. |
| `abstract` | string | `abstractNote`, cut to 1500 characters. |
| `source` | `"parent"` or `"standalone"` | Whether the metadata came from the parent item or from the attachment itself. |
| `orphan` | `true`, optional | Present only on entries carried forward from the previous file for keys that Zotero no longer has. |

Every attachment type is included, not just PDFs. Metadata comes from the parent item
when there is one.

## Limitations

- Zotero keeps `zotero.sqlite` locked while it runs. Close Zotero, or run
  `build_metadata.py` against a copy of the data directory.
- The `zotero://select/library/items/<KEY>` link targets your personal library.
  Items in group libraries need `zotero://select/groups/<id>/items/<KEY>`, which is
  not generated.
- `--ocr force` currently behaves exactly like `auto`: OCR runs only on PDFs without
  a usable text layer (empty or glyph codes).
- The reference-list stripper is tuned on scientific journal articles, books and
  theses in many citation styles. Documents that are mostly bibliography are
  kept whole by the safety valve rather than emptied.
- The index stripper acts only on the last quarter of a document and stops at the
  first table row, so an index laid out as a table, or placed earlier by a PDF's
  page order, is kept.

## Tests

From the `zotero-toolkit/` directory, in a fresh virtual environment:

```bash
python3 -m pip install -r corpus/requirements.txt pytest
python3 -m pytest tests/corpus
```

The tests are offline. The end-to-end test builds a tiny synthetic Zotero data
directory: a minimal `zotero.sqlite`, an HTML snapshot with a `.zotero-ft-cache`, text
attachments and generated PDFs. It runs `build_metadata.py` and then
`build_corpus.py` through their command lines. The PDF part is skipped when PyMuPDF
or pymupdf4llm is not installed.
