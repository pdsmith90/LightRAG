# mcp — LightRAG MCP server with Zotero citations

`lightrag_mcp_server.py` is a small stdio [MCP](https://modelcontextprotocol.io) server
that lets an agent (Claude Code, Claude Desktop, any MCP client) query a
[LightRAG](https://github.com/HKUDS/LightRAG) knowledge base built from a Zotero library.
It proxies the LightRAG Server REST API and turns the server's retrieved references
into real bibliographic citations using Zotero metadata. The LLM never writes them.

## Tools

| Tool | What it returns |
|---|---|
| `search_library(query, mode="mix", top_k=10, answer=False, include_graph=False, passages=8)` | Compiled citations **first**, then the retrieved passages as plain text, each headed `[n] Family Year · section` (see [Retrieved passages](#retrieved-passages)). No answer is generated, so it is faster. `passages` sets how many are retrieved. `include_graph=True` appends the knowledge-graph entities and relations. |
| `search_library(..., answer=True)` | The server's synthesized answer. Any reference section the model wrote is stripped, and a `### Sources (compiled from retrieval)` list is appended. |
| `paper_passages(paper, question)` | Up to 3 passages of **one** paper ranked against `question`, under that paper's citation and, when this host has it, the path of the paper's file in Zotero's storage folder. `paper` is a Zotero key, a DOI, or words that must all occur in the title, authors or year. If no indexed paper, several, or only near matches fit, it lists them instead of querying. |
| `paper_equation(paper, number, page=0)` | One numbered display equation of **one** paper as `$…$` LaTeX with its printed number as `\tag{N}`, read from the paper's own PDF (see [Equations and tables from the PDF](#equations-and-tables-from-the-pdf)), with the PDF's text layer for the same box, the page and the sentence that cites it. Without an OCR model the answer is the text layer, labelled as such. |
| `paper_table(paper, number)` | One numbered table of **one** paper as a Markdown table with its caption and page, read from the PDF's text layer by the layout converter. No OCR model. |
| `find_paper(paper)` | A lookup with the same `paper` forms: the matching papers that are indexed, then the Zotero items that match but are not, then near matches when nothing has every word. No retrieval and no LLM. |
| `add_note(text, description="")` | Inserts a text note into the knowledge base (`POST /documents/text`) under a unique document name (see below). |
| `health()` | The server's `/health` status, or `unreachable at <url>: <error>`. |

`mode` is one of `mix` (default: knowledge graph + vectors), `hybrid`, `local`, `global` or
`naive`. Any other value, including LightRAG's `bypass`, falls back to `mix`. `top_k` is
clamped to `1..LIGHTRAG_MCP_MAX_TOP_K` (see below). It sizes the knowledge-graph legs of
retrieval, not the result: `passages` (default 8, clamped to `1..12`) is sent as the
request's `chunk_top_k` and decides how many passages come back, at about 4 KB each.
`answer=True` does not send it.

Every tool returns one text block and declares no output schema, so a client that keeps
both a result's text and its `structuredContent` does not receive the text twice.

LightRAG uses a text insert's `file_source` as the document's name, and refuses with HTTP
409 a name that its document status storage already holds. So `add_note` never sends a
fixed name. Each note is named `note-<UTC time>-<first 8 hex digits of the text's SHA-1>`,
for example `note-20260101T120000Z-1a2b3c4d`. A `description`, if given, goes in front:
`reading notes (note-20260101T120000Z-1a2b3c4d)`. In the description, each run of
whitespace, control characters (line breaks included), `/` or `\` becomes one space,
because LightRAG keeps only the last path segment of a name and refuses control
characters. Citations list the note by this name. LightRAG still deduplicates by content:
a note whose text matches a document it already holds is accepted, then recorded as a
failed duplicate instead of being indexed.

## How citations are compiled

LightRAG's `/query` and `/query/data` responses carry a structured `references` array:
one `{"reference_id": "1", "file_path": "..."}` per retrieved document. The proxy renders
every entry, in the server's order, keeping the server's `reference_id`s so inline `[n]`
markers in an answer still line up.

- A file named `<KEY>__<slug>.md`, where `KEY` is the 8-character Zotero attachment key,
  uses the LightRAG server's current `citation` when supplied; otherwise it is looked up
  in the local `zotero_metadata.json`. The local fallback formats
  `Authors. (Year). Title. Publication. doi:DOI [zotero://select/library/items/KEY]`;
  the server's citation is kept verbatim with the same Zotero link appended.
- If the key is not in the metadata, the slug is de-slugged into a title (`Some_Paper_Title`
  becomes `Some Paper Title`) and the `zotero://` link is kept.
- Any other file name, such as a note added with `add_note`, is listed as the bare file
  name. Directory parts of a path are dropped.

`zotero_metadata.json` is the file the toolkit's corpus builder writes: a JSON object keyed
by attachment key. The proxy reads only these fields of each entry:

```json
{
  "AAAA0001": {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "year": "2017",
    "publication": "Advances in Neural Information Processing Systems",
    "doi": "10.48550/arXiv.1706.03762"
  }
}
```

The proxy checks the file for changes when compiling citations and reloads it without a
restart. An atomic replacement takes effect on the next request. If the file becomes
unreadable, the last valid metadata is retained until it recovers; if it was absent at
startup, citations fall back to de-slugged file names until the file appears.

### Retrieved passages

A context-only `/query` response carries the knowledge-graph entities, the relations, the
document chunks (one JSON object per line: `reference_id`, `content` and, on newer
servers, `content_headings`) and the server's own reference list. `search_library`
returns the compiled citations, then each chunk in the server's ranked order as plain
text, headed by its `[n]`, the first author's family name and year (from `creators`, else
the server's `citation`, else `authors`) and its section path. The label keeps the
passages of several same-topic papers apart. The section path is the server's
`content_headings` without the levels that do not place the passage: the paper's title
(whole, cut with `…`, or garbled into word pieces by the PDF converter: at least 80 % of
its words of three or more letters occur in the folded title, and it is long enough to be
the title), the journal (`publication`), an author line (two or more of the paper's family
names), `Abstract`/`Summary` (also letter-spaced, `S U M M A R Y`) and article-type labels
such as `LETTER`, and fragments with no three-letter word, an `=` or `|`, a link or a
received/accepted date. Bold, italics, HTML tags and backticks are removed, and a passage
with no level left shows no section. A paper's first chunk begins with the header
the corpus builder writes (`# Title`, `**Authors:**`, `**Year:**`, `**Tags:**`,
`**Zotero:**` lines, an optional `> **Abstract.**` line, then `---`). That header is
dropped, because the citation carries it; the abstract line is kept only when the first
80 letters and digits of the abstract (folded, see below, and without a leading
`Abstract`, which Zotero's copy often glues to the first word) do not occur among the
letters and digits of the text after it. A
chunk without the `**Zotero:**` line is left as it is. A passage whose text, whitespace
collapsed, repeats an earlier one in the same result (another Zotero copy of the paper) is
dropped, and its `[n]` joins the earlier heading: `[1][4] Family Year · section`. The entities and relations follow only
with `include_graph=True`, and the server's reference list is dropped because the
compiled citations replace it. When the result would exceed `LIGHTRAG_MCP_MAX_CHARS`,
the lowest-ranked passages are left out and a closing line says how many. A response in
any other layout is passed through whole after the citations.

### One paper's passages and the paper lookup

`find_paper` asks `GET /graph/papers/search?q=…&limit=100` (the zlightrag fork's paper
lookup, which lists processed documents only) which papers are indexed, and matches the
same words against `zotero_metadata.json`. Both sides are folded first: NFKD, accents
dropped, letters that NFKD keeps (`ı ł ø đ ð ß æ œ þ`) spelled out, every Unicode hyphen,
dash, minus sign and soft hyphen made `-`, apostrophes and primes deleted, lowercase. A
query word, without surrounding `()[]{},.;:"!?*`, curly double quotes and guillemets (and
an `et al` that follows a name, as in `Doe et al. (2011)`), matches when it occurs in an entry's
title, authors, year, DOI or key; when it does without its hyphens in that text without
hyphens; or when each of its parts split on `-` or `/` does (a DOI-shaped query is matched
whole, not loosely). So `widget-fo`, `widgetfo` and `orchards` match a title written
`Widget‐FO` (U+2010) or `Orchard’s`. A BibTeX key such as
`vaswaniAttentionAllYou2017` becomes the words `vaswani` and `2017` and the title words
`attention`, `all`, `you` (camel words split as the server splits them: acronyms and digit
runs apart, `IRI2016Model` is `iri`, `2016`, `model`), each matched against the text with
every non-alphanumeric removed. A token with no letter or digit (the ` - ` of
`Author - Year - Title`, an `&`) is no word. The server's search carries the same rule, and
`tests/mcp/paper_search_cases.json` holds the cases both suites run.

A metadata entry the words search did not return is confirmed against the server by key
before it is called not indexed: a server that sends `X-Paper-Search: 2` answers
`GET /graph/papers/search?keys=K1,K2,…` for 50 keys at a time; an older one gets one request
per key, four at a time. Entries are asked about in order until 10 are known not to be
indexed or 100 keys were asked (`CONFIRM_MAX`); entries past that are listed apart as
"index status NOT checked", never as not indexed. One copy is listed per DOI. Lookups are cached
in the proxy for 120 seconds (256 requests at most). Indexed papers are listed with a
title equal to the query's title words (those that are not an author's name or a year)
first, then titles that hold them as a phrase, then by query words in the title, then in
the server's order. After 10 rows a list says how many more it does not show, and that a
cut list does not show absence. When the server finds 100 papers its list may be cut, so
the Zotero-only list is withheld and the answer asks for a narrower lookup.

When no item has every word, the answer says only that, and that it does not show the
paper is missing (a wrong year, a differently spelled name, a topic word not in the title
give it too); it then lists up to 5 near matches in a separate
section, each marked indexed or Zotero only: entries matching every word but a year that is
off by one, then entries whose author family name is the query's surname (the BibTeX
family, else its first non-numeric word), within a year of a year the query gives. On a
stock LightRAG Server the route answers 404 and the index status is reported as unknown.

`paper_passages` resolves `paper` the same way and never queries for a near match. With
one indexed paper it sends `POST /graph/papers/passages` with all of the paper's keys (the
copy with a DOI, else the longest title, first), the bare question and `top_n` 3, and
renders each passage, best first, as `[1] Family Year · section · chunk i of n` (n is the chunk count of the paper's longest copy). Under the citation it names the paper's file, `Local file: <ZOTERO_STORAGE>/<key>/<filename>`, when that file exists on the host, because the index lacks what the corpus builder's text-layer converter drops from born-digital PDFs: display equations (only references such as "eq. (5)" remain) and often a table's layout. The tool's description sends the agent to `paper_equation` and `paper_table` for them. On a
server without that route (404) it falls back to a `naive` context-only `/query` led by the first author
and year (plus up to 3 title words when the metadata holds another work by that author in
that year), keeps only the chunks of the paper's keys, and lists at most 5 other papers
the question matched as `KEY · Family Year · title`. On a stock server it resolves `paper`
from the metadata alone. A server that caps chunks per document (the fork's
`MAX_CHUNKS_PER_DOC`) returns at most that many passages of the paper per fallback call.

### Equations and tables from the PDF

`paper_equation` and `paper_table` resolve `paper` as `paper_passages` does (an indexed paper, or the one Zotero item that fits) and read the paper's file from `ZOTERO_STORAGE`; a paper with no PDF on the host is reported as such. The PDF work is done by `paper_pdf_tools.py`, which the proxy runs as a subprocess with `-I` under `LIGHTRAG_MCP_PDF_PYTHON` (default `../.venv/bin/python` if that exists, else the proxy's own interpreter), so the proxy's interpreter never imports PyMuPDF. That interpreter needs `pymupdf` and `pymupdf4llm` (with its layout package). The helper prints one JSON object; the proxy words it.

- **Equation.** Pages that have a text line ending `(N)` are run through PyMuPDF's layout model, one page at a time. The `formula` box that holds the number is cropped at 300 dpi and transcribed to LaTeX by an OCR model on a CPU-only `llama-server` that is started for the call and stopped after it (`--device none -ngl 0`, GPUs hidden by environment, killed with its caller). The number comes from the text layer, as `\tag{N}`; the model leaves numbers out. A box that holds several numbered equations is paired with them by order when the counts match. The answer also gives the box's text layer (to check digits; numbers it has that the LaTeX lacks are listed), the page, and the sentence that cites the equation ("according to eqs (6) and (7)…"). When no OCR model is configured, it fails, it times out or the transcription does not finish, the answer is the box's text layer and says so in capitals. On a scanned page the text layer is an OCR layer and the answer says that too. Layout results, each page's word count, equation-number positions, caption blocks and text, and the OCR text are cached per file (the OCR text also by the model file's size and date), saved every few seconds and on SIGTERM, so the equations beside one already read are instant and a call that timed out continues from the pages it had read when asked again.
- **Table.** The `Table N` caption (the number may be Arabic or Roman, `A1`, `S2`) is matched to the nearest `table` box and to the pipe table the layout converter wrote for it. Cell line breaks stay as `<br>`. A row that the converter merged into several lines per cell is split into rows only when its first cell holds as many labels (wrapped and unit lines joined to the label above) as every other cell has lines; otherwise the lines stay joined by `<br>` and the answer says so. Superscripts become Unicode (`10⁻⁶`), the converter's emphasis marks go, and a dropped `fl`/`fi` ligature is restored from the text layer. No OCR model: a model that transcribes the page image follows the page's geometry, and a confident wrong table is worse than a missing one.
- **Guards.** A page the layout model cannot take (more than 3,500 text spans or 30 MB of content stream: it holds some 25 N×N matrices) is not given to it, the helper's address space is capped (`LIGHTRAG_MCP_PDF_MEM_GIB`, default 7) and it asks the kernel to kill it first. One OCR server runs at a time, whatever the number of callers. A call that outlives the proxy's time limit gets SIGTERM first (so the helper saves its cache and stops its OCR server), SIGKILL ten seconds later; the helper also asks the kernel for a SIGTERM if the proxy dies, and the OCR server for a SIGKILL if the helper does.

| Variable | Default | Meaning |
|---|---|---|
| `LIGHTRAG_MCP_PDF_PYTHON` | `../.venv/bin/python`, else the proxy's interpreter | Interpreter with PyMuPDF for the helper. |
| `LIGHTRAG_MCP_PDF_HELPER` | `paper_pdf_tools.py` beside the proxy | The helper script. |
| `LIGHTRAG_MCP_OCR_TIMEOUT` | `120` | Seconds for the OCR server's start and one transcription. The proxy stops the helper `120` s after that. |
| `LIGHTRAG_MCP_OCR` | *(on)* | `off` skips the OCR model: equations come back as text layer. |
| `LIGHTRAG_MCP_OCR_LLAMA_SERVER`, `LIGHTRAG_MCP_OCR_MODEL`, `LIGHTRAG_MCP_OCR_MMPROJ` | `llama-server` on `PATH`; none; none | The OCR model: any vision model that answers an image with LaTeX for a prompt like the one in the helper. Without a model and projector the OCR step is unavailable. |
| `LIGHTRAG_MCP_OCR_THREADS` | `8` | CPU threads of the OCR server. |
| `LIGHTRAG_MCP_CACHE` | `~/.cache/lightrag-mcp-pdf` | Layout and OCR cache. |

### Answer mode (`answer=True`)

The answering model is asked not to write a References section, and whatever reference
section it writes anyway is removed. A heading line that is exactly `References` or
`Sources` is removed together with the entry lines under it. The heading may be in
Markdown (`### References`) or bare, and may end with a colon; case does not matter. Entry
lines look like `[n] …`, `- [n] …` or `* [n] …`. Prose resumes after the block. The heading
must end the line, so prose headings such as `### Sources of error` are kept. A
server-side citation block titled `### Sources` is stripped the same way, so the compiled
list is never duplicated.

If `/query` returns no references (for example a cached answer on some LightRAG versions,
or a retrieval that admitted no document chunks), the proxy reuses references from an
identical earlier query in the same session. Failing that, it re-runs retrieval with
`POST /query/data`, capped at `LIGHTRAG_MCP_REF_FALLBACK_TIMEOUT`. If that also fails, the
answer is returned without a sources list rather than with an invented one.

### Server compatibility

The proxy needs only what stock LightRAG provides. It was verified against an
unmodified LightRAG Server **v1.5.7**: `/query` returns `references` for both answer and
context-only (`only_need_context`) requests, in `naive` and `mix` modes and on an answer
cache hit, and `/query/data` returns them under `data.references`. Server-side patches
are not required. The `references` field on `/query` first shipped in LightRAG v1.4.9,
so use v1.4.9 or newer.

For the `KEY__slug.md` lookup to work, each document's LightRAG file path must be its
corpus file name. Stock LightRAG records the base file name for scanned and uploaded
files, and the `file_source` for text inserts; the proxy drops any directory part itself.
Any other file name still works, but it cites as a bare name.

Authentication is by API key only. Set `LIGHTRAG_API_KEY` on the server, and the same value
here; it is sent as the `X-API-Key` header, never in a URL. A server protected only by
account login (`AUTH_ACCOUNTS` without `LIGHTRAG_API_KEY`) needs a session token, which
this proxy does not obtain.

## Install

Python 3.10 or newer. From the `zotero-toolkit/` directory:

```bash
python3 -m venv ~/.venvs/zotero-lightrag-mcp
~/.venvs/zotero-lightrag-mcp/bin/pip install -r mcp/requirements.txt
```

Dependencies are `mcp` 2.x and `httpx`. `paper_equation` and `paper_table` also need
`pymupdf` and `pymupdf4llm` in the interpreter named by `LIGHTRAG_MCP_PDF_PYTHON`. The server uses `mcp.server.mcpserver.MCPServer`,
which replaced `mcp.server.fastmcp` in mcp 2.0; mcp 1.x does not have it.

## Configuration

Every setting is an environment variable, set in the MCP client's server entry.

| Variable | Default | Meaning |
|---|---|---|
| `LIGHTRAG_BASE_URL` | `http://localhost:9621` | LightRAG Server base URL. A trailing `/` is ignored. |
| `LIGHTRAG_API_KEY` | *(empty: no header sent)* | Must equal `LIGHTRAG_API_KEY` in the server's `.env`. Sent as `X-API-Key`. |
| `ZOTERO_METADATA` | `~/Zotero/zotero_metadata.json` | Path to the metadata file. Unset or empty means the corpus builder's default output. A relative path resolves against the **working directory the client launches the server in**, which you usually do not control, so use an absolute path. `~` is expanded. |
| `ZOTERO_STORAGE` | `~/Zotero/storage` | Zotero's attachment folders (`storage/<key>/`). `paper_passages` names a paper's file there (the metadata's `filename`) when it exists; nothing is read from it. `~` is expanded. |
| `LIGHTRAG_MCP_TIMEOUT` | `300` | Seconds allowed for `/query` and `/documents/text` requests. |
| `LIGHTRAG_MCP_REF_FALLBACK_TIMEOUT` | `90` | Seconds allowed for the `/query/data` reference fallback in answer mode. |
| `LIGHTRAG_MCP_HEALTH_TIMEOUT` | `15` | Seconds allowed for `health()`. |
| `LIGHTRAG_MCP_MAX_TOP_K` | `40` | Upper bound for `top_k`. Requests above it are lowered to it, and values below 1 are raised to 1 (the server rejects them). |
| `LIGHTRAG_MCP_MAX_CHARS` | `48000` | Longest retrieved-passages result, in characters. Claude Code saves a tool result over 50,000 characters to a file and shows the agent a 2 KB preview, so the default keeps results inline there. |

**Choosing `LIGHTRAG_MCP_MAX_TOP_K`.** The default equals LightRAG's own default `TOP_K`, so
a stock server sees nothing it would not use anyway. On a server with a trimmed token
budget, a large `top_k` fills the budget with knowledge-graph entities and relations and
can leave no document chunks, and without chunks there are no references or citations. With
`MAX_TOTAL_TOKENS` of about 8000 (sized for an 8K-context model), for example, `top_k=40` can
return no chunks for a query where `top_k=10` returns chunks with references. On such a
server set `LIGHTRAG_MCP_MAX_TOP_K=15` or lower.

## Register it with a client

Use absolute paths. The interpreter should be the one you installed the requirements
into.

Claude Code:

```bash
claude mcp add zotero-lightrag \
  -e LIGHTRAG_BASE_URL=http://localhost:9621 \
  -e LIGHTRAG_API_KEY=your-lightrag-api-key \
  -e ZOTERO_METADATA=/path/to/zotero_metadata.json \
  -- /path/to/venv/bin/python /path/to/LightRAG/zotero-toolkit/mcp/lightrag_mcp_server.py
```

Claude Desktop (`claude_desktop_config.json`) and other clients that use an `mcpServers`
block:

```json
{
  "mcpServers": {
    "zotero-lightrag": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/LightRAG/zotero-toolkit/mcp/lightrag_mcp_server.py"],
      "env": {
        "LIGHTRAG_BASE_URL": "http://localhost:9621",
        "LIGHTRAG_API_KEY": "your-lightrag-api-key",
        "ZOTERO_METADATA": "/path/to/zotero_metadata.json",
        "LIGHTRAG_MCP_MAX_TOP_K": "40"
      }
    }
  }
}
```

The API key ends up in the client's config file in plain text. Keep that file private.

## Usage notes

- Prefer `search_library` with the default `answer=False` and let the calling agent
  synthesize from the passages. The passages are the primary source, and skipping answer
  generation saves an LLM call on the server.
- Trust the compiled citation list. Author, year or venue details an answering model
  writes inline are LLM output and can be wrong even when the claim itself is grounded.
- Parallel tool calls share the server's LLM and reranker concurrency (`MAX_ASYNC` and
  related settings). Beyond that they queue, and a long queue can exceed
  `LIGHTRAG_MCP_TIMEOUT`.
- `zotero://select/library/items/KEY` selects the item in your personal library. Items in
  Zotero group libraries need a `zotero://select/groups/<id>/items/KEY` link, which the
  proxy does not generate.
- A failed request (an HTTP 4xx or 5xx, a timeout, a refused connection) comes back as an
  MCP tool error whose text names the cause, for example `LightRAG /query failed: Server
  error '500 Internal Server Error' for url '…/query'`, not as an empty result. For an
  HTTP error the server's explanation follows on a `Server response:` line: the string
  `detail` of LightRAG's JSON error body, or else the raw body, cut at 500 characters. A
  409 from `/documents/text`, for instance, says which document already exists or that a
  scan or deletion is running. The error is raised as the SDK's `ToolError`, because mcp
  2.1 and later hide the text of any other exception from the client. `health()` never
  raises: it reports the error as its result.

## Tests

The tests run offline. A stdlib HTTP server fakes the LightRAG API, and one test launches
the proxy over stdio as an MCP client would. From the `zotero-toolkit/` directory:

```bash
python3 -m venv /tmp/zl-mcp-venv
/tmp/zl-mcp-venv/bin/pip install -r mcp/requirements.txt pytest
/tmp/zl-mcp-venv/bin/python -m pytest tests/mcp
```

The tests cover citation compilation (every source listed, server order kept, key
resolution, fallbacks), stripping of invented references (including keeping `### Sources
of error`), mode validation, the `top_k` clamp and `passages`, the `X-API-Key` header, the
`/query/data` fallback and its timeout, error propagation with the server's explanation,
unique note names for `add_note` (the fake refuses a repeated name with 409, as LightRAG
does), folding and the paper lookup (key confirmation, cache, ranking, cut lists, near
matches), both `paper_passages` paths and their local file line, the header strip
(checked against `corpus/build_corpus.py` when it imports), the section-heading cleanup,
duplicate passages, and the stdio tool listing
with text-only results. `test_mcp_paper_tools.py` covers `paper_equation` and `paper_table`
(which paper and file, the helper's failures, the wording of every answer, stdio) against a
stand-in helper; `test_mcp_pdf_tools.py` covers the helper (numbers, LaTeX pairing, citing
sentences, table cells and the order split, and both commands on invented PDFs with a
stand-in `llama-server`, including a hung or dead server and the server's death with its
caller). The PDF-level tests are skipped without PyMuPDF.

Do not add an `__init__.py` to `tests/mcp/`. A test package named `mcp` would shadow the
`mcp` SDK.
