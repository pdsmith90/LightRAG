# mcp — LightRAG MCP server with Zotero citations

`lightrag_mcp_server.py` is a small stdio [MCP](https://modelcontextprotocol.io) server
that lets an agent (Claude Code, Claude Desktop, any MCP client) query a
[LightRAG](https://github.com/HKUDS/LightRAG) knowledge base built from a Zotero library.
It proxies the LightRAG Server REST API and turns the server's retrieved references
into real bibliographic citations using Zotero metadata. The LLM never writes them.

## Tools

| Tool | What it returns |
|---|---|
| `search_library(query, mode="mix", top_k=10, answer=False, include_graph=False)` | Compiled citations **first**, then the retrieved passages as plain text, each headed `[n] Family Year · section` (see [Retrieved passages](#retrieved-passages)). No answer is generated, so it is faster. `include_graph=True` appends the knowledge-graph entities and relations. |
| `search_library(..., answer=True)` | The server's synthesized answer. Any reference section the model wrote is stripped, and a `### Sources (compiled from retrieval)` list is appended. |
| `paper_passages(paper, question)` | The passages of **one** paper that best match `question`, under that paper's citation, then the other papers the question matched (citations only). `paper` is a Zotero key, a DOI, or words that must all occur in the title, authors or year. If no indexed paper or several match, it lists them instead of querying. |
| `find_paper(paper)` | An exact lookup with the same `paper` forms: the matching papers that are indexed, then the Zotero items that match but are not. No retrieval and no LLM. |
| `add_note(text, description="")` | Inserts a text note into the knowledge base (`POST /documents/text`) under a unique document name (see below). |
| `health()` | The server's `/health` status, or `unreachable at <url>: <error>`. |

`mode` is one of `mix` (default: knowledge graph + vectors), `hybrid`, `local`, `global` or
`naive`. Any other value, including LightRAG's `bypass`, falls back to `mix`. `top_k` is
clamped to `1..LIGHTRAG_MCP_MAX_TOP_K` (see below).

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
passages of several same-topic papers apart. The entities and relations follow only
with `include_graph=True`, and the server's reference list is dropped because the
compiled citations replace it. When the result would exceed `LIGHTRAG_MCP_MAX_CHARS`,
the lowest-ranked passages are left out and a closing line says how many. A response in
any other layout is passed through whole after the citations.

### One paper's passages and the paper lookup

`find_paper` asks `GET /graph/papers/search?q=…&limit=100` (the zlightrag fork's paper
lookup, which lists processed documents only) which papers are indexed, and matches the
same words against `zotero_metadata.json`. A metadata entry is reported as not indexed
only after the server, asked for that entry's exact key, does not return it either; one
copy is listed per DOI. When the server finds 100 papers its list may be cut, so the
Zotero-only list is withheld and the answer asks for a narrower lookup. On a stock
LightRAG Server the route answers 404, and the index status is reported as unknown.

`paper_passages` resolves `paper` the same way, then sends a `naive` context-only query
that starts with the paper's first author, year and title, followed by the question, and
keeps only the chunks of that paper's keys. On a stock server it resolves `paper` from
the metadata alone. A server that caps chunks per document (the fork's
`MAX_CHUNKS_PER_DOC`) returns at most that many passages of the paper per call.

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

Dependencies are `mcp` 2.x and `httpx`. The server uses `mcp.server.mcpserver.MCPServer`,
which replaced `mcp.server.fastmcp` in mcp 2.0; mcp 1.x does not have it.

## Configuration

Every setting is an environment variable, set in the MCP client's server entry.

| Variable | Default | Meaning |
|---|---|---|
| `LIGHTRAG_BASE_URL` | `http://localhost:9621` | LightRAG Server base URL. A trailing `/` is ignored. |
| `LIGHTRAG_API_KEY` | *(empty: no header sent)* | Must equal `LIGHTRAG_API_KEY` in the server's `.env`. Sent as `X-API-Key`. |
| `ZOTERO_METADATA` | `~/Zotero/zotero_metadata.json` | Path to the metadata file. Unset or empty means the corpus builder's default output. A relative path resolves against the **working directory the client launches the server in**, which you usually do not control, so use an absolute path. `~` is expanded. |
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
of error`), mode validation, the `top_k` clamp, the `X-API-Key` header, the `/query/data`
fallback and its timeout, error propagation with the server's explanation, unique note
names for `add_note` (the fake refuses a repeated name with 409, as LightRAG does), and the
stdio tool listing.

Do not add an `__init__.py` to `tests/mcp/`. A test package named `mcp` would shadow the
`mcp` SDK.
