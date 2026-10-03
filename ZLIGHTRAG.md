# zlightrag

`zlightrag` is the default branch of this fork of
[LightRAG](https://github.com/HKUDS/LightRAG). It is LightRAG **v1.5.7** plus the
changes listed below, and a `zotero-toolkit/` directory. Together they run LightRAG
as a research knowledge base over a [Zotero](https://www.zotero.org) library: answers
cite real bibliographic entries, reference-list text is kept out of the answer
context, and retrieval budgets can be held to what a small LLM context window takes.

The rest of the code is upstream LightRAG v1.5.7, unchanged, and its documentation,
starting with [README.md](./README.md), applies as written.

## Changes to LightRAG

| Change | Switch | Default |
|---|---|---|
| [Reference lists compiled from Zotero metadata](#reference-lists-compiled-from-zotero-metadata) | none; metadata file via `ZOTERO_METADATA` | always active |
| [Ceiling on per-request retrieval budgets](#ceiling-on-per-request-retrieval-budgets) | `ENABLE_QUERY_BUDGET_CEILING` | `false` |
| [Query journal](#query-journal) | `QUERY_JOURNAL_FILE` | unset (off) |
| [Bibliography chunk filter](#bibliography-chunk-filter) | `DROP_BIBLIOGRAPHY_CHUNKS` | `false` |
| [PostgreSQL edge removal in plain SQL](#postgresql-edge-removal-in-plain-sql) | none | always active |

### Reference lists compiled from Zotero metadata

LightRAG gives the answering LLM only a file path per source and leaves the reference
section to it, so the list it writes can hold raw file names, invented entries, or
nothing. On this branch:

- `/query` and `/query/stream` remove any references section the LLM wrote and
  append a `### Sources` block that lists every retrieved source in `reference_id`
  order. The section is removed even when retrieval returned no sources, and nothing
  is appended then. Streamed answers hold back a short tail, so a heading split
  across chunks is never sent. A heading such as `### Sources of error` is kept as
  prose.
- Each reference gains a `citation` field in `/query`, `/query/stream` and
  `/query/data`, and the LLM's reference list shows a short citation instead of the
  file name, capped to the file name's length so the prompt does not grow.
- The answer text is not rewritten when a request sets `include_references=false`,
  nor for `only_need_context` / `only_need_prompt` output.

Citations resolve through `zotero_metadata.json`, which the corpus builder in
`zotero-toolkit/corpus/` writes. Corpus files named `<ZoteroKey>__<slug>.md` resolve
to authors, year, title, publication and DOI. The file is read from the server's
working directory, or from the path in `ZOTERO_METADATA`, and reloaded when it
changes. Without it, or for a file whose key it lacks, the citation falls back to the
de-slugged file name; the `### Sources` block is still appended. Code:
`lightrag/zotero_citations.py`.

### Ceiling on per-request retrieval budgets

`TOP_K`, `CHUNK_TOP_K`, `MAX_ENTITY_TOKENS`, `MAX_RELATION_TOKENS` and
`MAX_TOTAL_TOKENS` are defaults upstream: a request that sends its own `top_k`,
`chunk_top_k` or `max_*_tokens` overrides them, and the WebUI sends all five with
every query. With `ENABLE_QUERY_BUDGET_CEILING=true` they are also per-request maxima
on `/query`, `/query/stream` and `/query/data`: an omitted value gets the configured
one, a value at or below it is honoured, and a larger one is lowered to it, with one
INFO log line per such request. Off, nothing changes. The switch is reported in the
`/health` configuration and documented in `env.example`.

### Query journal

With `QUERY_JOURNAL_FILE` set to a path, every `/query` and `/query/stream` request
appends one JSON line to that file: time, endpoint, client address and user agent
(which tells WebUI queries from scripted ones), the query, the retrieval settings as
applied (after the ceiling above), the high- and low-level keywords retrieval searched
with, the answer exactly as the client received it (with its `### Sources` block),
the sources, the response time, and whether the answer completed. Streamed lines are
relayed unchanged and the entry is written when the stream ends, so an answer that
failed or that the client abandoned is recorded too, as incomplete. A file that cannot
be written logs a warning and never fails the query. Unset, the routes behave as
upstream. The path is reported in the `/health` configuration.

### Bibliography chunk filter

With `DROP_BIBLIOGRAPHY_CHUNKS=true`, chunks whose text is a reference list are
removed from every query's chunk candidates before reranking, `chunk_top_k` and token
truncation, in every retrieval mode. Detection reads the chunk text, so it covers every
chunker and documents ingested before the switch was turned on; stored data is not
changed. While it is on, the query-answer cache key carries it, so answers cached
without the filter are not served with it. The switch is reported in the `/health`
configuration. Rules, evidence and known misses:
[docs/design/BibliographyChunkFilter.md](./docs/design/BibliographyChunkFilter.md).

### PostgreSQL edge removal in plain SQL

`PGGraphStorage.remove_edges` deletes edges with one plain-SQL statement per batch on
the graph's edge tables, endpoints bound as arrays, instead of one Cypher `DELETE` per
edge. Below Apache AGE 1.8.0 every Cypher `DELETE` scans each edge label table, so
the old path slowed as the graph grew. What is removed is unchanged: every edge
between each pair, of any label and in either direction, with the vertices kept.

### Tests

The unit tests for these changes, including the upstream tests the branch modifies,
are marked `offline`, so upstream's
[.github/workflows/tests.yml](./.github/workflows/tests.yml) runs them; on this branch it
also triggers on pushes and pull requests to `zlightrag`. The exception is the Apache
AGE integration test for the edge removal,
`tests/kg/postgres_impl/test_postgres_remove_edges_age.py`, which needs
`--run-integration` and a live AGE server; no workflow runs it.

## `zotero-toolkit/`

Scripts that sit beside LightRAG rather than inside it: a corpus builder that turns a
Zotero library into citable Markdown and writes `zotero_metadata.json`, an MCP server
that answers with citations, PostgreSQL maintenance for chunk references left behind
by out-of-band deletions, and primary/fallback fronts for the LLM and reranker
servers. The toolkit neither imports nor patches the LightRAG code and also works
with a stock LightRAG Server. See [zotero-toolkit/README.md](./zotero-toolkit/README.md).
Its offline tests run from
[.github/workflows/zotero-toolkit.yml](./.github/workflows/zotero-toolkit.yml).

## How this branch follows upstream

The branch is rebased onto upstream release tags: when a new LightRAG release is
tagged, the changes above are replayed onto it and the branch is updated to the
result. Its history is therefore rewritten at each such step; pin a commit if you
need a fixed point.

Nothing here is proposed to or maintained by the LightRAG project. Three of the
changes are also kept on this fork as topic branches based on upstream's `main`:
`feat/query-token-budget-ceiling`, `feat/query-drop-bibliography-chunks` and
`perf/postgres-remove-edges-sql`.
