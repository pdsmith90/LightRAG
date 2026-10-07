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
| [Per-work chunk cap](#per-work-chunk-cap) | `MAX_CHUNKS_PER_DOC` | `0` (off) |
| [Lexical retrieval leg](#lexical-retrieval-leg) | `LEXICAL_CHUNK_TOP_K` | `0` (off) |
| [PostgreSQL edge removal in plain SQL](#postgresql-edge-removal-in-plain-sql) | none | always active |
| [Named-work pin and notice](#named-work-pin-and-notice) | `NAMED_WORK_PIN`, `NAMED_WORK_NOTICE` | `false`, `false` |

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

### Per-work chunk cap

With `MAX_CHUNKS_PER_DOC=N`, a query's final context holds at most N chunks of one
work. The cap applies after reranking and the rerank-score floor, before
`chunk_top_k` and token truncation; the reranker then returns every candidate in
score order, so the slots a capped work would have taken go to the next works.
A work is one document, or every copy of one paper filed under several Zotero items:
the same normalised title when that is long enough to name a single work, else the
same DOI (`lightrag/zotero_citations.py`, `work_key`). The setting is part of the
query-answer cache key, reported in the `/health` configuration, and documented in
`env.example`. Off by default.

### Lexical retrieval leg

Dense retrieval misses queries made of names, years and acronyms: an author and a
year are a few tokens of one chunk, and an acronym a paper never spells out is
invisible to it. With `LEXICAL_CHUNK_TOP_K=K`, `mix` and `naive` queries also take up
to K chunks from a PostgreSQL full-text search and interleave them with the vector
chunks, ahead of the bibliography filter and the reranker, which decide what is
kept. Each query term (English stemming, stop words dropped) weighs
`ln(N / (df + 1))`; candidates are the chunks holding one of the three rarest terms,
and each scores the summed weight of every query term it contains, so a query of
common words only adds nothing. Reference lists are removed before the cut to K when
the bibliography filter is on. Needs a GIN index on
`to_tsvector('english', content)` of `LIGHTRAG_DOC_CHUNKS`; without it the leg logs
once and stays off. Code: `PGKVStorage.lexical_search` in
`lightrag/kg/postgres_impl.py`, `_get_lexical_context` in `lightrag/operate.py`. Part
of the query-answer cache key; reported in `/health`. Off by default.

### Author-year and citation legs

A query that names a paper by surnames and year ("Okafor and Lindqvist 2019 ...") is
hard for every text leg: the paper's author line is one chunk and its method words
are others, so no single chunk of it matches the whole query, while the papers that
cite it match well. With `METADATA_CHUNK_TOP_K=N`, `mix` and `naive` queries resolve
the surnames and year against the Zotero metadata (`find_works` in
`lightrag/zotero_citations.py`: a year in the query must match; without one a single
surname counts only when it names at most two works) and add up to N chunks of the
named works -- each work's first chunk, then its best full-text matches -- ahead of
the vector chunks. With `CITATION_HOP_TOP_K=M`, the vector and lexical candidates are
scanned for author-year citations of library works (`cited_works_in`), and the M
most-cited works contribute two chunks each; this reaches a paper the corpus names
only by citation, such as the origin of an acronym the paper itself never spells
out. Both need a text-chunk storage with `get_chunks_for_works` (PostgreSQL). Code:
`_get_metadata_context` in `lightrag/operate.py`. Part of the query-answer cache key;
reported in `/health`. Off by default.

`find_works` returns every work tied at the best score, past its cap if need be, a tie
broken by the title words a work shares with the query and only then by key: two
authors who published several papers in one year would otherwise lose the right one
to an alphabetical cut. When no work matches the query's year, the adjacent years
stand in without the year point (a journal's online-first and print years differ by
one). A surname is the first token of a creator string, continued only through
particles (`van der Lind Tamás`), never the given names that follow it.

### Named-work pin and notice

The author-year leg brings a named work's chunks into the candidates, but the
reranker, the score floor, `chunk_top_k` and the token budget can still drop them
when the papers citing the work match the question better than the work itself.
With `NAMED_WORK_PIN=true`, whenever the query carries a year, up to two named works
keep up to `MAX_CHUNKS_PER_DOC` (else two) chunks each at the front of the context,
through the rerank floor and both cuts (`pin_named_work_chunks` in
`lightrag/utils.py`). When the query names one work beyond doubt -- a clear winner
on score or on title words shared with the query (`specific_named_work`) -- only that
work is pinned; otherwise every tied work is and the reranker picks among them. With
`NAMED_WORK_NOTICE=true`, `/query` and `/query/stream` open the answer with one
sentence per named work that is not among the sources, saying whether it is in the
knowledge base at all (`named_work_notices` in `lightrag/zotero_citations.py`), and
return them as `notices`; nothing is said when the query names no year, when the
specifically named work is a source, or, for an ambiguous naming, when any tied work is. Both off by default; the pin needs `METADATA_CHUNK_TOP_K`.

### Low-level keyword fallback

With `LL_KEYWORDS_FALLBACK=true`, a `local`, `hybrid` or `mix` query whose keyword
extraction returned no low-level keywords runs its entity leg on the high-level
keywords instead of skipping it (`_fallback_low_level_keywords` in
`lightrag/operate.py`). Off by default.

### Per-request rerank floor

`QueryParam.min_rerank_score` (and the same field on `/query`, `/query/stream` and
`/query/data`) replaces the server's `MIN_RERANK_SCORE` for one request, so an
evaluation can sweep the floor without restarts. Part of the cache key when set.

### Lexical document-frequency cap

`LEXICAL_DF_CAP_PCT` (default 2) is the share of the chunks above which the lexical
leg treats a query term as common: it keeps the floor weight and never generates
candidates. Raise it when a field's central authors are cited in more than that
share of the chunks.

### Strict entity types

With `ENTITY_TYPE_STRICT=true`, an extracted entity type that the active entity-type
guidance does not list (its `- Type: ...` lines) is stored as `other`
(`_configure_entity_type_allowlist` / `_normalize_and_validate_entity_type` in
`lightrag/operate.py`). The extraction LLM otherwise invents types freely. Ignored,
with a warning, when the guidance lists no types. Off by default.

### Symbol entity names

With `DROP_SYMBOL_ENTITIES=true`, an extracted entity whose name is a lone symbol or
variable -- one character (`x`, `θ`, `7`), or two that are not alphanumeric or are
digits (`a,`, `12`) -- is dropped, and so is every extracted relation that names one
(otherwise the merge would recreate the endpoint as an untyped node). Alphanumeric
pairs such as `J2`, `Io` or `L1` are kept (`is_symbol_entity_name` in
`lightrag/operate.py`). The extraction LLM turns formula variables into entities
whatever the prompt says, and on a paper corpus the one-letter hubs collect
thousands of unrelated edges. Applies to new extractions; existing symbol entities
stay until deleted. Off by default.

### Sidecar relations

`SIDECAR_RELATIONS=none` keeps a table/equation/drawing sidecar entity but no longer
links it to every entity extracted from its chunk with an "associated with, contained
in" edge (upstream behaviour, `all`). On a paper corpus those edges were about a tenth
of the graph, every one generic.

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
