# zlightrag

`zlightrag` is the default branch of this fork of
[LightRAG](https://github.com/HKUDS/LightRAG). It is upstream LightRAG's `main` as of
2026-10-14 (commit `aaebfe1bb`, which reports version **1.5.8**; upstream has tagged no
1.5.8 release) plus the changes listed below, and a `zotero-toolkit/` directory. Together they run LightRAG
as a research knowledge base over a [Zotero](https://www.zotero.org) library: answers
cite real bibliographic entries, reference-list text is kept out of the answer
context, and retrieval budgets can be held to what a small LLM context window takes.

The rest of the code is upstream LightRAG at that commit, unchanged, and its documentation,
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
| [Author-year and citation legs](#author-year-and-citation-legs) | `METADATA_CHUNK_TOP_K`, `CITATION_HOP_TOP_K` | `0`, `0` (off) |
| [Named-work pin and notice](#named-work-pin-and-notice) | `NAMED_WORK_PIN`, `NAMED_WORK_NOTICE` | `false`, `false` |
| [Low-level keyword fallback](#low-level-keyword-fallback) | `LL_KEYWORDS_FALLBACK` | `false` |
| [Per-request rerank floor](#per-request-rerank-floor) | `min_rerank_score` request field | the server's `MIN_RERANK_SCORE` |
| [Lexical document-frequency cap](#lexical-document-frequency-cap) | `LEXICAL_DF_CAP_PCT` | `2` |
| [Strict entity types](#strict-entity-types) | `ENTITY_TYPE_STRICT` | `false` |
| [Symbol entity names](#symbol-entity-names) | `DROP_SYMBOL_ENTITIES` | `false` |
| [Apparatus entity names](#apparatus-entity-names) | `DROP_JUNK_ENTITIES` | `false` |
| [Entity name fold](#entity-name-fold) | `ENTITY_NAME_FOLD`; lists and aliases via `ENTITY_NAME_FOLD_LISTS`, `ENTITY_NAME_ALIASES` | `false` |
| [Sidecar relations](#sidecar-relations) | `SIDECAR_RELATIONS` | `all` (upstream behaviour) |
| [Paper filter in the graph view](#paper-filter-in-the-graph-view) | none; reference lists via `ZOTERO_REFERENCES` | always active |
| [Paper lookup that forgives typography, and one paper's passages](#paper-lookup-that-forgives-typography-and-one-papers-passages) | none | always active |
| [PostgreSQL edge removal in plain SQL](#postgresql-edge-removal-in-plain-sql) | none | always active |
| [Resume purge keeps a failed document retryable](#resume-purge-keeps-a-failed-document-retryable) | none | always active |
| [One unusable multimodal analysis degrades its item](#one-unusable-multimodal-analysis-degrades-its-item) | none | always active |
| [Changed text of a processed document](#changed-text-of-a-processed-document) | `SCAN_DIVERT_CHANGED_PROCESSED` | `false` |
| WebUI query defaults for new browser profiles: `top_k` 10, `chunk_top_k` 12, entity/relation tokens 1500, total tokens 7000 (`lightrag_webui/src/stores/querySettings.ts`) | none | always active |

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

### Apparatus entity names

With `DROP_JUNK_ENTITIES=true`, an extracted entity whose name is bibliography or
document apparatus rather than subject matter is dropped, and so is every relation that
names one (`junk_entity_class` in `lightrag/entity_name_guard.py`). A paper corpus hands
the extraction LLM reference lists, in-text citations and captions, and the model turns
them into entities whatever the type guidance says. Each becomes a hub linking unrelated
papers, and because the local leg ranks an entity's relations by the summed degree of
their endpoints, those hubs crowd real neighbours out of the relation budget. Dropped:

- citations: `et al.` forms, author lists, an author and a year, URLs and DOIs, journal
  abbreviations unless typed as something other than a venue, and the titles of journals
  the library publishes in when typed as a publication or carrying a journal word;
- persons: a name typed as a person, or shaped like a surname with initials (`Doe, J.`);
  `J. Doe` only when `Doe` is one of the library's authors;
- numbered labels of figures, tables, equations, sections, chapters, appendices,
  theorems and the like;
- generic placeholder words (`Concept`, `Table Name`);
- names with no letter at all.

An entity is judged on its name and on the type the LLM wrote, before
`ENTITY_TYPE_STRICT` maps it, and some types protect a shape: an author-and-year name
typed as an event, mission, dataset, model, software or instrument stays (a model named
after its authors), as does a URL under a dataset, organization, software, model or
mission type, and location, mission, instrument and similar types keep a person-shaped
name (`N. America`). The two rules
that need the library's author surnames and journals read them from the Zotero metadata
(`lightrag/zotero_citations.py`); without it they never fire. Names of one or two
characters are left to `DROP_SYMBOL_ENTITIES`. Applied as each chunk's extraction output
is parsed, so existing apparatus entities stay until deleted. Reported in the `/health`
configuration. Off by default.

### Entity name fold

With `ENTITY_NAME_FOLD=true`, every entity name an extraction produces is mapped to the
spelling the graph already uses for the same fold key, or becomes that spelling when
the key is new (`lightrag/entity_name_fold.py`). The extraction LLM spells one entity
several ways across papers, and the merge keys on the exact string, so without it a
concept's relations scatter over several nodes. The map is read from the Postgres/AGE
graph in the background after the storages initialise (computed in a worker thread, so
the server keeps answering) until `ENTITY_NAME_FOLD armed: N name keys, A aliases, lists
<source>, from M nodes in Xs` is logged. Extraction and the deletion and kept-summary
rebuilds wait for it, for at most 300 s after startup; past that an error is logged and
they go on unfolded until the map arms. With other graph storages names pass unchanged.
Off by default.

The key folds case, spacing and punctuation (NFKC, casefold, letters and digits of every
script) and the plural of the last word: regular `-s`/`-es`/`-ies` plurals, acronym
plurals (`DEMs`), a `(s)` suffix, Latin/Greek and English irregulars (`spectra`,
`matrices`, `axes`, `leaves`), and `-ics` only after another word. Kept apart: keys
under four characters, an all-capitals acronym and a word up to five characters
(`TIDE`/`tide`), all-capitals words inside a mixed-case name (`GPS`), words ending
`-ss`/`-ous` or `-is`/`-us`/`-as`/`-os` unless listed, and anything with different
digits. A name holding an operator, comparison or prime (`=`, `+`, `^`, `<`, `>`, `|`,
`′`, `<sup>`, an apostrophe used as a prime) gets a separate symbolic key that keeps
those symbols, so `x + y` and `x - y`, or `k2` and `k2′`, stay apart; such names are
never plural-folded. Each key's spelling is the member with the smallest
`canonical_rank`: apparatus names (see `DROP_JUNK_ENTITIES`) never, then spellings
without markup, stray whitespace or CamelCase glue, then the best-connected, a singular
before a plural, and code-point order last, so the choice does not depend on scan order.

The code carries generic word lists only. `ENTITY_NAME_FOLD_LISTS` names a JSON file of
a deployment's own exceptions (surnames and place names that look like plurals, words
that do not fold when they are the whole name, `-is`/`-us`/`-as`/`-os` plurals,
irregulars of its field), merged into the built-in lists; the schema is in the module
docstring. `ENTITY_NAME_ALIASES` names a CSV of `variant,canonical` rows, keyed by the
variant's fold key and consulted before the map, for spellings joined by rules the key
does not know, such as a bulk merge's; rows whose two names share a key are skipped.
An alias whose canonical's own key is aliased follows that chain to its end, and rows
on a loop are dropped with a warning; an alias returns the spelling the map uses for its
canonical's key. Both files are read when the map is built, relative paths against the
server's working directory. A missing or malformed lists file arms the map with the
built-in lists (an error, and the armed line reads `lists defaults (FALLBACK: <why>)`,
so a deploy check that expects the file's path fails); an unreadable alias file arms it
without aliases (an error); alias rows with fewer than two fields or an empty field are
skipped and counted. The startup graph query is retried three times. `/health` shows
both paths (absolute) and `entity_name_fold_state`: `pending`, `armed` or
`failed:<why>`, with the lists source, alias and key counts.

Legacy fallback: names the key before the plural and symbol rules (`legacy_fold_key`)
joined may get different keys now (an all-capitals plural next to its Titlecase form).
The map also keeps, for every non-symbolic name, its legacy key's canonical (that of the
best-ranked member), used when a name's own key is in neither the aliases nor the map;
the key is then registered to that canonical. A key new to the map takes the
best-ranked of the chunk's spellings of it, not the first one seen.

Applied where results leave `extract_entities`, so the write-ahead anchors and the merge
name the same nodes, and as an alias in the knowledge rebuild, which files each cached
record under its original and its canonical name (and its legacy key's canonical when
that is a third name). Changing the key rules or the lists
changes keys: existing variant nodes are not merged by the switch itself.

### Sidecar relations

`SIDECAR_RELATIONS=none` keeps a table/equation/drawing sidecar entity but no longer
links it to every entity extracted from its chunk with an "associated with, contained
in" edge (upstream behaviour, `all`). On a paper corpus those edges were about a tenth
of the graph, every one generic.

### Paper filter in the graph view

The WebUI's graph view gains a paper picker beside the label search. Search by DOI,
Zotero key, or words of the title, authors or year; only papers in the knowledge base
are listed, with how many of their references are. Up to ten papers can be picked at
once (picking one again removes it; each shows as a chip under the picker), and the
button beside the picker cycles through three graphs:

- **Papers**: the entities and relations the papers' documents contributed. With
  several papers, each entity names the ones that contributed it in a
  `contributed_by` property.
- **Shared with their references**: only the entities that a picked paper and at
  least one library paper it cites both contributed, each listing those papers in a
  `shared_with` property, and the relations the shown documents contributed between
  them. A picked paper that another picked paper cites counts as a reference.
- **In common** (two or more papers): only the entities at least two of the picked
  papers contributed.

A paper's own relations leave many of its entities unconnected: on one corpus, 0.68
relations per entity, with a third of the entities isolated. A second button sets how
much context the view takes from the rest of the graph (remembered per browser):

- only the papers' own relations;
- also every other relation the graph holds between the shown entities (the default;
  on that corpus a quarter of a paper's entities stay isolated, and a tenth of the
  shared view's);
- also linking entities: entities outside the view that touch at least two shown
  ones, most connections first, at most a quarter as many as are shown, with their
  relations to the shown entities.

What the papers did not contribute carries a `paper_context` property and is drawn
apart (blue relations, faded entities). On the Postgres/AGE graph storage both lookups
run as plain SQL on the label tables (about 0.1 s for a 400-entity view; the Cypher
edge lookup they replace took about 11 ms per pair); other storages use the generic
graph calls.

A paper is its DOI, so every attachment with that DOI counts; a paper without one is
its Zotero key. What a document contributed is read from LightRAG's per-document
`full_entities` / `full_relations` rows, keeping only names that `entity_chunks` /
`relation_chunks` still trace to one of its chunks (a prune or merge can leave stale
names in those rows). Which library papers a paper cites comes from
`zotero_references.json`, which `zotero-toolkit/corpus/build_references.py` builds from
OpenAlex reference lists. It is read from the server's working directory, or the path
in `ZOTERO_REFERENCES`, and reloaded when it changes; without it the shared view is
empty and says so. The picker sets the graph label to `paper:<DOIs>`,
`paper+refs:<DOIs>` or `paper+common:<DOIs>` (the papers separated by spaces), so
refresh, retries and the label history work as for an entity. Endpoints:
`GET /graph/papers/search?q=` and
`GET /graphs/paper?paper=<id>[&paper=<id>...]&mode=paper|shared|common&context=own|kb|linked`
(`context` defaults to `own`). Code: `lightrag/zotero_papers.py`,
`lightrag_webui/src/components/graph/PaperFilter.tsx`, `lightrag_webui/src/lib/paperFilter.ts`.

### Paper lookup that forgives typography, and one paper's passages

`GET /graph/papers/search` matches with both the query and each paper's title,
authors, year, DOI and key folded the same way: accents and other combining marks
dropped, the letters Unicode decomposition keeps whole spelled in ASCII (ı ł ø đ ð ß
æ œ þ), every dash and the soft hyphen read as `-`, apostrophes and primes deleted,
lowercased. A query word, without its leading and trailing punctuation (brackets, `*`,
straight and curly quotes, guillemets; a token left with no letter or digit, such as the
` - ` of `Author - Year - Title` or an `&`, is no word; the `et al` after a name, as in
`Doe et al. (2011)`, is dropped), matches when
it occurs in the folded text, when its hyphen-less form occurs in the text without
hyphens, or when every part of it split on `-` or `/` occurs (a DOI query is matched
whole). So `trefoil-fo`, `trefoilfo` and `orchards` find titles typeset with U+2010 hyphens
or a curly apostrophe. A token shaped like a BibTeX key (`doePruningOrchardsAppleTrees2001`)
is read as its family name, its camel-case title words (matched against the text with
everything but letters and digits removed) and its year. The MCP proxy's `find_paper`
carries the same rule; `zotero-toolkit/tests/mcp/paper_search_cases.json` holds the
fold, word and lookup cases that both test suites run. `keys=K1,K2,...` (at most
50 storage keys) lists the indexed papers carrying any of them in one request, with
or without `q`; an indexed key without metadata is listed too. The scan runs off the
event loop over haystacks folded once per document-list or metadata change, and the
processed-document list is read at most every five minutes, one read shared by
concurrent requests and later refreshes run behind the request.

`POST /graph/papers/passages` with `{"keys": [...], "question": "...", "top_n": 3}`
(1 to 10 storage keys, all copies of one work; `top_n` 1 to 8) returns that work's
best passages for the question: `keys` (those in the knowledge base), `n_chunks` (the
chunks of the work's longest copy, so `chunk_order_index` + 1 of `n_chunks` holds), and
`passages` with `key`, `file_path`, `chunk_id`, `chunk_order_index`, `content`,
`content_headings` and `score`. A work of up to 60 chunks offers all of them; a longer
one the 40 closest to the question by the stored chunk vectors (ranked among the work's
own chunk ids, in SQL on Postgres, so 40 rows come back however long the work is) and the 40 best by Postgres full-text rank over the
work's chunk ids (other storages: the vector leg alone). Reference lists drop out under
`DROP_BIBLIOGRAPHY_CHUNKS`, a text two copies share counts once, and the server's
reranker orders the candidates (at most 80) against the bare question; without a
reranker, or when it fails, the vector similarity does (logged). The first chunk, with
its metadata header, competes like any other: `/query`'s author-year leg and the
named-work pin, which always bring it, are not involved. Both routes answer with the
header `X-Paper-Search: 2` (a FastAPI request-validation 422 carries none). Code:
`lightrag/zotero_papers.py`, `PGKVStorage.rank_chunks_by_text`,
`PGVectorStorage.rank_ids_by_vector`.

### PostgreSQL edge removal in plain SQL

`PGGraphStorage.remove_edges` deletes edges with one plain-SQL statement per batch on
the graph's edge tables, endpoints bound as arrays, instead of one Cypher `DELETE` per
edge. Below Apache AGE 1.8.0 every Cypher `DELETE` scans each edge label table, so
the old path slowed as the graph grew. What is removed is unchanged: every edge
between each pair, of any label and in either direction, with the vertices kept.

### Resume purge keeps a failed document retryable

Two gaps in upstream's fail-closed purge could leave a reprocessed document that every
retry and every delete refused (`RecoveryAnchorMissingError`), with nothing in the
graph to protect. After the resume purge in `_purge_stale_extraction_if_resuming` has
completed, the same write that retires its journal sets `kg_write_state` back to
`pre_graph` (the document owns nothing in the graph again); upstream kept
`graph_mutation_started`, so a failure before the new run's Phase 0, such as a
chunk-embed timeout, left no anchors, no journal and no proof. And when the resume
purge itself raises, the stored `kg_purge` journal is copied back into the in-memory
status before the FAILED write, which upstream let erase it. See
[docs/design/PurgeRecoveryContract.md](./docs/design/PurgeRecoveryContract.md).

### One unusable multimodal analysis degrades its item

Upstream fails the whole document when one drawing, table or equation analysis is
still invalid JSON after its one retry, so a single bad formula sinks a book of
hundreds, and a model run with a fixed seed fails it again on every retry. On
this branch that item is degraded instead: its `llm_analyze_result` is a success
with `degraded: true`, a deterministic name (`equation <item id>`), and its source
content (the LaTeX, the table body, or the image caption), so it still becomes an
ordinary multimodal chunk and entity. A degraded result is never cached, so the next
run of the document asks the model again. Each degraded item is logged as a WARNING,
and the document's `metadata.mm_degraded` counts them. Past a ceiling of
max(3, 2 % of the document's analyzed items) the cause is systemic, not one bad
formula, so the document fails as upstream would (`N of M multimodal analyses still
invalid after their retry`) instead of becoming a book of source-only stubs. Any other
analysis error (a model call that fails, a configuration limit, cancellation) still
fails the document.
The error message says `unparseable JSON` when nothing could be parsed, rather than
naming the first required field, and every such message ends with the response
length and `finish_reason=length` when the output budget cut the response off.

### Changed text of a processed document

A scan archives a file whose document is already processed into `__parsed__/` unread,
so a document first ingested as a stub never takes the full text that later arrives
under the same name. With `SCAN_DIVERT_CHANGED_PROCESSED=true`, a scanned `.md`, `.txt`
or `.markdown` file of a processed document whose stored text it no longer matches --
at least twice as long or at most half as long, and at least 2,000 characters
different -- is moved to `<input_dir>/__changed__/` with a scan warning instead. The
stored document is never touched: delete it and move the file back to replace it. Off,
the scan does no extra I/O. Reported in the `/health` configuration. Code:
`_content_materially_changed` in `lightrag/api/routers/document_routes.py`.

### Tests

The unit tests for these changes, including the upstream tests the branch modifies,
are marked `offline`, so upstream's
[.github/workflows/tests.yml](./.github/workflows/tests.yml) runs them; on this branch it
also triggers on pushes and pull requests to `zlightrag`, as does upstream's
[.github/workflows/linting.yaml](./.github/workflows/linting.yaml). The exception is the Apache
AGE integration test for the edge removal,
`tests/kg/postgres_impl/test_postgres_remove_edges_age.py`, which needs
`--run-integration` and a live AGE server; no workflow runs it.

## `zotero-toolkit/`

Scripts that sit beside LightRAG rather than inside it: a corpus builder that turns a
Zotero library into citable Markdown, with display equations written as LaTeX and merged
table rows split, and writes `zotero_metadata.json`; an MCP server that answers with
citations and reads one paper's numbered equations and tables from its PDF; PostgreSQL
maintenance for chunk references left behind by out-of-band deletions, and audits of
the corpus's text layers; primary/fallback fronts for the LLM and reranker servers; and
an on-demand OCR server that takes the GPU only while LightRAG has no work for it. The toolkit neither imports nor patches the LightRAG code and also works
with a stock LightRAG Server. See [zotero-toolkit/README.md](./zotero-toolkit/README.md).
Its offline tests run from
[.github/workflows/zotero-toolkit.yml](./.github/workflows/zotero-toolkit.yml).

## How this branch follows upstream

The changes above were rebuilt onto the v1.5.7 release tag on 2026-09-26, and the
branch has since taken upstream `main` once, by merge (`aaebfe1bb`, merged
2026-10-19). Its history is no longer rewritten: a rebuild onto a new base is published
as a merge commit whose tree is the rebuild's, so any commit you pin stays on the
branch.

Nothing here is proposed to or maintained by the LightRAG project.
