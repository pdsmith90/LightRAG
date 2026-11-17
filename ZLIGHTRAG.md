# zlightrag

`zlightrag` is the default branch of this fork of
[LightRAG](https://github.com/HKUDS/LightRAG). It is upstream LightRAG's `main` as of
2026-11-17 (commit `31bff1f60`, which reports version **1.5.8**; upstream has tagged no
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
| [Extraction contract: closed relation types](#extraction-contract-closed-relation-types) | `relationship_types` (and `relationship_inverses`) in the `ENTITY_TYPE_PROMPT_FILE` profile; schema via `python -m lightrag.kg_contract --schema` | absent (off) |
| [Extraction contract validator](#extraction-contract-validator) | `KG_CONTRACT_VALIDATOR` | `false` |
| [Entity name fold](#entity-name-fold) | `ENTITY_NAME_FOLD`; lists and aliases via `ENTITY_NAME_FOLD_LISTS`, `ENTITY_NAME_ALIASES` | `false` |
| [Sidecar relations](#sidecar-relations) | `SIDECAR_RELATIONS` | `all` (upstream behaviour) |
| [Paper filter in the graph view](#paper-filter-in-the-graph-view) | none; reference lists via `ZOTERO_REFERENCES` | always active |
| [Paper lookup that forgives typography, and one paper's passages](#paper-lookup-that-forgives-typography-and-one-papers-passages) | none | always active |
| [PostgreSQL edge removal in plain SQL](#postgresql-edge-removal-in-plain-sql) | none | always active |
| [A document delete keeps cache rows another document lists](#a-document-delete-keeps-cache-rows-another-document-lists) | none | always active |
| [Resume purge keeps a failed document retryable](#resume-purge-keeps-a-failed-document-retryable) | none | always active |
| [A rebuild reads a shared extraction through every chunk that lists it](#a-rebuild-reads-a-shared-extraction-through-every-chunk-that-lists-it) | none | always active |
| [One unusable multimodal analysis degrades its item](#one-unusable-multimodal-analysis-degrades-its-item) | none | always active |
| [Changed text of a processed document](#changed-text-of-a-processed-document) | `SCAN_DIVERT_CHANGED_PROCESSED` | `false` |
| [Upload to Zotero](#upload-to-zotero) | `ZOTERO_UPLOAD` (settings: `ZOTERO_*`, see `env.example`) | `false` |
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
failed or that the client abandoned is recorded too, as incomplete. A failure during a
stream is recorded with the text the client received (upstream's `Internal server error
(error_id: ...)`; the id is in the server log), a failure before the first line with the
error's own text. A file that cannot
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
`lightrag/operate.py`). Off by default. Upstream now applies the same fallback to every
such query (`_resolve_kg_search_queries`), so the switch changes only the keywords
reported with the answer and its cache key, not what is retrieved.

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

### Extraction contract: closed relation types

An entity-type profile (`ENTITY_TYPE_PROMPT_FILE`) may declare `relationship_types`, a
mapping of relation type name to definition in prompt order (`lightrag/kg_contract.py`).
The names are lower-case snake case; each definition is one line, its whitespace
collapsed. A profile that declares it switches the contract on:

- the list is appended to the profile's entity-type guidance, one `name = definition`
  line per type (never `- name:`, which `ENTITY_TYPE_STRICT` would read as an entity
  type), followed by a read-it-back check of each relation's direction, and a sentence
  that prefers `related_to` to a forced type when the list has `related_to`. The guidance
  goes into both the system and the user prompt, as before;
- the JSON extraction system prompt (`SYSTEM_PROMPT_EDITS`) follows the profile's naming
  rule and the text's spelling instead of title case, says never to glue words into one
  identifier, takes exactly one listed type in `keywords` instead of free keywords, makes
  relations directed, and lists an entity only when the text states a relation for it.
  The gleaning turn is sent with this contract system prompt and the unchanged gleaning
  template, which carries none of these rules. A base template that no longer holds a
  replaced passage the expected number of times fails startup rather than sending a
  half-edited prompt.

The contract needs `ENTITY_EXTRACTION_USE_JSON=true`; in text mode a profile with
`relationship_types` is refused at startup. A profile without the key, with no validator
and no new setting, sends exactly the prompts and writes exactly the graph, KV rows and
vectors the base branch did: `tests/extraction/test_kg_contract_inert.py` compares both
modes, with and without a profile, with `ENTITY_NAME_FOLD` unarmed and armed, through a
gleaning turn that re-outputs names and pairs with longer, shorter and equal-length
descriptions, three documents sharing entities, a deletion, a kept-summary deletion and
the re-ingest, against digests recorded from the base branch's own tree. The extraction
cache keys on the prompt, so switching the contract on re-extracts instead of reusing
cached results.

The JSON schema that constrains the extraction requests is set by the process in front
of the LLM server, not here. `python -m lightrag.kg_contract --schema --profile PROFILE`
prints it for a profile: the entity types its guidance lists, read as `ENTITY_TYPE_STRICT`
reads them and in order, plus `Other`, and the profile's relation types, both as enums.
`PROFILE` is what `ENTITY_TYPE_PROMPT_FILE` takes, a file name looked up in
`PROMPT_DIR/entity_type` as the server looks it up (or in the current directory when it
is not there), or a path; the command prints the file it used, and how it found it, on
stderr, so stdout holds the schema only.

Nothing here can make the front apply that schema (a missing or invalid schema file, or
a model the front does not attach it to, leaves `keywords` a free string), so the parser
checks every relation the LLM extracted against the list: the first pass and the gleaning
turn (`check_relation_keywords`). A knowledge rebuild reads the records extraction
stored and checks nothing (*Rebuilding from stored records*).
A keyword that equals a declared type after trimming and case-folding is rewritten to
the declared spelling (`Part_Of` to `part_of`; `part of` and `member of` match nothing);
a field of several keywords keeps the declared types among them and loses the rest. With
`relationship_inverses` (below), a field whose only names on the list are inverse names
(`has_part`: the model wrote "B has_part A") is that relation read the other way round:
the pair is swapped and the forward type stored (`A part_of B`); in a field that also
names a declared type, an inverse name is an off-list keyword. A field with no declared
type and no inverse name becomes `related_to` when the list declares it, and is
otherwise dropped with its relation. A `keywords` field the model returned as a JSON
list is read as its elements (without the contract it is stored as the string of the
list, as before). Each change is counted: a document whose relations needed any logs one
warning (also in the pipeline history) with the counts (`normalized`, `multi_type`,
`inverse`, `related_to`, `dropped`, of `checked`), and `/health` shows the running totals
as `kg_contract_relation_keywords`, under `extraction`. The totals are those of the
worker process that answers (each gunicorn worker keeps its own), and they count every
attempt: a document that fails after its extraction and is processed again is counted
again. Any change counted means the schema is not being enforced upstream. Relations the code makes without the
LLM (the sidecar fan-out) and relations a caller gives (`create_relation`,
`edit_relation`, `insert_custom_kg`) are not checked. Without `relationship_types`
nothing is checked or counted.

A sample profile for a geoscience paper corpus, with relation list v3.1 (17 types, each
defined by what its source and target must be, and their inverse names), is
`prompts/samples/entity_type_prompt.kg_contract.sample.yml`. `/health` reports the
active profile's counts as `kg_contract_profile` (`relationship_types`,
`relationship_inverses`; 0 and 0 when the contract is off).

#### Direction in the graph: inverse names

The graph stores each pair of entities once, in sorted order (Python string order, by
code point): the extraction merge sorts every extracted pair before merging,
`PGTableGraphStorage` keeps `src_id < tgt_id` by the same order, and the Apache AGE
storage keeps the direction the merge writes, which is that order. So "GRACE part_of
Constellation" is stored as (Constellation, GRACE) with keywords `part_of`, and two chunks
that give A `part_of` B and B `has_property` A merge into one edge with keywords
`has_property,part_of`. Without more, the stored keywords do not say which node is the
part: the direction survives only in the relation's description text.

A profile may also declare `relationship_inverses`, a mapping of a directed type name to
its inverse name; a type the mapping does not name is symmetric. It needs
`relationship_types` in the same profile; each key must be a declared type, each inverse
lower-case snake case, not a declared type and the inverse of one type only, or the
profile is refused at load. With it, every write path below puts a relation into sorted
order and, when that flips an extracted or given relation "A t B", stores the inverse name
of `t`, so a stored edge (src, tgt) with keyword k reads "src k tgt" whenever k is a
declared type or inverse name: Constellation `has_part` GRACE, and the two chunks above
give `has_part,has_property`. Every keyword of an LLM-extracted relation is one, because
the parser checks it against the list (above). A keyword that is neither, given by a
caller (`create_relation`, `edit_relation`, `insert_custom_kg`), made by the code (the
sidecar links), or stored before the profile declared the list, is kept as written and
not inverted when its pair is flipped, so it can read backwards.
In a keyword field that holds several comma-separated types after merges, each declared
type or inverse name is rewritten on its own (exact match); symmetric types and any other
keyword stay as they are. The LLM never sees an inverse name: the prompt, the guidance and
the schema keep the forward list, and the rewrite is a deterministic storage-side step
(`orient_relation` in `lightrag/kg_contract.py`). Without the key every write is what it
was (`tests/extraction/test_kg_contract_direction.py`). The sample profile's inverses:

| type | inverse | | type | inverse |
|---|---|---|---|---|
| `is_a` | `has_kind` | | `derived_from` | `source_of` |
| `part_of` | `has_part` | | `operates` | `operated_by` |
| `measures` | `measured_by` | | `located_in` | `location_of` |
| `estimates` | `estimated_by` | | `affects` | `affected_by` |
| `uses` | `used_by` | | `corrects` | `corrected_by` |
| `applied_to` | `target_of` | | `depends_on` | `determines` |
| `produces` | `produced_by` | | `has_property` | `property_of` |

`same_as`, `compared_with` and `related_to` are symmetric.

The write paths and what each does with the key:

| path | with `relationship_inverses` |
|---|---|
| gleaning merge (`_merge_extraction_pass`) | within one chunk, a gleaning relation that states a first-pass relation's pair the other way round with a directed type (one the mapping names) removes each first-pass record of the reversed pair whose types are all directed and all restated (the gleaning turn is the correction). A symmetric type has no direction: when both sides are of symmetric types only, the reversed restatement is the same relation and the version with the longer description is kept, as for the same pair (a tie keeps the first pass). Any other reversed relation (another directed type, or a mix of directed and symmetric types) is a second relation and is kept beside the first |
| extraction merge (`merge_nodes_and_edges`) | at the end of each chunk's extraction every relation record gets `reverse_keywords`, its keywords read the other way round (the inverse names, resolved with the table then; `add_reverse_keywords`); the merge puts each record into sorted order with `oriented_record`, which swaps the pair and takes `reverse_keywords` when the sort flips it, so the stored edge's keywords read in stored order |
| rebuild after a deletion, the kept-summary rebuild and the rollback rebuild (`rebuild_knowledge_from_chunks`, `_rebuild_single_relationship`) | a chunk extracted under the contract has its stored records read (*Rebuilding from stored records*), each oriented by the same `oriented_record` from what the record carries, so the direction, a gleaning correction included, is what extraction stored whatever the profile says now; a chunk extracted without the contract is re-parsed from its cache as it was, without the rule and without inversion; the edge is written back in sorted order |
| `ENTITY_NAME_FOLD` and `KG_CONTRACT_VALIDATOR` renames | no change needed: both rename endpoints before the merge and keep the extracted direction, so the merge orients the renamed pair; the stored records carry the renamed pair, so a rebuild orients the same pair |
| entity rename (`edit_entity`), entity merge (`merge_entities`, and `edit_entity` with `allow_merge`) | the stored keywords read in the old sorted pair; the renamed or redirected pair is re-sorted, inverted when it flips, and written in sorted order before duplicates are joined |
| `create_relation`, `edit_relation`, `insert_custom_kg` | keywords given with (source, target) read "source keywords target"; they are inverted when the stored order is the other way, and the edge is written in sorted order |

Limits:

- Turning the key on does not rewrite edges already in the graph: they keep their forward
  names in whichever order the merge sorted them, and later merges add inverse names next
  to them; a rebuild after a deletion re-reads the cached extractions of documents
  extracted before as they were stored, forward names unchanged. Re-extract the
  documents (delete and insert) for a graph that holds the invariant throughout.
  Turning it off later leaves the inverse names in place.
- The gleaning merge keeps, for one pair in one order, the record list with the longer
  first description whatever its types (the rule extraction always had): a vaguer type
  (`related_to`) with a longer description replaces a specific one stated in the same
  order. A reversed relation of another directed type is kept beside the first, so two
  contradictory directions can both be stored, and a first-pass record of several types
  is never corrected. These are left as they are.
- `edit_relation` reads the keywords in its request in the caller's order, while
  `get_relation_info` and the graph view return the stored keywords with the pair as
  asked or as stored; a caller that reads in the reversed order and writes the same
  keywords back inverts them. Edit in the stored order (smaller name first), as the
  graph view does.
- Tools that copy or re-key edges outside these paths do not know the profile:
  the graph migration tool (`lightrag/tools/migrate_graph_storage.py`) copies each
  pair in the order the source lists it, so an AGE target can draw the arrow of a pair a
  NetworkX source lists unsorted the other way round (the keywords still read in sorted
  order); `PGTableGraphStorage`'s start-up normalization of reversed legacy rows swaps the
  endpoints without touching keywords (only rows written before the storage kept sorted
  order); and a bulk tool that renames entities directly in the database must apply
  `kg_contract.orient_relation` to each edge it re-keys.
- Sidecar links (`SIDECAR_RELATIONS=all`) are undirected containment links: their
  keywords, `associated with, contained in`, say that the entity and the table, drawing
  or equation occur together, whichever order the pair is stored in. They are not
  declared types, so they are neither checked nor inverted; read them without a
  direction, or filter them out by keyword or by the sidecar entity's type.
- Readers take the keywords as text: the relation vector content (the keywords, the
  sorted pair and the description), the web UI edge label and `get_relation_info` show
  the stored keywords; the query context does not include keywords. None of them assumes
  the forward list.

### Extraction contract validator

With `KG_CONTRACT_VALIDATOR=true` the API server passes
`lightrag.kg_contract.kg_extraction_validator` as `kg_extraction_validator`, the hook that
sees each chunk's parsed records before the merge (see *Extraction Quality Hook* in
`docs/ProgramingWithCore.md`). It runs after `DROP_SYMBOL_ENTITIES` and
`DROP_JUNK_ENTITIES` and does not need the relation list. Per chunk it

- drops a name that is mathematical notation of three or more characters (`x_k`, `q_n(x)`,
  `f(x)`, `x1*`, TeX characters) and every relation that names it; `L1`, `J2`, `C20`,
  `F10.7` and `P(Y)` stay;
- keeps a name glued from several words (`MetOp`) when the chunk text contains it, renames
  it to the text's spelling of its words when the text has them (`ScaleHeight` to
  `scale height`), drops it when a part is one or two characters, a Greek-letter or
  accent word, carries a digit or is an `et al.` (`PhiTau`, `p_y_2`), and otherwise splits
  it into lower-case words; relations follow the rename;
- drops a relation whose endpoint is not an extracted entity of the chunk and is not in
  the chunk text either, or is notation; a grounded undeclared endpoint is kept, and the
  merge creates its node as before;
- drops a relation whose two endpoints a rename joined.

Names are looked up in the chunk text after the same normalization extraction applies to
names, case-folded. The rename and the grounding depend on the chunk's text, so one name
can be kept in one chunk and renamed in another.

The hook runs once per chunk, at extraction. Its result (with the multimodal links added
after it, under the names `ENTITY_NAME_FOLD` gave them when it is on) is stored with the
chunk, and the knowledge rebuild after a document deletion,
the kept-summary rebuild of a re-ingest and the rollback purge read that and never call
the hook (*Rebuilding from stored records*): a renamed or dropped name's records are
found under the name the graph holds, so a deletion leaves the edge and its nodes with
the surviving documents' chunk ids and text only, as without a hook, except a node that
the surviving chunks name only as the end of a relation and hold no entity record for: it
is rebuilt from its edges' current descriptions and keeps the deleted document's text
(as on the base branch, with or without a hook). A hook that is
changed, removed or raising later changes nothing a deletion rebuilds. The
`lightrag-repair-chunk-tracking` tool attributes such a chunk from its stored records.
Reported in the `/health` configuration.
Off by default.

### Rebuilding from stored records

A knowledge rebuild (after a document deletion, in the kept-summary rebuild of a
re-ingest, in the rollback purge of a custom-chunk patch) and the
`lightrag-repair-chunk-tracking` tool need what each surviving chunk contributed to the
graph. On the base branch they re-parse the chunk's cached LLM output, which is all its
extraction did to it. Under the extraction contract or an extraction hook, extraction does
more (the keyword check, the gleaning direction rule, the inverse names, the hook), and
re-running that later gives a different answer as soon as anything changed: the relation
list, the inverse names, the hook, the code's rules, or the contract switched off. So
extraction stores, with each chunk, the records it accepted, and the rebuild reads them.

When. Only while the profile declares `relationship_types` or a `kg_extraction_validator`
is set (`KG_CONTRACT_VALIDATOR`). Without either nothing is written, and the chunk rows
and the rebuild are those of the base branch (for PostgreSQL see *Storages*).

What. One chunk's records as the merge consumes them: after the first-pass parse and
its keyword check, the gleaning parse, its check and the gleaning merge with its
direction rule, the hook, the multimodal links (the base rebuild does not make those
again, so they are stored too) and `ENTITY_NAME_FOLD`. What was accepted is what is
stored: a truncated gleaning reply that was merged but not cached is in the records; a
relation whose two ends the fold joined into one name, which the merge never sees, is
not. The records are written for all of a document's chunks after the fold and before
its extraction returns, so no chunk reaches the merge without them.

Where. In the text-chunk row, field `kg_records` (`utils.CHUNK_KG_RECORDS_FIELD`), so it
is deleted with the chunk. Its value has a fixed shape (no key varies per chunk):

```
{"format": 1,
 "entities":  [{"entity_name", "entity_type", "description", "source_id",
                "file_path", "timestamp"}, ...],
 "relations": [{"src_id", "tgt_id", "keywords", "reverse_keywords",
                "description", "weight", "source_id", "file_path",
                "timestamp"}, ...]}
```

A record holds those keys only (the ones it has); `entity_name` and `src_id`/`tgt_id` are
the name and the pair it is filed under: the ones the merge wrote to, the model's
spelling as the hook and the fold left it. The model's own spelling is not kept (the
extraction cache has it). `keywords` read from `src_id` to `tgt_id`; `reverse_keywords`
read from `tgt_id` to `src_id`, resolved with the inverse names at extraction (the same
keywords without `relationship_inverses`). The graph stores each pair sorted: an edge
(a, b) gets a relation's `keywords` when `src_id <= tgt_id` and its `reverse_keywords`
otherwise (`kg_contract.oriented_record`; the extraction merge and the rebuild both use
it, on the same stored pair), so the stored graph can be checked against the stored
records without the profile. Measured on 5,988 chunks of the production cache (7,918
first-pass and gleaning replies): 3.8 kB of JSON per chunk on average (median 3.5 kB,
95th percentile 10.9 kB, largest 49 kB), 1.2 kB as PostgreSQL stores it compressed.

Storages. JSON, Redis and OpenSearch keep the field in the chunk document: a row read and
written back keeps it, and the pipeline's upsert of a re-created chunk (a document
processed again) replaces the document without it. MongoDB's chunk upsert is a `$set`, so
it unsets the field when the row written does not carry it. `PGKVStorage` keeps it in a
nullable JSONB column `kg_records` of `LIGHTRAG_DOC_CHUNKS` that only three statements
name (write, read, clear); the general chunk reads and the upsert leave it alone, so the
base branch keeps working on a table that has it. New tables get it in `CREATE TABLE`. On
an existing table the start-up column lookup only notes whether it is there; it is added
at start-up when the contract or a hook is on, with `ADD COLUMN IF NOT EXISTS` under a
5 s lock timeout, three attempts; if it still cannot be added, start-up is refused with a
message that says how to add it by hand. While the column is absent no statement names
it. Because the PostgreSQL upsert leaves the column alone, extraction clears the column
for a document's chunks before it extracts them, in every configuration, whenever the
column exists (one statement per document), and a rebuild reads it with one statement.

Lifecycle. A records write that fails fails the document's extraction, which is retried
as any failed document. A process that stops between the write and the merge leaves
the document unfinished: it is processed again (its records cleared or replaced first)
or deleted with its chunks. A chunk extracted again with neither contract nor hook
keeps no records (the document store replaced it; PostgreSQL cleared it) and is rebuilt
from its cache.

The rebuild. Each surviving chunk with stored records contributes them, whatever the
configuration is now: nothing reads the current relation list, inverse table, hook or
name fold for it, and it needs no extraction cache. Its records are filed under their
stored names, which are the graph's, so with `ENTITY_NAME_FOLD` off, its map not armed
within the wait, a key whose canonical spelling moved or another alias file, the rebuild
finds them as it does with the map extraction used. Stored records of an unknown shape
make the chunk unusable, as an unparseable cache does. A chunk without them is re-parsed
from its cache as on the base branch and filed under the model's spelling and the
spellings the fold map in force gives it (`alias_chunk_result`). An entity or relation
fed by both kinds gets each chunk's contribution its own way: with the fold off or
changed, a re-parsed chunk's contribution to a name the fold changed is not found (its
chunk id stays), as on the base branch, while a chunk with records still contributes.

The `lightrag-repair-chunk-tracking` tool attributes a chunk with stored records from
them, names the fold changed included; a chunk without them from its cache, where a
name the fold changed gets no attribution, as on the base branch.

Chunks extracted before. Every chunk extracted before this rule, with or without the
contract (production has none with it), carries no records and is rebuilt from its
cache, as the base branch rebuilds it: switching the contract or the hook on does not
change what a deletion rebuilds for them. Re-ingest those documents to give them
records. Switching the contract or the hook on does not re-check,
re-key or rename anything already in the graph; switching it off, or changing the list,
the inverse names, the hook or the name fold, changes nothing for chunks that have
records. Tests:
`tests/extraction/test_kg_contract_reproduce.py`,
`tests/kg/test_chunk_kg_records_storage.py`, and
`tests/kg/postgres_impl/test_pg_kg_records_smoke.py` for the PostgreSQL column
(`pg_smoke`, below).

### Entity name fold

With `ENTITY_NAME_FOLD=true`, every entity name an extraction produces is mapped to the
spelling the graph already uses for the same fold key, or becomes that spelling when
the key is new (`lightrag/entity_name_fold.py`). The extraction LLM spells one entity
several ways across papers, and the merge keys on the exact string, so without it a
concept's relations scatter over several nodes. The map is read from the PostgreSQL
graph (`PGGraphStorage`, Apache AGE, or `PGTableGraphStorage`, plain tables, for its own
workspace and namespace) in the background after the storages initialise (computed in a worker thread, so
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

Applied where results leave `extract_entities`, so the write-ahead anchors, the merge
and the records stored with each chunk (*Rebuilding from stored records*) name the same
nodes, and as an alias in the knowledge rebuild, which files each cached record of a
chunk without stored records under its original and its canonical name (and its legacy
key's canonical when that is a third name). Changing the key rules or the lists
changes keys: existing variant nodes are not merged by the switch itself.

### Sidecar relations

`SIDECAR_RELATIONS=none` keeps a table/equation/drawing sidecar entity but no longer
links it to every entity extracted from its chunk with an "associated with, contained
in" edge (upstream behaviour, `all`). On a paper corpus those edges were about a tenth
of the graph, every one generic. The links carry no direction: with the extraction
contract on they are stored in either order and their keywords are left as they are
(see *Direction in the graph*).

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

### A document delete keeps cache rows another document lists

An extraction or multimodal-analysis cache row is keyed on its prompt, so a chunk of
another document with the same text shares it: its cache hit lists the key in its own
`llm_cache_list`. Upstream's `adelete_by_doc_id(delete_llm_cache=True)` deleted every
key the document's chunks listed, so the surviving copy's references dangled and its
next rebuild or re-ingest found nothing cached. On this branch the delete keeps any
row a chunk outside the document still lists, and reports how many it kept ("kept N
that chunks of other documents still list"); the last document to list a row deletes
it. The lookup runs after the purge, so the rebuild still reads every row, and it
covers the retry of a delete whose chunks were already removed and the custom-chunk
rollback. It is `cache_ids_listed_by_other_chunks` on the text-chunk storage:
`JsonKVStorage` scans its rows in memory, and `PGKVStorage` runs one
`llm_cache_list ?| keys` statement per delete, a scan of the workspace's chunks unless
a GIN index exists on `llm_cache_list` (none is created; a comparable read-only lookup
took 1.1 s over about 296,000 chunks, and this statement has not been run against a
live database). Redis, MongoDB and OpenSearch lack the method and delete
every key, as upstream did, with a warning. A reference an earlier delete left
dangling is not repaired; its chunk re-extracts on its next reprocess. Because the kept rows still answer the
shared prompts, re-ingesting a deleted copy no longer re-extracts its shared chunks.
Smart-heading rows ride document metadata, not a chunk list, and are still deleted
whatever another document shares. See
[docs/design/PurgeRecoveryContract.md](./docs/design/PurgeRecoveryContract.md). Tests:
`tests/pipeline/test_delete_keeps_shared_cache_rows.py`.

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

### A rebuild reads a shared extraction through every chunk that lists it

An extraction cache row is keyed on its prompt, so chunks with identical text (two
copies of a paper, say) share one row, and the row's `chunk_id` names only the chunk
that wrote it first; a later chunk's cache hit adds the key to its own
`llm_cache_list` and leaves the row alone. Upstream's `_get_cached_extraction_results`
kept a row only when that `chunk_id` was among the surviving chunks, so deleting the
copy that wrote the rows dropped the survivor's whole cached extraction: the rebuild
found no cache and, under the default policy, left the shared entities and relations
untouched, still naming the deleted chunks. On this branch the rows are grouped by the
surviving chunk whose `llm_cache_list` names them, and a row two surviving chunks share
counts for each, as at insert time. The same function feeds the document-delete
rebuild (with or without `keep_summaries`), the re-ingest rebuild of kept summaries,
the rollback purge and the `lightrag-repair-chunk-tracking` tool, which now also
attributes a chunk from rows another chunk wrote. Existing data is read correctly at
once; nothing is rewritten. A delete with `delete_llm_cache=True` keeps the rows a
surviving copy lists (see "A document delete keeps cache rows another document
lists"), so its next rebuild still finds them. Tests:
`tests/pipeline/test_rebuild_shared_cache_rows.py`.

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

### Upload to Zotero

With `ZOTERO_UPLOAD=true`, the WebUI's Upload button (`POST /documents/upload`) adds
the file to a Zotero library instead of indexing it, and the document reaches
LightRAG through the existing Zotero corpus build under its Zotero key
(`KEY__slug.md`). Nothing is written to the input directory, so the upload cannot
create a second copy of what the corpus build later produces. Only PDFs are
accepted, up to `MAX_UPLOAD_SIZE`; `GET /documents/supported_file_types` then lists
`.pdf` alone and `/health` reports `zotero_upload: true`, which is what makes the
WebUI say "Add to Zotero". Unset, the route is upstream's, unchanged. With the
switch on but a setting missing, the upload answers 503 naming it; it never falls
back to indexing.

Order of operations, in `lightrag/zotero_add.py`:

1. The file's md5 is looked up in the upload ledger (`ZOTERO_UPLOAD_LEDGER`) and,
   when `ZOTERO_SQLITE` names a `zotero.sqlite`, in its live attachments: a known
   file is refused with 409.
2. The first pages and the PDF metadata yield DOI candidates and a title; CrossRef
   (`api.crossref.org`, polite pool through `ZOTERO_CROSSREF_MAILTO`) is asked for
   the DOI, then for the title. A record counts only when its title is found in
   the PDF's own text (a cited paper's DOI does not), and for a title search its
   first author's surname too. No match, or CrossRef unreachable, gives a stub
   item `Uploaded: <file name>` tagged `unmatched-upload`, carrying an unverified
   DOI candidate in `extra`; the response says which happened.
3. A DOI already in the library (the local metadata file, the ledger, then a title
   search of the Zotero API filtered on the DOI, since the API cannot search that
   field) is refused with 409 when the item holds a PDF. When the item has none,
   as with the library's reference stubs, the PDF is attached to it, filling an
   empty attachment row if there is one; no second item is created. If the API
   check itself fails the upload goes on and the response carries a warning.
4. The parent item (item template from the API, filled from the CrossRef record;
   `DOI` goes into `extra` for types without that field), the attachment row, the
   WebDAV files `<KEY>.zip` and `<KEY>.prop` (the layout the Zotero client reads,
   `.prop` removed first and written last), then `md5` and `mtime` onto the
   attachment row. A failure from the attachment row on deletes what the call
   created in Zotero and on WebDAV and answers 502 saying so, or naming what could
   not be removed. An upload holds a file lock for these steps, so two uploads of
   one file cannot both pass the checks.
5. With `ZOTERO_SYNC_COMMAND` set, the command is launched detached (no shell,
   output to `ZOTERO_SYNC_LOG` or discarded). Nothing starts while a command this
   process launched is alive or `ZOTERO_SYNC_LOCK_FILE` is held; the command itself
   should take that lock without waiting (`flock -n -o`), which settles two
   workers racing. The response says `started`, `already_running`,
   `not_configured` or `failed` (`sync_triggered` is true only for the first). An
   upload that arrives while a build is past its Zotero sync waits for the next run.

Settings are documented in `env.example`; the Zotero library id, a write-enabled
API key and a WebDAV store (`WEBDAV_ENABLED` must be 1 in a settings file) are
required. Files are read on every upload, so a rotated key needs no restart.
Tests: `tests/test_zotero_add.py`, `tests/api/routes/test_zotero_upload_route.py`.

### Tests

The unit tests for these changes, including the upstream tests the branch modifies,
are marked `offline`, so upstream's
[.github/workflows/tests.yml](./.github/workflows/tests.yml) runs them; on this branch it
also triggers on pushes and pull requests to `zlightrag`, as does upstream's
[.github/workflows/linting.yaml](./.github/workflows/linting.yaml). The exceptions are the Apache
AGE integration test for the edge removal,
`tests/kg/postgres_impl/test_postgres_remove_edges_age.py`, which needs
`--run-integration` and a live AGE server, and the fold map build and the extraction
contract's inverse names on the table graph storage,
`tests/kg/pgtable_impl/test_pgtable_fold_map_smoke.py` and
`tests/kg/pgtable_impl/test_pgtable_relation_direction_smoke.py`, and the extraction
contract's text-chunk column, `tests/kg/postgres_impl/test_pg_kg_records_smoke.py`,
marked `pg_smoke` like the storage's other live-PostgreSQL tests, which need
`--run-integration` and `POSTGRES_PASSWORD`. No workflow on this branch runs them.

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
branch has since taken upstream `main` by merge, most recently `31bff1f60` (merged
2026-11-17). Its history is no longer rewritten: a rebuild onto a new base is published
as a merge commit whose tree is the rebuild's, so any commit you pin stays on the
branch.

Nothing here is proposed to or maintained by the LightRAG project.
