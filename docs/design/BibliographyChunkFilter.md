# Bibliography Chunk Filter

This note covers the query-time option `DROP_BIBLIOGRAPHY_CHUNKS`
(`LightRAG.drop_bibliography_chunks`) and its detector,
`lightrag.utils.is_bibliography_chunk`: where the filter runs, how it relates to the
chunk-time `drop_references` option, the detection rule, the evidence behind its
thresholds, the alternatives that were rejected, and what it misses.

## Why

A reference list is dense in exactly the words a question uses: paper titles. Its
chunks therefore score well in vector search and with rerankers while carrying no
content the answer can use. On a knowledge base of academic papers they hold
`chunk_top_k` and token-budget slots that prose could fill. In a probe query
against such a corpus, most of the top-ranked chunks were bibliography text: the
same citation, lifted from several papers' reference lists. With the filter on,
none were. Whether prose then takes the freed slots depends on the retrieval
mode; see *Where it runs*.

## Where it runs

At the head of `process_chunks_unified`, before rerank, the `min_rerank_score`
filter, `chunk_top_k` and token truncation. Every retrieval mode goes through that
function (`naive` through `naive_query`, `local` / `global` / `hybrid` / `mix`
through `_build_context_str`), so a bibliography chunk is never sent to the
reranker and never takes a context slot. If every candidate is a bibliography the
function returns no chunks, as it does when `min_rerank_score` removes them all.

What takes the freed slots depends on the candidate pool:

| Mode | Candidates | Freed slots and budget |
|---|---|---|
| `local`, `global`, `hybrid`, `mix` | chunks of the retrieved entities and relations, plus (`mix`) the vector leg's `chunk_top_k` chunks, merged into one pool | taken by the next candidates whenever the pool is larger than `chunk_top_k` |
| `naive` | exactly `chunk_top_k` chunks from the vector store (`_get_vector_context`), no other source | stay empty: the context shrinks, as it does when `min_rerank_score` drops a chunk |

In `naive` mode the filter therefore removes reference lists from the context
without promoting prose ranked below `chunk_top_k`; that is an accepted residue,
listed under *Known misses*.

Only the candidate list changes. Stored chunks, vectors, entities and relations
are untouched, and turning the option off restores the previous behaviour
exactly.

KG modes read the flag from `text_chunks_db.global_config`, the snapshot the
storages take at construction; `naive` reads the per-call `global_config`. The
query-answer cache key reads the same dict as the code that applies the filter,
and carries a component only when the filter is on: enabling it cannot serve an
answer cached from unfiltered context, and entries written with it off keep
their key.

## Relation to `drop_references`

The paragraph-semantic chunker's `drop_references` option (`CHUNK_P_DROP_REFERENCES`,
hint `drop_rf`) removes reference sections at chunk time. When it applies it is
the better tool, because it also saves the entity extraction the reference list
would cost. It does not apply in these cases, each read from the code:

| Case | `drop_references` (chunk time) | Query-time filter |
|---|---|---|
| Chunking strategy | `P` only (`ParamSpec` targets in `lightrag/parser/param_schema.py`) | any chunker, including custom ones |
| No usable `.blocks.jsonl` sidecar | `P` falls back to recursive-character chunking, which returns before the drop runs | unaffected |
| Detection | block heading matches a prefix in `CHUNK_P_REFERENCES_HEADINGS` (default `References`, `Bibliography`, `参考文献`) | chunk text; no heading needed |
| List under another heading ("Literature Cited", "Works Cited"), or heading lost by the parser | kept | dropped |
| Documents chunked before the option was enabled | kept until re-processed (the switch is frozen into `chunk_options` at enqueue) | dropped from the next query on |

Position in the document is not a difference under the defaults:
`CHUNK_P_REFERENCES_TAIL_N=0` scans every block, so a reference section in the
middle of a document is found by heading just as a trailing one is. Only a
positive `CHUNK_P_REFERENCES_TAIL_N` restricts the chunk-time scan to the end.

Running both is the intended setup: `drop_references` at ingest where it applies,
the query-time filter for everything else already in the store.

## Detection rule

The detector works line by line. A **reference-entry line**:

- starts like a reference: an optional list marker (`-`, `*`, `•`, `[n]`, `n.`,
  `n)`), optional lower-case name particles (`van`, `de`, `von`, ...), then
  `Surname,` / `Surname Surname,` / `Surname I. I.` / `Surname AB,` / `I. Surname`;
- contains a year between 1600 and 2039, optionally with a letter suffix (`2020a`);
- and contains either a citation cue (`doi`, `doi.org`, `http(s)://`, `arXiv`,
  `p.`/`pp.` followed by a number, `vol.` followed by a number, `48(12), 4384`,
  `62, 4384–4399`, `In: X`, `(ed.)` / `(eds.)`, `edited by`) or an author-initials
  pattern (`Surname, I.`, or `SURNAME W. A.` followed by `&`, `,`, `(`, a digit or
  `and`).

A chunk is a bibliography when it contains **five or more** entry lines,
anywhere in the chunk and not necessarily consecutive, or a single entry line
that is a **packed list** — a whole reference list extracted as one paragraph:
longer than 600 characters, with at least 3 DOI/URL markers (`doi`, `doi.org`
and `http(s)://` each count, so one `https://doi.org/...` link counts twice; a
bare `10.xxxx/...` DOI does not) and at least 5 years. A list extracted as one
paragraph that misses any of these counts is only one entry line toward the
five.

## Evidence

Measured over a corpus of academic papers with about a hundred thousand chunks:

- About a tenth of the chunks match.
- The number of entry lines per chunk is bimodal: most chunks have none, a small
  share have ten or more, and the 3 and 4 buckets are nearly empty. Five sits in
  the valley between the modes.
- In the probe query described under *Why*, no bibliography chunk was left among
  the top-ranked chunks with the filter on.

The valley does not show that chunks straddling prose and a reference list
survive. A chunk whose reference tail has four entry lines or fewer is kept, but
one whose tail reaches five, or holds a packed line, is dropped whole, prose
included; *Known misses* gives how often that happened on a proxy corpus.

These figures were measured before the two pattern changes described under
*Cost*. The current patterns made the same decision as the measured ones on every
chunk of a second sample of academic full text.

## Rejected alternatives

- **Counting DOIs.** Prose that cites two DOIs, followed by a three-entry
  reference tail in the same chunk, already reaches five; and a data-availability
  paragraph that lists dataset DOIs is not a bibliography.
- **Heading-based detection at query time.** Text chunks from the built-in
  chunkers other than `P` carry no heading, and matching headings is what
  `drop_references` already does at chunk time.
- **Relying on the reranker or `min_rerank_score`.** Rerankers score reference
  lists highly for title-like queries; that is the failure being fixed.

## Known misses

These are accepted residues of a heuristic filter:

- **Kept although reference text:** a chunk that straddles prose and four or
  fewer entries (deliberate — it still carries the prose); styles that do not
  open with an author name (title-first entries, some web citations); entries
  whose first author's name does not begin with an ASCII capital (`É`, `Ø`, CJK
  scripts); entries wrapped over several lines whose first line carries no year;
  a list extracted as one paragraph with fewer than three DOI/URL markers, such
  as an older list printed without DOIs or with bare ones, which counts as a
  single entry line; entries with no citation cue whose only author-initials
  form has a surname the initials pattern cannot match from its first letter.
  That pattern takes ASCII letters only, starts at an ASCII capital and, since
  the lookbehind described under *Cost*, cannot start right after a hyphen or
  an apostrophe (`'` or `’`). So it misses a surname with a non-ASCII letter
  (`García, J.`, `García-Lopez, J.`) and a joined one whose first letter is
  lowercase or non-ASCII (`al-Farabi, A.`, `d'Alembert, J.`,
  `Álvarez-Lopez, K.`). Before the lookbehind it restarted at the capital after
  the join and found `Lopez, J.`, `Farabi, A.`, `Alembert, J.` and `Lopez, K.`
  in the joined examples; `García, J.` and `Lopez-García, J.`, whose last part
  holds the non-ASCII letter, were missed then too. A first author whose name
  starts lowercase or non-ASCII already fails the entry-start pattern, so those
  joined forms matter only for a later author
  (`Smith, John, and al-Farabi, A. 1990. …`); `García-Lopez, J.` is missed in
  either position. `O'Neil, K.` and `Smith-Jones, A.` match whole and are
  unaffected.
- **Dropped although prose:** a single paragraph longer than 600 characters that
  opens like `However,` (a capitalised word and a comma), carries at least three
  DOI/URL markers and five years, and contains a citation cue or an initials
  pattern is read as a packed list. Five lines anywhere in one chunk,
  consecutive or not, that each open with a capitalised word and a comma and
  carry a year and a citation cue are read as entries, and the chunk as a
  bibliography.
- **Dropped together with the prose before the list:** the verdict covers the
  whole chunk. A chunk that ends in a reference list is dropped as soon as the
  list reaches five entry lines or one packed line, and the prose before the
  list goes with it. Size-based chunkers such as `F` and `R` cut without regard
  to headings, so that prose is typically the end of a paper's discussion or
  conclusions plus its acknowledgments. Nothing is deleted from the store, but
  the prose no longer reaches the query context, except for the part the chunk
  overlap repeats at the end of the previous chunk. How often this happens was
  not measured on the corpus of *Evidence*. On a proxy corpus it was: text
  extracted from a few thousand academic PDFs, cut into windows of 4,800
  characters (about one default 1,200-token chunk) overlapping by 400
  characters. The detector flags a little under a tenth of the windows. About
  one in seven of the flagged windows (about 1% of all windows) holds at least
  1,000 characters of prose-like lines before its first entry line, through
  either rule. A prose-like line here is one of 20 or more characters with no
  year, citation cue or initials pattern. The median is about a third of the
  window, and the largest most of it. These windows come from about three in
  ten of the documents with any flagged window. In a small random sample of
  them, every one held the document's body text.

  Cutting a flagged chunk at its reference heading instead of dropping it would
  keep most of that prose: in about five of six such windows, a heading line
  such as `References` or `7. REFERENCES` is the last non-empty line before the
  first entry. It is not done here. The cut would change a chunk's text in the
  query context, where the filter now only removes chunks, and its rule would
  need its own measurement on chunks cut by the real chunkers. A cut at the
  first entry line alone is not enough: when a window starts inside a reference
  list, the text before its first recognised entry line is often reference text
  too. About half of the flagged windows hold at least 1,000 characters before
  that line, and under a third of those hold that much prose-like text.
  `tests/utils/test_bibliography_chunk_filter.py` pins the current behaviour,
  for five entries and for a packed line.
- **Not refilled in `naive` mode:** the slots and token budget a dropped chunk
  held stay empty, because `naive` fetches exactly `chunk_top_k` candidates (see
  *Where it runs*). Nothing is lost that the unfiltered query would have used:
  the prose among the fetched candidates is all still there. To get more prose
  into a `naive` context, raise `chunk_top_k`, or use `mix`, whose graph legs
  refill the slots. Over-fetching from the vector store while the filter is on
  would refill them too, but needs an over-fetch factor with no measurement
  behind it and changes what the vector store and the reranker receive, so it is
  not done; `tests/test_operate_naive_bibliography_slots.py` pins the current
  behaviour.
- **Not covered at all:** entities and relations extracted from reference-list
  chunks during ingestion. The filter removes chunk candidates only; use
  `drop_references` at ingest to avoid that extraction.

## Cost

The detector runs a few regular expressions per line, rejects a line that does
not open like a reference with the first (anchored) one, and stops at the fifth
entry line. It runs inline on the event loop.

Timed on the proxy windows of *Known misses* (up to 4,800 characters of text
extracted from academic PDFs), taking the fastest of five calls per window with
`time.perf_counter_ns`, single-threaded in CPython 3.13: a window it keeps took
about 10 µs at the median (a few tens of µs at the 90th percentile, under half a
millisecond at worst), a window it flags about 80 µs (about 140 µs at the 90th
percentile). Three hundred windows drawn at random took about 10 ms at most, so
a few hundred candidates cost milliseconds, well below a rerank call. A repeat
run agreed closely.

Chunk text is uploaded content, so no pattern may backtrack super-linearly on
a crafted line. Two patterns did, each
taking several seconds on a line of 32k–64k characters:

- **Author initials.** `\b` also matches after a hyphen or apostrophe inside a
  name, so on a run of hyphen-joined capitals every capital opened a new attempt
  that rescanned the rest of the run. A lookbehind now lets a match start only
  where a run of name characters starts.
- **The `48(12), 4384` cue.** The whitespace before and after its optional `,` /
  `:` separator could each take part of a whitespace run, so after a `1(1)` token
  every split of a long run was tried before the match failed. The separator and
  the whitespace after it are now optional as one unit, which accepts exactly the
  same strings.

`tests/utils/test_bibliography_chunk_filter.py` pins both shapes, and only
those. A scaling sweep of every pattern and of the whole detector over about
2,000 crafted line shapes (an entry-like opening followed by a long run of one
repeated unit) flags both old patterns and nothing in the current ones, but that
sweep is not part of the suite.
