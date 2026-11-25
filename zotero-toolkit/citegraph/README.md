# citegraph

Builds the citation graph that the fork's paper-structure routes read (`/graph/papers/related`,
`/connect`, `/citation`, `/similar`, `/outside`, `/graphs/paper/map`; see the "Paper structure"
section of [ZLIGHTRAG.md](../../ZLIGHTRAG.md)). It reads the `zotero_metadata.json` that
[`corpus/build_metadata.py`](../corpus/README.md) writes, asks [OpenAlex](https://openalex.org) and
Semantic Scholar's public API ([S2](https://www.semanticscholar.org/product/api)) about every paper
with a DOI, and writes CSV tables plus the SQL that loads them into a PostgreSQL schema of the
LightRAG database. A daily timer can rebuild and reload it without a server restart.

Everything here is standard-library Python and bash; the only external programs are `psql`,
`rsync`, `ssh` and `flock`. Nothing in this directory connects to a database except
`load_zstruct.sh`.

## What it builds

The schema (default name `zstruct`) holds these tables. `lightrag/zotero_struct.py` requires
`papers`, `paper_attachments`, `authors`, `paper_authors`, `venues`, `topics`, `paper_topics`,
`cites` and `coupled`; it reads `cite_contexts`, `cites_external`, `external_works` and
`paper_embeddings` when they exist, and answers as before from a schema without them. The keyword
tables are written for other readers: the server does not use them.

| table | rows |
| --- | --- |
| `papers` | one per library paper: DOI, OpenAlex id, title, year, venue, counts, and (S2) corpus id, citation counts, TLDR |
| `paper_attachments`, `authors`, `paper_authors`, `venues`, `topics`, `paper_topics` | attachment keys, authors (with position), venues, and OpenAlex's four-level topic hierarchy |
| `keywords`, `paper_keywords` | OpenAlex keywords with scores |
| `cites` | citation edges between library papers, from OpenAlex's reference lists and the S2 citation store (`source` is `openalex`, `s2` or `both`; `is_influential`, `intents`, `n_contexts` from S2) |
| `coupled` | bibliographic coupling: pairs of library papers that cite common works, with a score |
| `cite_contexts` | the sentences in which one library paper cites another, in order, with S2's intent labels |
| `external_works`, `cites_external` | works outside the library that at least two library papers cite, and who cites them (`n_library_citers`) |
| `paper_embeddings` | S2's SPECTER2 vector per paper (pgvector), for `/graph/papers/similar` |

## The schema contract

`build_struct_layer.py` and `lightrag/zotero_struct.py` are two halves of one contract: table and
column names, the `zot:<item key>` / `doi:<doi>` paper ids, and the fingerprint query the server
uses to notice a swap. They live in the same repository so one commit changes both, and
`tests/citegraph/test_build_struct_layer.py` checks that the server's fingerprint query matches the
tables the builder writes. Change a table in one place and that test tells you to change the other.

The server reads the schema named by `ZOTERO_STRUCT_SCHEMA`; set it to the `CITEGRAPH_SCHEMA` of the
load (default `zstruct`). It notices a reload within a minute: the load swaps whole schemas, so a
rebuild is never an in-place update.

## Data flow

```text
zotero_metadata.json, zotero_references.json          (corpus/build_metadata.py, build_references.py)
    │  rsync (refresh.sh step 1)
    ▼
build_struct_layer.py --metadata ... --out out.daily.new              (step 2, online)
    │   OpenAlex: works by DOI (batches of DOIs), outside works, lookups of ids the filter misses
    │   S2 Graph API: paper records and TLDRs, SPECTER2 embeddings, title match for papers without a DOI
    │   writes CSVs, apply_/swap_/drop_prev_<schema>.sql, qa.json, QA.md, library_corpus_ids.txt
    ▼
s2_citation_sweep.py --out out.daily.new --store s2store/api_citations.sqlite     (step 3)
    │   S2 /paper/{id}/citations for the papers that are due: the citation contexts
    ▼
build_struct_layer.py ... --s2-store s2store/api_citations.sqlite --offline       (step 4)
    │   no request: joins the store's rows into cites and cite_contexts
    ▼
refresh_gates.py: hard checks + reference check pass; no table below 98% of the live rows    (step 5)
    │   build identical to out.loaded/ → stop, nothing to load
    ▼
rsync → <database host>:<data dir>/out.daily/ ; load_zstruct.sh out.daily                     (step 6)
        builds <schema>_next in one transaction, checks every row count against qa.json,
        runs the server's pgvector `similar` query once, swaps: <schema> → <schema>_prev
```

`refresh.sh` runs all six steps, writes one log line per step to
`$CITEGRAPH_DATA/logs/refresh_YYYYMM.log` and the output of each program to
`logs/refresh_*.log`. After a successful load `out.loaded/` holds the build now live and
`out.loaded.prev/` the one before. A failure leaves the live tables as they were and calls
`citegraph_notify` if the config defines it.

### The programs

```bash
# the builder: online (asks only for what the caches lack), then offline with the citation store
python3 build_struct_layer.py --metadata input/zotero_metadata.json --out out \
    --cache cache/openalex_works.json --reference-check input/zotero_references.json
python3 build_struct_layer.py --metadata input/zotero_metadata.json --out out \
    --cache cache/openalex_works.json --reference-check input/zotero_references.json \
    --s2-store s2store/api_citations.sqlite --offline

# the sweep: citation contexts from the S2 Graph API for the papers that are due
python3 s2_citation_sweep.py --out out --store s2store/api_citations.sqlite --max-hours 3
python3 s2_citation_sweep.py --store s2store/api_citations.sqlite --report     # summary of the store

# the load, on the database host
bash load_zstruct.sh --config citegraph.env out.daily
```

`build_struct_layer.py --help` and `s2_citation_sweep.py --help` list every option. The builder's
exit status is 1 when a request failed after its retries, a hard check failed or the reference
check's Jaccard index is below 0.98; it writes its outputs in every case, and `refresh.sh` decides
from `qa.json` whether the build is good enough to load.

`s2_api.py` is the S2 client the builder and the sweep share. `s2_citations_sync.py` is the
alternative to the sweep, a local copy of the S2 citation dataset; see Known limits.

## Config

`refresh.sh` and `load_zstruct.sh` each take `--config FILE` (or `CITEGRAPH_CONFIG=FILE`) and
refuse to run without one. The file is bash, sourced by both scripts; copy
[`citegraph.env.example`](citegraph.env.example) outside the repository and edit it. It documents
every setting:

| variable | meaning |
| --- | --- |
| `CITEGRAPH_DATA` | required: the data directory (`input/ cache/ s2store/ logs/ out.*`) |
| `CITEGRAPH_CODE` | the directory of the Python modules; default: the scripts' own directory |
| `CITEGRAPH_METADATA_SRC`, `CITEGRAPH_REFERENCES_SRC` | required: rsync sources of the two input files |
| `CITEGRAPH_REMOTE` | ssh host of the database; empty: the database is local |
| `CITEGRAPH_REMOTE_CODE`, `_CONFIG`, `_DATA` | where `load_zstruct.sh`, its config and `out.daily/` are on that host |
| `CITEGRAPH_SCHEMA` | schema name, default `zstruct` |
| `PGHOST PGPORT PGUSER PGDATABASE` | the usual libpq settings |
| `CITEGRAPH_PGPASSWORD_FILE` | a file that holds only the password |
| `CITEGRAPH_LOAD_WINDOW` | `HHMM-HHMM`: the hours in which a load may run; empty: any time |
| `CITEGRAPH_SWEEP_MAX_HOURS` | the daily sweep's time limit, default 3 |
| `CITEGRAPH_SWEEP_PROCESS_PATTERN` | `pgrep -f` pattern of a sweep running elsewhere, which makes `refresh.sh` skip the day |
| `citegraph_with_key CMD...` | function: run CMD with `OPENALEX_API_KEY` in its environment |
| `citegraph_notify MESSAGE` | function: called when a real run fails |

When the database is on another host, the load script runs there over ssh (`CITEGRAPH_REMOTE_CODE`)
and reads its own copy of the config (`CITEGRAPH_REMOTE_CONFIG`) for the PG settings and the
password file. `refresh.sh` counts the live rows by running `psql` on that host with its own copy of
`PGHOST PGPORT PGUSER PGDATABASE` and the password file named by `CITEGRAPH_PGPASSWORD_FILE`, read
there: the password never travels, only the file's name does. Write the path as `'~/.pgpw'`, in
single quotes, so that the `~` is the database host's home.

## Scheduling

[`systemd/`](systemd) has a service and a timer example: daily at 14:00, `nice` 19, idle I/O
class, six hours to finish. `refresh.sh` holds a lock, so a run still going at the next firing makes
that firing skip.

## Costs and rate limits

- **OpenAlex.** The key (optional; it raises the daily limit) is read from `OPENALEX_API_KEY` and
  sent only as the request header `api_key`, never in a URL, so it does not reach logs. A run asks
  only for DOIs and works the cache lacks: an estimate of 10 to 60 calls a day once the first build is done, against a limit of 10,000.
  Set `OPENALEX_MAILTO` to join the polite pool.
- **Semantic Scholar.** No key is needed and none is sent by the builder or the sweep. Requests go
  through one pacer: 2 s between calls, slowing by 1.5 times at each throttle up to 10 s and
  speeding back after 20 successes; a 429 or 5xx backs off exponentially from 4 s up to a cap and gives up on
  that request after a fixed number of retries (a paper that fails is recorded and tried again next run). A daily
  sweep of the papers that are due is an estimated 100 to 400 calls, so half an hour to two hours.
  `s2_citations_sync.py` needs a key (`S2_API_KEY`, sent only as the header `x-api-key`, only to
  `api.semanticscholar.org`).
- **First run.** The first full build and sweep of a library of several thousand papers takes
  many hours at the S2 pace. Run the sweep in a loop outside the timer (the timer skips while it
  runs; set `CITEGRAPH_SWEEP_PROCESS_PATTERN` to match your loop) until it reports no paper due.
- **Database.** A load writes a fresh `<schema>_next` (on the order of 100 MB for a library of ten thousand papers) and
  swaps it in; the server keeps answering throughout. `<schema>_prev` stays until you drop it
  (`drop_prev_<schema>.sql` in the build folder).

## Known limits

- Only outside works that at least two library papers cite are stored and looked up. A work cited
  once is not in `external_works`, and `/graph/papers/outside` says so ("2 or more").
- Some publishers withhold reference lists from S2, so a paper's own reference list is incomplete
  there. The sweep therefore also asks for each cited library paper's citers
  (`/paper/{id}/citations`), which are listed regardless; that is the cited-side sweep.
- The sweep visits a paper again when it is new, failed last time, is cited by a library paper
  added since, or was fetched more than 180 days ago.
- The dataset mode of `s2_citations_sync.py` (`--full`, `--diff`: stream S2's bulk citation
  dataset and keep the rows that touch the library) needs a key, downloads hundreds of gigabytes
  for a full pass, and is dormant: `refresh.sh` does not use it. The sweep writes a store of the
  same schema, so the builder reads either.
- `refresh.sh` counts the live tables before it loads, so the first load into a database that has
  no such schema yet is a manual `load_zstruct.sh --config FILE DIR`.
- Papers without a DOI are matched by title through S2 and accepted only when the title, the year (within one) and the first author's family name agree (`QA.md` lists the matches and the conflicts).

## Dry run

```bash
bash refresh.sh --config citegraph.env --dry-run
```

builds into `out.daily.dryrun/` (it makes the online requests, so it spends the daily call budget
like a real run), runs the gates against the live row counts, and copies and loads nothing; the
sweep and the sweep-running skip are off. The result is one line in the log:
`gates passed; would copy out.daily.dryrun to the database host as out.daily and load it: ...` or
`FAILED: ... -- not loaded`. A dry run notifies nobody. Delete `out.daily.dryrun/` afterwards.

## Tests

From `zotero-toolkit/`: `scripts/test.sh tests/citegraph`. They are offline. The shell scripts run
from a temp directory against stub `ssh`, `rsync`, `psql` and `pgrep` and a directory of stub
builder and sweep modules (`test_citegraph_scripts.py`); `test_refresh_gates.py` covers the gates
including the 98% boundary. `tests/citegraph/input/citations-sample.jsonl.gz` is a 72-line sample
of the public S2 citation dataset used by the dataset-mode tests.
