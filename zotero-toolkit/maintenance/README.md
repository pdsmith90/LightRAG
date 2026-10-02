# maintenance: repair dangling chunk references

`clean_dangling_refs.py` repairs a LightRAG workspace stored in PostgreSQL (pgvector +
Apache AGE) after documents disappeared without going through LightRAG's own delete path.

## The problem

Every entity and relation in LightRAG records the chunks it was extracted from. The same
chunk ids appear in four places:

| where | table / object | field |
| --- | --- | --- |
| entity vectors | `lightrag_vdb_entity_<model>_<dim>d` | `chunk_ids`, `file_path` |
| relation vectors | `lightrag_vdb_relation_<model>_<dim>d` | `chunk_ids`, `file_path` |
| chunk maps | `lightrag_entity_chunks`, `lightrag_relation_chunks` | `chunk_ids`, `count` |
| knowledge graph | AGE vertices and edges in `<workspace>_chunk_entity_relation` | `source_id`, `file_path` |

LightRAG's delete path rewrites all four. Some changes skip it: rows removed with SQL, a
document reingested under the same doc_id with fewer chunks, or an interrupted deletion.
Those changes leave ids of chunks that no longer exist in entities that also cite
surviving documents. Queries then log
`data inconsistency detected ... Falling back to WEIGHT method`, and retrieval for those
entities quietly degrades.

A reference is **dangling** when it has LightRAG's text-pipeline form
`doc-<32 hex>-chunk-<n>` and no row with that id exists in `lightrag_doc_chunks`. Ids in
any other form, such as custom-KG `chunk-<hash>` ids, are never touched.

## Install

In a virtual environment, from the `zotero-toolkit/` directory:

```bash
python -m pip install -r maintenance/requirements.txt   # asyncpg; Python >= 3.10
```

Built against the schema of lightrag-hku 1.5.7 (PGKVStorage, PGDocStatusStorage,
PGVectorStorage, PGGraphStorage).

## Quick start

Point the tool at the `.env` your LightRAG server uses and look before you write:

```bash
python maintenance/clean_dangling_refs.py --env-file /path/to/lightrag/.env --sweep --dry-run
python maintenance/clean_dangling_refs.py --env-file /path/to/lightrag/.env --sweep
python maintenance/clean_dangling_refs.py --env-file /path/to/lightrag/.env --sweep --dry-run   # expect 0s
```

Every run starts by printing the workspace, graph and vector tables it resolved.

## Modes

| mode | writes | what it does |
| --- | --- | --- |
| `--sweep --dry-run` | nothing | counts rows that hold dangling references, per table |
| `--sweep` | in place | removes dangling ids from each row's **current** value, set-based, in one transaction |
| `--scan` | plan file only | read-only scan; writes every intended change to the plan file (`--plan`) |
| `--show` | nothing | prints the summary of the current plan file |
| `--commit` | yes | applies the plan in one transaction, then renames it to `<plan>.done` |

**`--sweep`** is idempotent and converges. Each statement filters the row as it is at
that moment, so a row that extraction rewrote in the meantime is still cleaned, and
chunks added meanwhile are kept. Recomputing `file_path` works the same way. The sweep
does not delete anything: rows left with no chunks at all are reported
(`rows left with NO chunks`) so you can inspect them.

**`--scan` / `--commit`** is the auditable path. The plan lists every change with the old
and new values. The commit also deletes rows left with no surviving chunk: vector row, map
row, graph vertex with every edge touching it, or graph edge. Each update or delete is
compare-and-set on the old value. A row that changed since the scan is skipped and counted
as `skipped` rather than overwritten; a later `--sweep` or a fresh scan picks it up. The
commit refuses a plan scanned in a different workspace or graph (exit 2).

### Backups

Before deleting anything, `--commit` copies every row it will delete into new tables, in
the same transaction:

- `lightrag_vdb_entity_bak_dangling_<YYYYmmdd_HHMMSS>`, `lightrag_vdb_relation_bak_dangling_…`,
  `lightrag_entity_chunks_bak_dangling_…` and `lightrag_relation_chunks_bak_dangling_…`
  go in the schema LightRAG's tables live in.
- `"<graph>".bak_dangling_ag_label_vertex_<stamp>` and `…_ag_label_edge_<stamp>` go inside
  the graph's own schema.

Updated rows are not copied; their old values are in the plan file, kept as
`<plan>.done`. `--sweep` makes no backups. If you want an undo record, use
`--scan`/`--commit`, or take a `pg_dump` first.

## Safety

- **Idle check.** `--commit` and `--sweep` (without `--dry-run`) first ask the LightRAG
  server `GET /documents/pipeline_status`. They refuse with **exit 3** unless it answers
  `busy: false`, because extraction merges into exactly the rows being repaired. An
  unreachable server or an HTTP error is exit 1, never "idle". `--force` skips the check.
  Use it only when no LightRAG server is writing to this database, for example with
  library-only use.
- **One transaction** per commit or sweep: it applies completely or not at all.
- **Scan and dry run are read-only** and safe at any time.
- **Credentials.** The password is never printed. The API key travels only in the
  `X-API-Key` header, never in a URL.
- **Order matters on reingest.** Chunk ids are per-document positions
  (`doc-<hash>-chunk-003`). Once a deleted document is reingested under the same doc_id,
  a stale reference to one of its chunk numbers looks exactly like a live one. Sweep
  after deleting and **before** re-adding the same documents.

## Configuration

Settings are resolved in this order: command-line flag, then process environment, then
`--env-file`. The environment winning over the file matches how LightRAG's server loads
its `.env`. The variable names are LightRAG's own, so its `.env` works unchanged.

| setting | flag | environment | default |
| --- | --- | --- | --- |
| settings file | `--env-file PATH` | | none: only the environment is read |
| database host | | `POSTGRES_HOST` | `localhost` |
| database port | | `POSTGRES_PORT` | `5432` |
| database user | | `POSTGRES_USER` | `postgres` |
| database password | | `POSTGRES_PASSWORD` | none: libpq's `PGPASSWORD` / `~/.pgpass` |
| database name | | `POSTGRES_DATABASE` | `postgres` |
| workspace | `--workspace` | `POSTGRES_WORKSPACE`, else `WORKSPACE` | `default` |
| vector tables | `--embedding-model`, `--embedding-dim` | `EMBEDDING_MODEL` + `EMBEDDING_DIM` | detected, see below |
| LightRAG server | `--api-url` | `LIGHTRAG_BASE_URL`, else `PORT` | `http://localhost:9621` |
| API key | | `LIGHTRAG_API_KEY` | none: no header sent |
| plan file | `--plan PATH` | | `./dangling_refs_plan.json` |

- **Workspace** follows LightRAG's precedence: `POSTGRES_WORKSPACE` overrides the server's
  `WORKSPACE`, and an empty workspace means `default`.
- **Graph name** is derived exactly as `PGGraphStorage` derives it. For the workspace
  `default` it is `chunk_entity_relation`. Otherwise it is
  `<workspace>_chunk_entity_relation`, with every character outside `[A-Za-z0-9_]`
  replaced by `_`, case preserved, and the name cut to PostgreSQL's 63 bytes.
- **Vector tables** are `lightrag_vdb_entity_<model>_<dim>d` and
  `lightrag_vdb_relation_<model>_<dim>d`, where `<model>` is `EMBEDDING_MODEL` lowercased
  with every character outside `[a-z0-9_]` replaced by `_`. When `EMBEDDING_MODEL` and
  `EMBEDDING_DIM` are not both set, the server may have used a provider default, so the
  tool looks instead. Among the tables named `lightrag_vdb_entity` or
  `lightrag_vdb_entity_<model>_<dim>d`, it takes the one that holds rows of the workspace,
  or the only one there is. When that is ambiguous it exits and asks for the model.
  Unsuffixed tables are what LightRAG creates when `EMBEDDING_MODEL` is unset.
- **SSL** and other connection options follow asyncpg/libpq environment variables
  (`PGSSLMODE`, `PGSSLROOTCERT`, …).
- **Authentication.** The idle check authenticates with `LIGHTRAG_API_KEY` only. A
  server that accepts nothing but account logins answers 401, which is exit 1.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | done (or nothing to do) |
| 1 | error: server unreachable or HTTP error, database error, ambiguous vector tables |
| 2 | `--commit`/`--show` without a plan, or a plan from another workspace |
| 3 | pipeline busy, nothing written |

## Waiting for an idle pipeline: `run_when_idle.sh`

A busy pipeline makes `--sweep`/`--commit` exit 3 without writing. `run_when_idle.sh`
retries until the tool gets through, then exits with the tool's own status. Its arguments
go to `clean_dangling_refs.py`; without a mode flag (`--scan`, `--commit`, `--show`,
`--sweep`) it adds `--sweep`. Before the first attempt it checks that its interpreter
(`PYTHON`) can import asyncpg, and exits 1 with a message naming that interpreter if not.

```bash
maintenance/run_when_idle.sh --env-file /path/to/lightrag/.env              # sweep
IDLE_RETRY_INTERVAL=15 maintenance/run_when_idle.sh --env-file /path/to/lightrag/.env \
    --commit --plan /path/to/dangling_refs_plan.json                      # apply a plan
```

| variable | default | meaning |
| --- | --- | --- |
| `IDLE_RETRY_INTERVAL` | `300` | seconds between attempts; use a short one for `--commit`, whose plan goes stale while the pipeline keeps writing |
| `IDLE_MAX_WAIT` | `21600` | stop after this many seconds and exit 3 |
| `PYTHON` | `python3` | interpreter to run the tool with; it must be able to import asyncpg |

Only one instance runs at a time, where `flock(1)` exists. Log lines go to stdout, so
redirect them where you want them.

cron and systemd start the wrapper without activating your virtual environment, and cron
with a minimal `PATH`, so a bare `python3` there is the system interpreter, which usually
lacks asyncpg. Set `PYTHON` to the virtual environment's own interpreter. For a scheduled
backstop sweep, something like:

```cron
@daily  PYTHON=/path/to/venv/bin/python /path/to/LightRAG/zotero-toolkit/maintenance/run_when_idle.sh --env-file /path/to/lightrag/.env >> /path/to/sweep.log 2>&1
```

## Finding glyph-code documents: `pua_scan.py`

A scanned PDF whose fonts map every glyph into the Unicode Private Use Area (U+E000-F8FF
and planes 15-16) passes every text extractor as text, so it can reach the knowledge
base as thousands of chunks of glyph codes that extract nothing and keep the ingest
pipeline busy for hours. `corpus/build_corpus.py` now routes such text layers to OCR and
`corpus/quarantine_junk.py` catches them at the corpus root; `pua_scan.py` is the audit
for documents that were ingested before those checks existed.

```bash
python maintenance/pua_scan.py --env-file /path/to/lightrag/.env
```

Read-only. One pass over `lightrag_doc_full` of the workspace, joined to
`lightrag_doc_status`, prints one `WARN` line per document whose PUA code points make up
at least `--min-ratio` (default `0.30`) of its content, with its status, chunk count and
length, then a summary line. Garbled scans score 0.70-0.95; genuine documents with a
symbol font stay under 0.05. It needs only `POSTGRES_*` and the workspace from the
configuration above — not the vector tables or the server — and it never writes.

| code | meaning |
| --- | --- |
| 0 | no document at or above the bar |
| 1 | documents reported |
| 2 | database error |

To remove a reported document, use LightRAG's own delete path (`DELETE
/documents/delete_document` with its id; it rebuilds the entities the document shared
with others, which takes minutes for a large book), then either add its storage key to
the corpus denylist so the next build does not regenerate it, or remove its parsed
output and let the build re-convert it through OCR.

## Tests

From the `zotero-toolkit/` directory, in a virtualenv with `maintenance/requirements.txt` and `pytest`:

```bash
python -m pytest tests/maintenance
```

The unit tests run offline (`pua_scan.py`'s replace the database driver with a fake). The integration test
(`tests/maintenance/test_clean_dangling_refs_pg.py`) is skipped unless
`CLEAN_DANGLING_TEST_DSN` points at a **scratch** PostgreSQL database with Apache AGE and
`lightrag-hku==1.5.7` is installed. It writes a workspace through LightRAG's own storages,
removes one document behind LightRAG's back, and checks three things. First, that
`--sweep --dry-run` counts the dangling rows. Second, that `--sweep` repairs them and a
second dry run reports 0. Third, that `--scan`/`--commit` backs up, deletes and skips as
described above. pgvector is not needed: the test types the vector tables' embedding
column as `REAL[]`, a column the tool never reads.

```bash
podman run -d --name lightrag-age-test -p 127.0.0.1:55433:5432 \
    -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=test \
    docker.io/apache/age:release_PG15_1.6.0
python -m pip install lightrag-hku==1.5.7
CLEAN_DANGLING_TEST_DSN=postgresql://test:test@127.0.0.1:55433/test python -m pytest tests/maintenance
podman rm -f lightrag-age-test
```
