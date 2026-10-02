# zotero-toolkit

Scripts for running [LightRAG](https://github.com/HKUDS/LightRAG) as a research knowledge
base over a [Zotero](https://www.zotero.org) library. They turn the library into a clean
Markdown corpus that cites back to Zotero, give agents an MCP tool that answers with real
bibliographic citations, repair a PostgreSQL + Apache AGE deployment after out-of-band
deletions, and keep queries working when a model server is busy or down.

This toolkit is the `zotero-toolkit/` directory of `zlightrag`, the default branch of the
[pdsmith90/LightRAG](https://github.com/pdsmith90/LightRAG) fork. It sits beside the
fork's LightRAG code but neither imports nor patches it, and it works with a stock
LightRAG Server. Every path in it is relative to this directory, so the commands in this
README and the component READMEs run from `zotero-toolkit/`.

## The problem

LightRAG builds a knowledge graph and vector index from text and answers questions over
it. Pointing it at a Zotero library leaves several gaps:

- **Zotero stores files, LightRAG needs text.** A Zotero data directory is a SQLite
  database plus `storage/<KEY>/<attachment>` folders of PDFs, web snapshots, EPUBs and
  scans. PDFs need reading-order extraction, scans need OCR, and saved web pages are
  mostly markup and base64.
- **Reference lists pollute the graph.** Every paper ends in dozens of citations. Indexed
  as-is, they turn into entities and relations that crowd out the paper's content.
- **Answers need real citations.** LightRAG reports which files it retrieved, but a file
  name is not a citation, and the answering model tends to write its own reference list
  with wrong years or invented titles.
- **Agents need a tool interface.** Coding and research agents talk MCP, not LightRAG's
  REST API.
- **PostgreSQL workspaces drift.** Documents removed outside LightRAG's own delete path
  (SQL, an interrupted deletion, a reingest with fewer chunks) leave chunk ids behind in
  entities and relations, and retrieval for those entities quietly degrades.
- **One model server per role, no fallback.** When the LLM server for a role fails, every
  call of that role fails. When the reranker fails, LightRAG answers from unranked chunks
  and the answer looks normal.

## How the pieces fit

```text
Zotero data directory (zotero.sqlite + storage/)
    │
    │  corpus/build_metadata.py   ──▶  zotero_metadata.json
    │  corpus/build_corpus.py     ──▶  rag_corpus/<KEY>__<slug>.md
    │  corpus/quarantine_junk.py       optional: moves web-snapshot and glyph-code junk aside
    ▼
rag_corpus/  ──INPUT_DIR, scan──▶  LightRAG Server  ──▶  LLM and reranker servers
                                      ▲       │         (optionally via fronts/:
                                      │       ▼         primary, else fallback)
           REST: /query, /query/data, │   PostgreSQL + pgvector + Apache AGE
           /documents/text            │       ▲
                                      │       └── maintenance/clean_dangling_refs.py, pua_scan.py
                          mcp/lightrag_mcp_server.py  ◀──  zotero_metadata.json
                                      ▲
                                      │  MCP over stdio
                          agent (Claude Code, Claude Desktop, any MCP client)
```

| Component | What it does | Runtime needs |
| --- | --- | --- |
| [`corpus/`](corpus/README.md) | Converts every Zotero attachment into Markdown with a metadata header, strips reference lists and page furniture, and names each file `<KEY>__<slug>.md` after its Zotero storage key. Resumable and parallel. | Python ≥ 3.10, `pymupdf4llm`; optional format handlers, `pandoc`, `djvutxt`, `ocrmypdf` |
| [`mcp/`](mcp/README.md) | A stdio MCP server that proxies the LightRAG Server API. It compiles citations (authors, year, title, venue, DOI, `zotero://` link) from LightRAG's `references` array and `zotero_metadata.json`, and replaces any reference list the answering model wrote. | `mcp` 2.x, `httpx`; LightRAG Server ≥ 1.4.9 |
| [`maintenance/`](maintenance/README.md) | Finds and removes references to chunks that no longer exist, across the vector tables, chunk maps and AGE graph of a PostgreSQL workspace. Dry run, in-place sweep, or an auditable scan/commit with backups. Refuses to write while the ingest pipeline is busy. Also lists documents whose stored text is glyph codes from a font without a Unicode mapping (`pua_scan.py`). | `asyncpg`; the schema of lightrag-hku 1.5.7 |
| [`fronts/`](fronts/README.md) | Two small HTTP fronts, one for OpenAI-compatible chat completions and one for `/v1/rerank`. Each tries a primary backend and replays the request against a fallback, with a circuit breaker and health endpoint. | Python standard library only |

The components are independent. Each is a directory of scripts with its own README and
`requirements.txt`, and nothing is installed as a package. The only coupling is a data
contract: corpus files are named `<KEY>__<slug>.md`, and the MCP server looks `KEY` up in
the `zotero_metadata.json` that `corpus/build_metadata.py` writes. Keep that naming if you
build the corpus some other way.

## Quick start

Clone the fork and work in the toolkit's directory. The commands below run from
`zotero-toolkit/` in a virtual environment. Each component README has the full details.

```bash
git clone https://github.com/pdsmith90/LightRAG.git
cd LightRAG/zotero-toolkit
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install -r corpus/requirements.txt -r mcp/requirements.txt
```

To download only the toolkit, clone sparsely instead. Git then checks out
`zotero-toolkit/` and the files at the repository root, and fetches no other file
contents:

```bash
git clone --filter=blob:none --sparse https://github.com/pdsmith90/LightRAG.git
cd LightRAG
git sparse-checkout set zotero-toolkit
cd zotero-toolkit
```

1. **Build the metadata and the corpus.** Zotero locks `zotero.sqlite` while it runs, so
   close it first or point `--zotero` at a copy of the data directory.

   ```bash
   python3 corpus/build_metadata.py --zotero ~/Zotero
   python3 corpus/build_corpus.py --zotero ~/Zotero --jobs 8 --limit 25   # pilot run
   python3 corpus/build_corpus.py --zotero ~/Zotero --jobs 8              # everything
   python3 corpus/quarantine_junk.py --corpus ~/Zotero/rag_corpus         # optional
   ```

   This writes `~/Zotero/zotero_metadata.json` and `~/Zotero/rag_corpus/`.

2. **Index the corpus with LightRAG.** Install and configure the server as LightRAG's
   documentation describes (`pip install "lightrag-hku[api]"`, a `.env`, then
   `lightrag-server`). Set `INPUT_DIR` to the corpus directory and start a scan from the
   WebUI or with `POST /documents/scan`. Set `LIGHTRAG_API_KEY` if the server is reachable
   by anyone but you.

3. **Give your agent the MCP tools.** For Claude Code:

   ```bash
   claude mcp add zotero-lightrag \
     -e LIGHTRAG_BASE_URL=http://localhost:9621 \
     -e LIGHTRAG_API_KEY=your-lightrag-api-key \
     -e ZOTERO_METADATA=/absolute/path/to/zotero_metadata.json \
     -- /absolute/path/to/.venv/bin/python /absolute/path/to/mcp/lightrag_mcp_server.py
   ```

   The agent then has `search_library`, `get_sources`, `add_note` and `health`.

4. **Optional.** Put [`fronts/`](fronts/README.md) between LightRAG and its model servers
   if you have a second place to run a model. Run
   [`maintenance/clean_dangling_refs.py`](maintenance/README.md) with `--sweep --dry-run`
   if your workspace lives in PostgreSQL and documents were ever removed behind LightRAG's
   back, and [`maintenance/pua_scan.py`](maintenance/README.md#finding-glyph-code-documents-pua_scanpy)
   once, to find documents ingested from scanned PDFs whose text layer is glyph codes.
   Both need their own requirements in the same virtual environment first:
   `python3 -m pip install -r maintenance/requirements.txt`.

When the library changes, re-run step 1 and scan again. Only new or changed attachments
are converted again, and the scan indexes the new ones but not a changed one: LightRAG
identifies a document by its file name and sets aside a file whose name it has already
processed. To re-index a changed attachment, or everything a `--force` rebuild produced,
first delete the old document in LightRAG together with its files, then rebuild and scan;
[corpus/README.md](corpus/README.md#re-indexing-a-changed-attachment) has the steps.
Restart the MCP server after regenerating `zotero_metadata.json`, because it reads the
file once at startup.

## Configuration

Every component is configured with command-line flags or environment variables.

| Component | Configured by | Main settings | Reference |
| --- | --- | --- | --- |
| corpus | command-line flags | `--zotero`, `--out`, `--meta`, `--jobs`, `--ocr`, `--denylist` | [corpus/README.md](corpus/README.md#settings) |
| mcp | environment variables in the MCP client's server entry | `LIGHTRAG_BASE_URL`, `LIGHTRAG_API_KEY`, `ZOTERO_METADATA`, `LIGHTRAG_MCP_MAX_TOP_K`, timeouts | [mcp/README.md](mcp/README.md#configuration) |
| maintenance | flags, then the environment, then `--env-file` (LightRAG's own `.env` works unchanged) | `POSTGRES_*`, `POSTGRES_WORKSPACE`/`WORKSPACE`, `EMBEDDING_MODEL` + `EMBEDDING_DIM`, `LIGHTRAG_BASE_URL`, `LIGHTRAG_API_KEY` | [maintenance/README.md](maintenance/README.md#configuration) |
| fronts | environment variables, optionally from an env file the launcher sources | `LLM_FO_PRIMARY`, `LLM_FO_FALLBACK`, `LLM_FO_FALLBACK_API`, `RERANK_FO_PRIMARY`, `RERANK_FO_FALLBACK`, timeouts, breaker | [fronts/README.md](fronts/README.md#settings-llm_failoverpy) |

Two settings are shared:

- **`ZOTERO_METADATA`** is read by `build_metadata.py`, `build_corpus.py` and the MCP
  server. Unset, the corpus scripts use `<zotero>/zotero_metadata.json` and the MCP server
  uses `~/Zotero/zotero_metadata.json`, which agree for the default `--zotero ~/Zotero`.
  If your data directory is elsewhere, set it to one absolute path so all three agree.
- **`LIGHTRAG_API_KEY`** is the value from the LightRAG server's `.env`. The MCP server and
  the maintenance idle check send it only as the `X-API-Key` header, never in a URL.

## Requirements

- Python 3.10 or newer.
- A LightRAG Server. The MCP server needs v1.4.9 or newer (the first release whose
  `/query` returns `references`) and was verified against an unmodified v1.5.7. The
  maintenance tool follows the storage schema of lightrag-hku 1.5.7, and the LightRAG
  settings named in the fronts' README are those of 1.5.7.
- For `maintenance/`, LightRAG's PostgreSQL storage (`PGKVStorage`, `PGVectorStorage`,
  `PGDocStatusStorage`, `PGGraphStorage`) with Apache AGE.
- Optional system tools: `pandoc`, `djvutxt` (djvulibre) and `ocrmypdf` (with Tesseract
  and Ghostscript) for the corpus builder; llama.cpp's `llama-server` for the example
  fallback reranker.
- The shell launchers are bash scripts. The toolkit was developed and tested on Linux.

## Tests

From the `zotero-toolkit/` directory:

```bash
python3 -m pip install -r requirements-dev.txt
scripts/test.sh            # extra arguments go to pytest, e.g. scripts/test.sh tests/mcp -x
```

`python3 -m pytest tests` works too. The toolkit keeps no `.github/` directory of its
own: continuous integration runs this suite from a workflow at the root of the
`zlightrag` branch.

The suite runs offline. The LightRAG API and the model servers are faked with stdlib HTTP
servers on loopback ports, and the corpus tests build a small synthetic Zotero data
directory. The PDF part of the corpus end-to-end test is skipped when PyMuPDF or
pymupdf4llm is not installed. The
PostgreSQL + Apache AGE integration test in `tests/maintenance` runs only when
`CLEAN_DANGLING_TEST_DSN` points at a scratch database; see
[maintenance/README.md](maintenance/README.md#tests).

## Limitations

- **Personal library only.** `zotero://select/library/items/<KEY>` links select items in
  your personal library. Group-library items need a `zotero://select/groups/<id>/...` link,
  which nothing here generates. Linked files that live outside `storage/` are not
  converted.
- **Zotero must be closed** (or the data directory copied) while `build_metadata.py`
  reads `zotero.sqlite`.
- **OCR.** `--ocr force` behaves like `auto`: OCR runs only on PDFs without a usable text
  layer. Legacy `.ppt` files come out with an empty body.
- **Reference stripping is heuristic.** It was tuned on scientific journal articles,
  books and theses in many citation styles. A safety valve keeps a document whole rather
  than emptying it when most of it looks like bibliography.
- **The MCP server authenticates with an API key only.** A LightRAG server protected only
  by account login needs a session token, which the proxy does not obtain.
- **`maintenance/` supports PostgreSQL storage only**, and it relies on the table and graph
  layout of lightrag-hku 1.5.7. Other LightRAG storage backends are not supported. The
  in-place `--sweep` keeps no backup; use `--scan`/`--commit` or `pg_dump` if you want one.
- **The fronts have no authentication.** Keep them on loopback. A streamed response that
  has already started cannot fall back, and the ollama fallback does not forward `tools`,
  `stop` or `top_p`.

## Relationship to LightRAG

This is an independent companion project to
[LightRAG](https://github.com/HKUDS/LightRAG) (HKUDS, MIT License). It lives in a LightRAG
fork, but the toolkit itself neither vendors nor modifies LightRAG, and it needs no changes
to it. It prepares input files for a stock LightRAG Server, talks to that server through
its public REST API, reads and repairs the PostgreSQL tables that LightRAG's own storage
classes create, and sits in front of the model servers LightRAG calls. It is not affiliated with or endorsed by the LightRAG
authors. LightRAG's REST responses and storage schema can change between releases; the
versions these tools were checked against are listed under [Requirements](#requirements).

## License

MIT. See [LICENSE](LICENSE).
