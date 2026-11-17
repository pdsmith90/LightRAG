"""Real-PostgreSQL smoke test for the paper structure (lightrag/zotero_struct.py).

The offline suite answers zotero_struct's SQL from a stub; this runs it against a
live server: the catalog check, the schema reads, and the document-status filter
on the real LIGHTRAG_DOC_STATUS table that PGDocStatusStorage creates. Each test
builds a small structure schema of its own (the table shapes ZLIGHTRAG.md lists),
two workspaces of document-status rows, and drops both afterwards.

Carries ``pg_smoke`` and ``integration`` like the other live-PostgreSQL tests, so
it runs only with ``--run-integration`` and POSTGRES_PASSWORD (plus the other
POSTGRES_* settings) naming a scratch database.
"""

import os
import uuid
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.pg_smoke, pytest.mark.integration]

if not os.environ.get("POSTGRES_PASSWORD", ""):
    pytest.skip(
        "POSTGRES_PASSWORD not set — skipping pg_smoke tests", allow_module_level=True
    )

import lightrag.zotero_struct as zs  # noqa: E402

DDL = """
CREATE SCHEMA {s};
CREATE TABLE {s}.venues (venue_id text PRIMARY KEY, display_name text,
  openalex_source_id text, n_papers integer);
CREATE TABLE {s}.topics (topic_id text PRIMARY KEY, display_name text, level text,
  parent_id text REFERENCES {s}.topics (topic_id));
CREATE TABLE {s}.papers (paper_id text PRIMARY KEY, doi text, openalex_id text,
  title text, year integer, venue_id text REFERENCES {s}.venues (venue_id), type text,
  cited_by_count integer, n_refs_openalex integer, n_attachments integer);
CREATE TABLE {s}.paper_attachments (attachment_key text PRIMARY KEY, item_key text,
  paper_id text REFERENCES {s}.papers (paper_id));
CREATE TABLE {s}.authors (author_id text PRIMARY KEY, display_name text,
  openalex_id text, n_papers integer);
CREATE TABLE {s}.paper_authors (paper_id text NOT NULL, author_id text,
  position integer NOT NULL, source text, PRIMARY KEY (paper_id, position));
CREATE TABLE {s}.paper_topics (paper_id text NOT NULL, topic_id text NOT NULL,
  score double precision, is_primary boolean, PRIMARY KEY (paper_id, topic_id));
CREATE TABLE {s}.cites (citing_paper_id text NOT NULL, cited_paper_id text NOT NULL,
  PRIMARY KEY (citing_paper_id, cited_paper_id));
CREATE TABLE {s}.coupled (paper_a text NOT NULL, paper_b text NOT NULL,
  shared_refs integer, score double precision, PRIMARY KEY (paper_a, paper_b));
INSERT INTO {s}.venues VALUES ('v:1', 'Journal of Invented Results', NULL, 3);
INSERT INTO {s}.topics VALUES
  ('domain:3', 'Physical Sciences', 'domain', NULL),
  ('field:19', 'Earth and Planetary Sciences', 'field', 'domain:3'),
  ('subfield:1908', 'Geophysics', 'subfield', 'field:19'),
  ('topic:T1', 'Gravity Measurements', 'topic', 'subfield:1908');
INSERT INTO {s}.papers (paper_id, doi, title, year, venue_id, cited_by_count) VALUES
  ('doi:10.1/a', '10.1/a', 'Alpha', 2010, 'v:1', 5),
  ('doi:10.1/b', '10.1/b', 'Bravo', 2012, 'v:1', 9),
  ('doi:10.1/c', '10.1/c', 'Charlie', 2014, 'v:1', 1),
  ('zot:ITEM0001', NULL, 'A report', NULL, NULL, NULL);
INSERT INTO {s}.paper_attachments VALUES
  ('AAAA0001', 'I1', 'doi:10.1/a'), ('BBBB0001', 'I2', 'doi:10.1/b'),
  ('CCCC0001', 'I3', 'doi:10.1/c'), ('KKKK0001', 'ITEM0001', 'zot:ITEM0001');
INSERT INTO {s}.authors VALUES ('oa:1', 'Ada Okafor', NULL, 2);
INSERT INTO {s}.paper_authors VALUES
  ('doi:10.1/a', 'oa:1', 1, 'openalex'), ('doi:10.1/c', 'oa:1', 1, 'openalex');
INSERT INTO {s}.paper_topics VALUES
  ('doi:10.1/a', 'topic:T1', 0.9, true), ('doi:10.1/b', 'topic:T1', 0.8, true);
INSERT INTO {s}.cites VALUES ('doi:10.1/a', 'doi:10.1/b'), ('doi:10.1/c', 'doi:10.1/a');
INSERT INTO {s}.coupled VALUES ('doi:10.1/b', 'doi:10.1/c', 4, 0.3);
"""


@pytest.fixture(scope="session", autouse=True)
def init_shared_storage():
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)


@pytest.fixture
async def rag(monkeypatch):
    from lightrag.kg.postgres_impl import PGDocStatusStorage

    tag = uuid.uuid4().hex[:8]
    workspace = f"smoke_{tag}"
    schema = f"zs_smoke_{tag}"
    store = PGDocStatusStorage(
        namespace="doc_status",
        workspace=workspace,
        global_config={},
        embedding_func=None,
    )
    await store.initialize()
    db = store.db
    if store.workspace != workspace:
        # POSTGRES_WORKSPACE (environment, .env or config.ini) overrides the
        # storage's workspace: the rows below would land in, and be deleted
        # from, that workspace.
        await store.finalize()
        pytest.skip(
            "POSTGRES_WORKSPACE is set, so the storage ignores this test's own "
            "workspace; unset it to run the smoke test"
        )
    # rows of two workspaces, with ids of this test's own
    rows = [
        (store.workspace, f"{tag}-doc-a", "AAAA0001__alpha.md", "processed"),
        (store.workspace, f"{tag}-doc-c", "CCCC0001__charlie.md", "failed"),
        (store.workspace + "_other", f"{tag}-doc-b", "BBBB0001__bravo.md", "processed"),
    ]
    inserted = []
    monkeypatch.setenv(zs.SCHEMA_ENV, schema)
    zs._reset_for_tests()
    try:
        for statement in DDL.format(s=schema).split(";"):
            if statement.strip():
                await db.execute(statement)
        for w, doc_id, path, status in rows:
            await db.execute(
                "INSERT INTO LIGHTRAG_DOC_STATUS (workspace, id, file_path, status) "
                "VALUES ($1, $2, $3, $4)",
                {"workspace": w, "id": doc_id, "file_path": path, "status": status},
            )
            inserted.append((w, doc_id))
        yield SimpleNamespace(doc_status=store, schema=schema, db=db)
    finally:
        zs._reset_for_tests()
        await db.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        for w, doc_id in inserted:  # only the rows inserted above
            await db.execute(
                "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = $1 AND id = $2",
                {"workspace": w, "id": doc_id},
            )
        await store.finalize()


async def test_reads_the_schema_and_the_servers_processed_documents(rag):
    assert await zs.available(rag)
    result = await zs.related_papers(rag, "AAAA0001")
    related = {r["paper_id"]: r for r in result["related"]}
    assert result["paper"]["in_kb"] and result["paper"]["key"] == "AAAA0001"
    assert related["doi:10.1/b"]["reasons"] == ["cites"]
    assert related["doi:10.1/c"]["reasons"] == ["cited_by", "coauthor"]
    # b is processed only in another workspace, c failed in this one
    assert not related["doi:10.1/b"]["in_kb"] and not related["doi:10.1/c"]["in_kb"]


async def test_paths_topics_and_map(rag):
    connect = await zs.connect_papers(rag, "10.1/a", "10.1/c")
    assert connect["paths"][0]["hops"] == 1
    assert [p["hops"] for p in connect["paths"]] == [1, 2]  # a-b-c by coupling
    topics = await zs.papers_by_topic(rag, "field:19")
    assert [p["paper_id"] for p in topics["papers"]] == ["doi:10.1/b", "doi:10.1/a"]
    graph = await zs.paper_map(rag, ["AAAA0001"])
    assert {n["id"] for n in graph["nodes"]} == {"10.1/a", "10.1/b", "10.1/c"}
    assert len(graph["edges"]) == 3


async def test_a_rebuilt_schema_is_read_again(rag, monkeypatch):
    first = await zs.related_papers(rag, "10.1/a")
    assert first["totals"]["papers"] == 2
    await rag.db.execute(f"TRUNCATE {rag.schema}.cites")
    monkeypatch.setattr(zs, "_CHECK_INTERVAL", 0.0)
    again = await zs.related_papers(rag, "10.1/a")
    assert again["totals"]["papers"] == 1  # c, now only a co-author


async def test_a_missing_schema_is_unavailable(rag, monkeypatch):
    monkeypatch.setenv(zs.SCHEMA_ENV, "zs_smoke_absent")
    zs._reset_for_tests()
    assert not await zs.available(rag)
    with pytest.raises(zs.Unavailable) as e:
        await zs.related_papers(rag, "10.1/a")
    assert e.value.reason == "no_schema"
