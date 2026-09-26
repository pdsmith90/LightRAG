#!/usr/bin/env python3
"""clean_dangling_refs.py — strip references to chunks that no longer exist.

A document removed from a LightRAG PostgreSQL workspace without going through
LightRAG's delete path (or reingested under the same doc_id with fewer chunks) leaves
its chunk ids inside entities and relations that ALSO cite surviving documents.
Queries then log "data inconsistency detected ... Falling back to WEIGHT method" and
retrieval silently degrades for those entities. This does what LightRAG's own
per-chunk cleanup would do:

  * vdb entity / relation rows : drop dead chunk ids from chunk_ids, recompute file_path
  * entity_chunks / relation_chunks (KV maps) : same, and fix `count`
  * AGE graph vertices / edges  : same on the source_id / file_path properties
  * anything left with NO surviving chunk is deleted outright (vdb row, map row,
    vertex together with its edges, or edge) -- by --commit; --sweep reports them

A missing chunk is a `doc-<32 hex>-chunk-` id with no lightrag_doc_chunks row,
including after reingest under the same doc_id. Chunk ids of any other form are kept.

Two phases so the write is short enough to fit an idle window:

  --scan    read-only DB scan. Computes every change and writes the plan file (--plan).
  --commit  applies the plan by primary key in ONE transaction, compare-and-set on the
            old value (a row changed since the scan is skipped and reported). Every
            row it deletes is first copied into a backup table. Refuses unless
            /documents/pipeline_status says busy=false (override: --force), because
            extraction merges into exactly these rows. Exit 3 = not idle.

--sweep is the set-based alternative that converges on rows extraction keeps
rewriting (see sweep()). Configuration (POSTGRES_*, workspace, vector tables, API
URL and key) is described in README.md; `--help` lists the flags.
"""
import argparse, asyncio, json, os, re, sys, time, urllib.request

import asyncpg

SEP = "<SEP>"
CHUNK_RE = re.compile(r"^(doc-[0-9a-f]{32})-chunk-\d+$")
MISSING = ("c ~ '^doc-[0-9a-f]{32}-chunk-' and not exists (select 1 from lightrag_doc_chunks d"
           " where d.workspace=$1 and d.id=c)")
T_ECH = "lightrag_entity_chunks"
T_RCH = "lightrag_relation_chunks"
VDB_BASES = ("lightrag_vdb_entity", "lightrag_vdb_relation")
GRAPH_NAMESPACE = "chunk_entity_relation"
PG_NAME_MAX = 63  # bytes in a PostgreSQL identifier; AGE clips graph names to it

# Set by configure(); these defaults are what an unconfigured LightRAG uses.
ENV = {}            # values from --env-file; the process environment wins over them
WS = "default"
GRAPH = GRAPH_NAMESPACE
T_ENT = T_REL = None  # vector tables; resolved in connect() unless configured
PLAN = "dangling_refs_plan.json"
API_URL = "http://localhost:9621"


def load_env(path):
    """Parse a dotenv file the way LightRAG's server reads its .env (python-dotenv)."""
    env = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            elif " #" in v:
                v = v.split(" #", 1)[0].rstrip()
            env[k.strip()] = v
    return env


def setting(name, default=None):
    """Process environment first, then --env-file (LightRAG loads .env with override=False)."""
    v = os.environ.get(name)
    if v is None:
        v = ENV.get(name)
    return default if v in (None, "") else v


def graph_name(workspace):
    """The AGE graph LightRAG's PGGraphStorage uses for a workspace."""
    ws = (workspace or "").strip()
    if ws and ws.lower() != "default":
        name = f"{re.sub(r'[^a-zA-Z0-9_]', '_', ws)}_{GRAPH_NAMESPACE}"
    else:
        name = GRAPH_NAMESPACE
    return name[:PG_NAME_MAX]


def vdb_suffix(model, dim):
    """Table suffix LightRAG's PGVectorStorage derives from the embedding model."""
    return f"{re.sub(r'[^a-zA-Z0-9_]', '_', model.strip().lower())}_{int(dim)}d"


def short_name(table):
    """lightrag_vdb_entity_<model>_<dim>d -> lightrag_vdb_entity; other tables unchanged."""
    for base in VDB_BASES:
        if table.startswith(base):
            return base
    return table


def configure(a):
    global ENV, WS, GRAPH, T_ENT, T_REL, PLAN, API_URL
    ENV = load_env(a.env_file) if a.env_file else {}
    WS = a.workspace or setting("POSTGRES_WORKSPACE") or setting("WORKSPACE") or "default"
    GRAPH = graph_name(WS)
    model = a.embedding_model or setting("EMBEDDING_MODEL")
    dim = a.embedding_dim or setting("EMBEDDING_DIM")
    T_ENT = T_REL = None
    if model and dim:
        T_ENT, T_REL = (f"{base}_{vdb_suffix(model, dim)}" for base in VDB_BASES)
    PLAN = a.plan
    API_URL = (a.api_url or setting("LIGHTRAG_BASE_URL")
               or f"http://localhost:{setting('PORT', '9621')}").rstrip("/")


async def resolve_vdb_tables(con):
    """Pick the entity/relation vector tables when EMBEDDING_MODEL/EMBEDDING_DIM are unset.

    LightRAG names them lightrag_vdb_entity[_<model>_<dim>d]; take the one that holds
    rows of this workspace (or the only one there is), otherwise ask for the model.
    The pattern ends in <dim>d so --commit's *_bak_dangling_<stamp> copies never match.
    """
    global T_ENT, T_REL
    names = [r["table_name"] for r in await con.fetch(
        "select table_name from information_schema.tables where table_schema = any(current_schemas(false))"
        " and table_name ~ '^lightrag_vdb_entity(_[a-z0-9_]+_[0-9]+d)?$' order by table_name")]
    used = [n for n in names
            if await con.fetchval(f"select exists (select 1 from {n} where workspace=$1)", WS)]
    pick = used or names
    if len(pick) != 1:
        sys.exit(f"cannot tell which vector tables hold workspace {WS!r} (candidates: {pick or 'none'}); "
                 "set EMBEDDING_MODEL and EMBEDDING_DIM as in LightRAG's .env, or pass "
                 "--embedding-model/--embedding-dim")
    T_ENT = pick[0]
    T_REL = VDB_BASES[1] + T_ENT[len(VDB_BASES[0]):]


async def connect():
    con = await asyncpg.connect(
        host=setting("POSTGRES_HOST", "localhost"), port=int(setting("POSTGRES_PORT", "5432")),
        user=setting("POSTGRES_USER", "postgres"), password=setting("POSTGRES_PASSWORD"),
        database=setting("POSTGRES_DATABASE", "postgres"))
    if T_ENT is None:
        await resolve_vdb_tables(con)
    return con


def target():
    return f"workspace {WS!r}, graph {GRAPH!r}, vector tables {T_ENT} / {T_REL}"


def pipeline_idle():
    url = f"{API_URL}/documents/pipeline_status"
    key = setting("LIGHTRAG_API_KEY")
    req = urllib.request.Request(url, headers={"X-API-Key": key} if key else {})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=15))
    except OSError as e:
        sys.exit(f"cannot read {url}: {e} (is the LightRAG server running? --force skips this check)")
    return d.get("busy") is False


def split_refs(chunks, live_docs, live_fp, valid_chunks):
    """Return (kept_chunks, docs_with_missing_chunks, new_file_path or None)."""
    kept, missing = [], set()
    for c in chunks:
        m = CHUNK_RE.match(c)
        if m and c not in valid_chunks:
            missing.add(m.group(1))
        else:
            kept.append(c)
    fp = None
    if missing:
        docs = []
        for c in kept:
            m = CHUNK_RE.match(c)
            if not m or m.group(1) not in live_fp:
                docs = None
                break
            p = live_fp[m.group(1)]
            if p not in docs:
                docs.append(p)
        fp = SEP.join(docs) if docs is not None else None
    return kept, missing, fp


async def scan():
    con = await connect()
    print(target())
    t0 = time.time()
    rows = await con.fetch("select id, file_path from lightrag_doc_status where workspace=$1", WS)
    live_docs = {r["id"] for r in rows}
    live_fp = {r["id"]: r["file_path"] for r in rows if r["file_path"]}
    valid_chunks = {r["id"] for r in await con.fetch(
        "select id from lightrag_doc_chunks where workspace=$1", WS)}
    plan = {"generated": time.strftime("%F %T"), "workspace": WS, "graph": GRAPH, "dead_docs": [],
            "vdb": {T_ENT: [], T_REL: []}, "maps": {T_ECH: [], T_RCH: []},
            "vertices": [], "edges": []}
    dead_all = set()

    for t in (T_ENT, T_REL):
        recs = await con.fetch(
            f"select id, chunk_ids, file_path from {t} v where v.workspace=$1 and exists ("
            f" select 1 from unnest(v.chunk_ids) c where {MISSING})", WS)
        for r in recs:
            kept, dead, fp = split_refs(list(r["chunk_ids"] or []), live_docs, live_fp, valid_chunks)
            dead_all |= dead
            plan["vdb"][t].append({"id": r["id"], "old": list(r["chunk_ids"]), "new": kept,
                                   "old_fp": r["file_path"],
                                   "new_fp": fp if fp is not None else r["file_path"],
                                   "delete": not kept})
    for t in (T_ECH, T_RCH):
        recs = await con.fetch(
            f"select id, chunk_ids::text as cj from {t} m where m.workspace=$1 and exists ("
            f" select 1 from jsonb_array_elements_text(m.chunk_ids) c where {MISSING})", WS)
        for r in recs:
            old = json.loads(r["cj"])
            kept, dead, _ = split_refs(old, live_docs, live_fp, valid_chunks)
            dead_all |= dead
            plan["maps"][t].append({"id": r["id"], "old": old, "new": kept, "delete": not kept})

    await con.execute('set search_path = ag_catalog, "$user", public')
    for kind, tbl in (("vertices", "_ag_label_vertex"), ("edges", "_ag_label_edge")):
        extra = ", start_id::text as s, end_id::text as e" if kind == "edges" else ""
        recs = await con.fetch(
            f'select id::text as gid, properties::text as p{extra} from "{GRAPH}".{tbl} g where exists ('
            f" select 1 from unnest(string_to_array((g.properties::text)::jsonb->>'source_id', '{SEP}')) c"
            f" where {MISSING})", WS)
        for r in recs:
            props = json.loads(r["p"])
            old = [c for c in (props.get("source_id") or "").split(SEP) if c]
            kept, dead, fp = split_refs(old, live_docs, live_fp, valid_chunks)
            dead_all |= dead
            newp = dict(props)
            newp["source_id"] = SEP.join(kept)
            if fp is not None:
                newp["file_path"] = fp
            item = {"gid": r["gid"], "old_props": r["p"], "new_props": json.dumps(newp, ensure_ascii=False),
                    "delete": not kept}
            if kind == "edges":
                item["start"], item["end"] = r["s"], r["e"]
            plan[kind].append(item)
    # a vertex deleted outright takes every edge touching it
    del_v = {v["gid"] for v in plan["vertices"] if v["delete"]}
    if del_v:
        recs = await con.fetch(
            f'select id::text as gid from "{GRAPH}"._ag_label_edge where start_id::text = any($1) or end_id::text = any($1)',
            list(del_v))
        known = {e["gid"] for e in plan["edges"]}
        for r in recs:
            if r["gid"] in known:
                for e in plan["edges"]:
                    if e["gid"] == r["gid"]:
                        e["delete"] = True
            else:
                plan["edges"].append({"gid": r["gid"], "old_props": None, "new_props": None,
                                      "delete": True, "cascade": True})
    await con.close()
    plan["dead_docs"] = sorted(dead_all)
    with open(PLAN, "w") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=1)
    summary(plan, time.time() - t0)


def summary(plan, secs=None):
    print(f"plan {PLAN} (generated {plan['generated']}{'' if secs is None else f', scan {secs:.0f}s'})")
    print(f"docs with missing chunk refs: {len(plan['dead_docs'])} -> {plan['dead_docs']}")
    for t, items in plan["vdb"].items():
        print(f"  {t:36s} update {sum(1 for i in items if not i['delete']):5d}  delete {sum(1 for i in items if i['delete']):4d}")
    for t, items in plan["maps"].items():
        print(f"  {t:36s} update {sum(1 for i in items if not i['delete']):5d}  delete {sum(1 for i in items if i['delete']):4d}")
    for k in ("vertices", "edges"):
        items = plan[k]
        print(f"  graph {k:30s} update {sum(1 for i in items if not i['delete']):5d}  delete {sum(1 for i in items if i['delete']):4d}")


def read_plan():
    if not os.path.exists(PLAN):
        print(f"no plan at {PLAN} — run --scan first")
        return None
    with open(PLAN) as fh:
        return json.load(fh)


async def commit(force):
    plan = read_plan()
    if plan is None:
        return 2
    # gids and ids are only meaningful in the workspace and graph that were scanned
    if (plan.get("workspace"), plan.get("graph")) != (WS, GRAPH):
        print(f"plan is for workspace {plan.get('workspace')!r} / graph {plan.get('graph')!r}, "
              f"not {WS!r} / {GRAPH!r} — not committing")
        return 2
    if not force and not pipeline_idle():
        print("pipeline busy — not committing (exit 3)"); return 3
    con = await connect()
    stats = {"updated": 0, "deleted": 0, "skipped": 0}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    async with con.transaction():
        # Undo record: every row the commit DELETES is copied whole into a backup table
        # first (updates are recoverable from the plan file's old values). These are
        # created before search_path puts ag_catalog first, so they land in the schema
        # LightRAG's own tables are in.
        for t, items in list(plan["vdb"].items()) + list(plan["maps"].items()):
            ids = [i["id"] for i in items if i["delete"]]
            if ids:
                await con.execute(f"create table {short_name(t)}_bak_dangling_{stamp} as select * from {t} where workspace=$1 and id = any($2::text[])", WS, ids)
        await con.execute('set local search_path = ag_catalog, "$user", public')
        for kind, tbl in (("vertices", "_ag_label_vertex"), ("edges", "_ag_label_edge")):
            gids = [i["gid"] for i in plan[kind] if i["delete"]]
            if gids:
                await con.execute(f'create table "{GRAPH}".bak_dangling_{tbl.strip("_")}_{stamp} as select * from "{GRAPH}".{tbl} where id::text = any($1::text[])', gids)
        print(f"backup tables suffixed _bak_dangling_{stamp} / bak_dangling_*_{stamp} written for deleted rows")
        for t, items in plan["vdb"].items():
            for i in items:
                if i["delete"]:
                    r = await con.execute(f"delete from {t} where workspace=$1 and id=$2 and chunk_ids=$3::varchar[]", WS, i["id"], i["old"])
                else:
                    r = await con.execute(f"update {t} set chunk_ids=$1::varchar[], file_path=$2, update_time=now() where workspace=$3 and id=$4 and chunk_ids=$5::varchar[]",
                                          i["new"], i["new_fp"], WS, i["id"], i["old"])
                tally(stats, r, i["delete"])
        for t, items in plan["maps"].items():
            for i in items:
                if i["delete"]:
                    r = await con.execute(f"delete from {t} where workspace=$1 and id=$2 and chunk_ids=$3::jsonb", WS, i["id"], json.dumps(i["old"]))
                else:
                    r = await con.execute(f"update {t} set chunk_ids=$1::jsonb, count=$2, update_time=now() where workspace=$3 and id=$4 and chunk_ids=$5::jsonb",
                                          json.dumps(i["new"]), len(i["new"]), WS, i["id"], json.dumps(i["old"]))
                tally(stats, r, i["delete"])
        # edges first (a deleted vertex must not leave edges behind), then vertices
        for e in plan["edges"]:
            if e["delete"]:
                if e.get("cascade"):
                    r = await con.execute(f'delete from "{GRAPH}"._ag_label_edge where id::text=$1', e["gid"])
                else:
                    r = await con.execute(f'delete from "{GRAPH}"._ag_label_edge where id::text=$1 and properties::text=$2', e["gid"], e["old_props"])
            else:
                r = await con.execute(f'update "{GRAPH}"._ag_label_edge set properties=$1::text::agtype where id::text=$2 and properties::text=$3',
                                      e["new_props"], e["gid"], e["old_props"])
            tally(stats, r, e["delete"])
        for v in plan["vertices"]:
            if v["delete"]:
                r = await con.execute(f'delete from "{GRAPH}"._ag_label_vertex where id::text=$1 and properties::text=$2', v["gid"], v["old_props"])
            else:
                r = await con.execute(f'update "{GRAPH}"._ag_label_vertex set properties=$1::text::agtype where id::text=$2 and properties::text=$3',
                                      v["new_props"], v["gid"], v["old_props"])
            tally(stats, r, v["delete"])
    await con.close()
    print(f"committed: {stats}")
    os.rename(PLAN, PLAN + ".done")
    return 0


def tally(stats, status, is_delete):
    n = int(status.split()[-1])
    if n == 0:
        stats["skipped"] += 1
    elif is_delete:
        stats["deleted"] += n
    else:
        stats["updated"] += n


async def update_graph_rows(con, table, rows):
    """Update graph properties in batches, skipping rows changed since the read.

    id::text defeats AGE's graphid index, so one UPDATE per row scans the entire
    graph thousands of times. A single join scans it once per batch instead.
    """
    sql = (f'update "{GRAPH}".{table} g set properties=u.new_props::text::ag_catalog.agtype '
           "from jsonb_to_recordset($1::jsonb) as u(gid text, old_props text, new_props text) "
           "where g.id::text=u.gid and g.properties::text = u.old_props")
    changed = 0
    for start in range(0, len(rows), 1000):
        result = await con.execute(sql, json.dumps(rows[start:start + 1000], ensure_ascii=False))
        changed += int(result.rsplit(" ", 1)[-1])
    return changed


async def sweep(dry, force=False):
    """Set-based removal of dead chunk refs — the converging counterpart to --commit.

    --commit compares the WHOLE chunk_ids array (compare-and-set), so a row that
    extraction rewrites between --scan and --commit is skipped. On a corpus's hottest
    entities that can be every pass, so they never converge and keep the
    WEIGHT-fallback warning alive. Here each statement filters the row's CURRENT
    array against doc_chunks, preserving newly added chunks. Refuses writes while
    the pipeline is busy (override: force); no plan file.
    """
    if not dry and not force and not pipeline_idle():
        print("pipeline busy — not sweeping (exit 3)", flush=True)
        return 3
    con = await connect()
    print(target(), flush=True)
    tot = {}
    async with con.transaction():
        await con.execute('set local search_path = ag_catalog, "$user", public')
        for t in (T_ENT, T_REL):
            where = f"v.workspace=$1 and exists (select 1 from unnest(v.chunk_ids) c where {MISSING})"
            n = await con.fetchval(f"select count(*) from {t} v where {where}", WS)
            tot[t] = n
            if n and not dry:
                await con.execute(f"""update {t} v set
                      chunk_ids = array(select c from unnest(v.chunk_ids) with ordinality u(c, o)
                                        where not ({MISSING}) order by o),
                      file_path = coalesce((select string_agg(distinct s.file_path, '{SEP}')
                                             from unnest(v.chunk_ids) c
                                             join lightrag_doc_chunks d on d.workspace=$1 and d.id=c
                                             join lightrag_doc_status s on s.workspace=$1
                                              and s.id = d.full_doc_id),
                                            v.file_path),
                      update_time = now()
                    where {where}""", WS)
        for t in (T_ECH, T_RCH):
            where = (f"m.workspace=$1 and exists (select 1 from"
                     f" jsonb_array_elements_text(m.chunk_ids) c where {MISSING})")
            n = await con.fetchval(f"select count(*) from {t} m where {where}", WS)
            tot[t] = n
            if n and not dry:
                await con.execute(f"""update {t} m set
                      chunk_ids = coalesce((select jsonb_agg(c order by o)
                                             from jsonb_array_elements_text(m.chunk_ids)
                                                  with ordinality u(c, o)
                                             where not ({MISSING})), '[]'::jsonb),
                      count = (select count(*) from jsonb_array_elements_text(m.chunk_ids) c
                                where not ({MISSING})),
                      update_time = now()
                    where {where}""", WS)
        live_chunks = {r["id"] for r in await con.fetch(
            "select id from lightrag_doc_chunks where workspace=$1", WS)}
        rows = await con.fetch("select id, file_path from lightrag_doc_status where workspace=$1", WS)
        live = {r["id"] for r in rows}
        fps = {r["id"]: r["file_path"] for r in rows if r["file_path"]}
        for kind, tbl in (("vertices", "_ag_label_vertex"), ("edges", "_ag_label_edge")):
            recs = await con.fetch(
                f'select id::text as gid, properties::text as p from "{GRAPH}".{tbl} g where exists ('
                f" select 1 from unnest(string_to_array((g.properties::text)::jsonb->>'source_id',"
                f" '{SEP}')) c where {MISSING})", WS)
            tot[kind] = len(recs)
            if recs and not dry:
                updates = []
                for r in recs:
                    props = json.loads(r["p"])
                    kept, _, fp = split_refs([c for c in (props.get("source_id") or "").split(SEP) if c],
                                             live, fps, live_chunks)
                    props["source_id"] = SEP.join(kept)
                    if fp is not None:
                        props["file_path"] = fp
                    updates.append({"gid": r["gid"], "old_props": r["p"],
                                    "new_props": json.dumps(props, ensure_ascii=False)})
                changed = await update_graph_rows(con, tbl, updates)
                if changed < len(updates):
                    print(f"{kind}: {len(updates) - changed} row(s) changed since the read; left for the next sweep",
                          flush=True)
        empt = {}
        for t in (T_ENT, T_REL):
            empt[t] = await con.fetchval(
                f"select count(*) from {t} where workspace=$1 and cardinality(chunk_ids)=0", WS)
        for t in (T_ECH, T_RCH):
            empt[t] = await con.fetchval(
                f"select count(*) from {t} where workspace=$1 and jsonb_array_length(chunk_ids)=0", WS)
    await con.close()
    print(("DRY-RUN " if dry else "") + "rows with missing chunk refs: "
          + ", ".join(f"{short_name(k)}={v}" for k, v in tot.items()))
    left = {short_name(k): v for k, v in empt.items() if v}
    print("rows left with NO chunks (delete separately if any): " + (str(left) if left else "none"))
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Remove references to deleted chunks from a LightRAG PostgreSQL workspace.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", action="store_true", help="read-only scan; writes the plan file")
    g.add_argument("--commit", action="store_true", help="apply the plan (compare-and-set, backups)")
    g.add_argument("--show", action="store_true", help="print the current plan summary")
    g.add_argument("--sweep", action="store_true", help="set-based in-place removal (converges on hot rows; no plan)")
    ap.add_argument("--dry-run", action="store_true", help="with --sweep: count only")
    ap.add_argument("--force", action="store_true",
                    help="--commit/--sweep: skip the pipeline idle check (only with no LightRAG server writing)")
    ap.add_argument("--env-file", help="dotenv file to read settings from, e.g. LightRAG's .env; "
                                       "the process environment takes precedence")
    ap.add_argument("--workspace", help="default: POSTGRES_WORKSPACE, else WORKSPACE, else 'default'")
    ap.add_argument("--embedding-model", help="default: EMBEDDING_MODEL (names the vector tables)")
    ap.add_argument("--embedding-dim", type=int, help="default: EMBEDDING_DIM (names the vector tables)")
    ap.add_argument("--api-url", help="LightRAG server for the idle check; default: LIGHTRAG_BASE_URL, "
                                      "else http://localhost:$PORT (PORT default 9621)")
    ap.add_argument("--plan", default=PLAN, help="plan file for --scan/--commit/--show (default: %(default)s)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    configure(a)
    if a.scan:
        asyncio.run(scan())
        return 0
    if a.sweep:
        return asyncio.run(sweep(a.dry_run, a.force))
    if a.show:
        plan = read_plan()
        if plan is None:
            return 2
        summary(plan)
        return 0
    return asyncio.run(commit(a.force))


if __name__ == "__main__":
    sys.exit(main())
