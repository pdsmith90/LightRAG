"""Fork (ENTITY_NAME_FOLD): one graph node for names that differ only in case,
spacing or punctuation ("Kalman Filter", "Kalman filter", "KalmanFilter",
"Kalman-Filter").

The extraction LLM spells the same entity several ways across papers and the
merge keys on the exact string, so a concept's relations scatter over several
nodes and none of them collects its whole neighbourhood. With the switch on,
every name an extraction produces is mapped to the spelling the graph already
uses for its fold key -- the most-connected one -- or becomes that spelling when
the key is new. It is applied where extraction results leave
``extract_entities`` (so the write-ahead anchors and the merge see the same
names) and, as an alias, where the knowledge rebuild re-parses cached results.

Kept apart:
- names whose key has fewer than four characters ("CO"/"Co", "SAR"/"Sar");
- up to five characters, an all-capitals acronym and a word ("TIDE"/"tide",
  "MARS"/"Mars");
- different digits or scripts (the key keeps every letter and digit).
Names the apparatus guard (``entity_name_guard``) would drop never become the
canonical spelling.

The map is built from the graph once the storages are up (Postgres graph
storage only) and is armed when complete; until then names pass unchanged.
"""

from __future__ import annotations

import time
from collections import defaultdict

from lightrag.entity_name_guard import fold, junk_entity_class
from lightrag.utils import logger

MIN_KEY = 4
SHORT_KEY = 5

_ARMED = False
_MAP: dict[str, str] = {}


def _case_class(name: str) -> str:
    cased = [c for c in name if c.isupper() or c.islower()]
    return "U" if cased and all(c.isupper() for c in cased) else "m"


def fold_key(name: str) -> str | None:
    """The map key for ``name``, or None when the name is never folded."""
    f = fold(name)
    if len(f) < MIN_KEY:
        return None
    return f"{f}|{_case_class(name)}" if len(f) <= SHORT_KEY else f


def canonical_name(name: str, register: bool = True) -> str:
    """The spelling the graph uses for ``name``'s key; with ``register``, a name
    whose key is new becomes that spelling."""
    if not _ARMED:
        return name
    key = fold_key(name)
    if key is None:
        return name
    canon = _MAP.get(key)
    if canon is None:
        if register:
            _MAP[key] = name
        return name
    return canon


def load_fold_map(rows) -> int:
    """Fill the map from ``(name, stored type, degree)`` rows; arm it. Returns the
    number of keys. The best-connected spelling of each key wins; apparatus names
    are skipped."""
    global _ARMED
    new: dict[str, str] = {}
    for name, typ, _degree in sorted(rows, key=lambda r: -int(r[2] or 0)):
        if not name:
            continue
        key = fold_key(name)
        if key is None or key in new:
            continue
        if junk_entity_class(name, (typ or "").replace(" ", "").lower()):
            continue
        new[key] = name
    _MAP.clear()
    _MAP.update(new)
    _ARMED = True
    return len(_MAP)


def disarm() -> None:
    global _ARMED
    _ARMED = False
    _MAP.clear()


def is_armed() -> bool:
    return _ARMED


async def build_fold_map(graph) -> int | None:
    """Read every node's name, type and degree from a Postgres/AGE graph storage
    and arm the map. Returns the key count, or None when the storage is not
    supported (folding then stays off)."""
    graph_name = getattr(graph, "graph_name", None)
    if graph_name is None or not hasattr(graph, "_query"):
        logger.warning(
            "ENTITY_NAME_FOLD ignored: it needs the Postgres (AGE) graph storage"
        )
        return None
    started = time.monotonic()
    sql = f"""SELECT (b.properties::text)::jsonb->>'entity_id' AS name,
                     coalesce((b.properties::text)::jsonb->>'entity_type', '') AS type,
                     coalesce(d.n, 0) AS degree
              FROM "{graph_name}".base b
              LEFT JOIN (SELECT id, count(*) AS n FROM (
                           SELECT start_id AS id FROM "{graph_name}"."DIRECTED"
                           UNION ALL SELECT end_id FROM "{graph_name}"."DIRECTED") x
                         GROUP BY id) d ON d.id = b.id"""
    rows = await graph._query(sql)
    keys = load_fold_map(
        (r.get("name"), r.get("type"), r.get("degree")) for r in rows or []
    )
    logger.info(
        f"ENTITY_NAME_FOLD armed: {keys} name keys from {len(rows or [])} nodes "
        f"in {time.monotonic() - started:.1f}s"
    )
    return keys


def fold_chunk_result(maybe_nodes: dict, maybe_edges: dict) -> tuple[dict, dict]:
    """Rename one chunk's entities and relation endpoints to canonical spellings.

    Records of names that fold together are merged under one name; a relation
    whose endpoints fold together is dropped (it would be a self-loop).
    """
    if not _ARMED:
        return maybe_nodes, maybe_edges
    nodes: dict = defaultdict(list)
    for name, records in maybe_nodes.items():
        canon = canonical_name(name)
        for record in records:
            record["entity_name"] = canon
        nodes[canon].extend(records)
    edges: dict = defaultdict(list)
    for (src, tgt), records in maybe_edges.items():
        csrc, ctgt = canonical_name(src), canonical_name(tgt)
        if csrc == ctgt:
            continue
        for record in records:
            record["src_id"], record["tgt_id"] = csrc, ctgt
        edges[(csrc, ctgt)].extend(records)
    return dict(nodes), dict(edges)


def alias_chunk_result(maybe_nodes: dict, maybe_edges: dict) -> tuple[dict, dict]:
    """For the knowledge rebuild: keep every record under its original name and
    add a copy under the canonical name.

    Nodes merged before the fold keep their old spelling until their documents
    are rebuilt, while new merges write the canonical one; a rebuild looks
    records up by the node's exact name, so both must find theirs. Nothing is
    registered here.
    """
    if not _ARMED:
        return maybe_nodes, maybe_edges
    nodes: dict = defaultdict(list)
    for name, records in maybe_nodes.items():
        nodes[name].extend(records)
        canon = canonical_name(name, register=False)
        if canon != name:
            nodes[canon].extend({**r, "entity_name": canon} for r in records)
    edges: dict = defaultdict(list)
    for (src, tgt), records in maybe_edges.items():
        edges[(src, tgt)].extend(records)
        csrc = canonical_name(src, register=False)
        ctgt = canonical_name(tgt, register=False)
        if (csrc, ctgt) != (src, tgt) and csrc != ctgt:
            edges[(csrc, ctgt)].extend(
                {**r, "src_id": csrc, "tgt_id": ctgt} for r in records
            )
    return dict(nodes), dict(edges)
