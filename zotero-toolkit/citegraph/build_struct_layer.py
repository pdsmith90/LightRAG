#!/usr/bin/env python3
"""
build_struct_layer.py - build a typed "structural layer" for a knowledge graph
from a Zotero library's bibliographic metadata plus OpenAlex.

Reads zotero_metadata.json (a dict keyed by Zotero attachment key), asks
OpenAlex for every DOI in it, and writes CSV tables ready to load into
PostgreSQL (schema "zstruct"): papers, authors, venues, a four-level topic
hierarchy, keywords, citation edges between library papers and
bibliographic-coupling edges, plus schema.sql, load.sql, unmatched_dois.txt and
a QA report (qa.json, QA.md). Nothing is loaded into a database.

Version 2 adds Semantic Scholar (S2): paper metadata and TLDRs, SPECTER2
embeddings, a title gap fill for papers without a usable DOI, and citation
contexts from the local store that s2_citations_sync.py keeps; OpenAlex works
outside the library that at least two library papers cite; and the SQL files
apply_<schema>.sql, swap_<schema>.sql and drop_prev_<schema>.sql (generated,
never run). The builder never connects to a database.

Usage:
  python3 build_struct_layer.py --metadata zotero_metadata.json --out out \
      [--cache cache/openalex_works.json] [--refresh] [--offline] \
      [--reference-check zotero_references.json] [--mailto ADDR] \
      [--schema zstruct] [--workspace zotero] [--s2-legs papers,embeddings,match] \
      [--s2-max-calls N] [--s2-store s2store/citations.sqlite]

The OpenAlex key, if any, is read from OPENALEX_API_KEY and sent only as the
request header "api_key" (never in a URL, which reaches logs). OPENALEX_MAILTO
(or --mailto) joins OpenAlex's polite pool. The S2 key, if any, is read from
S2_API_KEY and sent only as the header "x-api-key", only to
api.semanticscholar.org. A hard check scans every file under out/ and the cache
directory for the key values.

The cache (--cache) holds the raw OpenAlex work objects keyed by normalized DOI
and the list of DOIs OpenAlex did not know. A rerun asks only for DOIs not in
the cache (unknown DOIs are asked again); --refresh asks for all; --offline
makes no request.

Exit status: 0 on success; 1 when a request failed after retries, a hard check
failed, or the reference check's Jaccard index is below 0.98. Outputs are
written from what is known in every case.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import http.client
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

import s2_api

API = "https://api.openalex.org/works"
BATCH = 50  # DOIs per filter request
PAUSE = 0.2  # seconds between requests
TRIES = 4
RETRY_CODES = (429, 500, 502, 503, 504)
SELECT = (
    "id,doi,publication_year,type,cited_by_count,referenced_works,primary_topic,"
    "topics,keywords,authorships,primary_location"
)

# OpenAlex works outside the library
MIN_EXTERNAL_CITERS = 2
EXT_BATCH = 50
RESOLVE_MAX_FAILS = (
    10  # single lookups of unknown outside works failing in a row before the rest wait
)
EXT_SELECT = "id,doi,title,display_name,publication_year,type,cited_by_count,authorships,primary_location"

# S2 context cleaning
MAX_CONTEXT = 2000

# bibliographic coupling
MIN_DF = 2
MAX_DF = 200
MIN_SHARED = 3
TOP_K = 10

ORACLE_MIN_JACCARD = 0.98
LEVELS = ("topic", "subfield", "field", "domain")

# table name -> definition, in dependency (load) order
TABLES = {
    "venues": {
        "cols": [
            ("venue_id", "text"),
            ("display_name", "text"),
            ("openalex_source_id", "text"),
            ("n_papers", "integer"),
        ],
        "pk": ["venue_id"],
        "indexes": [],
    },
    "topics": {
        "cols": [
            ("topic_id", "text"),
            ("display_name", "text"),
            ("level", "text"),
            ("parent_id", "text"),
        ],
        "pk": ["topic_id"],
        "fks": [("parent_id", "topics", "topic_id")],
        "checks": ["level IN ('topic', 'subfield', 'field', 'domain')"],
        "indexes": [["parent_id"], ["level"]],
    },
    "papers": {
        "cols": [
            ("paper_id", "text"),
            ("doi", "text"),
            ("openalex_id", "text"),
            ("title", "text"),
            ("year", "integer"),
            ("venue_id", "text"),
            ("type", "text"),
            ("cited_by_count", "integer"),
            ("n_refs_openalex", "integer"),
            ("n_attachments", "integer"),
            ("s2_paper_id", "text"),
            ("s2_corpus_id", "bigint"),
            ("s2_citation_count", "integer"),
            ("s2_influential_citation_count", "integer"),
            ("s2_reference_count", "integer"),
            ("tldr", "text"),
            ("doi_source", "text"),
        ],
        "pk": ["paper_id"],
        "fks": [("venue_id", "venues", "venue_id")],
        "checks": ["doi_source IN ('zotero', 's2_title')"],
        "indexes": [["doi"], ["openalex_id"], ["venue_id"], ["year"], ["s2_corpus_id"]],
    },
    "paper_attachments": {
        "cols": [
            ("attachment_key", "text"),
            ("item_key", "text"),
            ("paper_id", "text"),
        ],
        "pk": ["attachment_key"],
        "fks": [("paper_id", "papers", "paper_id")],
        "indexes": [["paper_id"], ["item_key"]],
    },
    "authors": {
        "cols": [
            ("author_id", "text"),
            ("display_name", "text"),
            ("openalex_id", "text"),
            ("n_papers", "integer"),
        ],
        "pk": ["author_id"],
        "indexes": [["openalex_id"]],
    },
    "paper_authors": {
        "cols": [
            ("paper_id", "text"),
            ("author_id", "text"),
            ("position", "integer"),
            ("source", "text"),
        ],
        "pk": ["paper_id", "position"],
        "unique": [["paper_id", "author_id"]],
        "fks": [
            ("paper_id", "papers", "paper_id"),
            ("author_id", "authors", "author_id"),
        ],
        "checks": ["source IN ('openalex', 'zotero')"],
        "indexes": [["author_id"]],
    },
    "paper_topics": {
        "cols": [
            ("paper_id", "text"),
            ("topic_id", "text"),
            ("score", "double precision"),
            ("is_primary", "boolean"),
        ],
        "pk": ["paper_id", "topic_id"],
        "fks": [
            ("paper_id", "papers", "paper_id"),
            ("topic_id", "topics", "topic_id"),
        ],
        "indexes": [["topic_id"]],
    },
    "keywords": {
        "cols": [
            ("keyword_id", "text"),
            ("display_name", "text"),
            ("n_papers", "integer"),
        ],
        "pk": ["keyword_id"],
        "indexes": [],
    },
    "paper_keywords": {
        "cols": [
            ("paper_id", "text"),
            ("keyword_id", "text"),
            ("score", "double precision"),
        ],
        "pk": ["paper_id", "keyword_id"],
        "fks": [
            ("paper_id", "papers", "paper_id"),
            ("keyword_id", "keywords", "keyword_id"),
        ],
        "indexes": [["keyword_id"]],
    },
    "cites": {
        "cols": [
            ("citing_paper_id", "text"),
            ("cited_paper_id", "text"),
            ("source", "text"),
            ("is_influential", "boolean"),
            ("intents", "text[]"),
            ("n_contexts", "integer"),
        ],
        "pk": ["citing_paper_id", "cited_paper_id"],
        "notnull": ["source"],
        "fks": [
            ("citing_paper_id", "papers", "paper_id"),
            ("cited_paper_id", "papers", "paper_id"),
        ],
        "checks": [
            "citing_paper_id <> cited_paper_id",
            "source IN ('openalex', 's2', 'both')",
        ],
        "indexes": [["cited_paper_id"]],
    },
    "coupled": {
        "cols": [
            ("paper_a", "text"),
            ("paper_b", "text"),
            ("shared_refs", "integer"),
            ("score", "double precision"),
        ],
        "pk": ["paper_a", "paper_b"],
        "fks": [
            ("paper_a", "papers", "paper_id"),
            ("paper_b", "papers", "paper_id"),
        ],
        # Byte order, as Python sorts: under the database's own collation (en_US) punctuation is
        # ignored and some pairs compare the other way round.
        "checks": ['paper_a COLLATE "C" < paper_b COLLATE "C"'],
        "indexes": [["paper_b"]],
    },
    "cite_contexts": {
        "cols": [
            ("citing_paper_id", "text"),
            ("cited_paper_id", "text"),
            ("ordinal", "integer"),
            ("context", "text"),
            ("intents", "text[]"),
        ],
        "pk": ["citing_paper_id", "cited_paper_id", "ordinal"],
        "fks": [
            (
                ["citing_paper_id", "cited_paper_id"],
                "cites",
                ["citing_paper_id", "cited_paper_id"],
            ),
        ],
        "indexes": [["cited_paper_id"]],
    },
    "external_works": {
        "cols": [
            ("work_id", "text"),
            ("openalex_id", "text"),
            ("doi", "text"),
            ("title", "text"),
            ("year", "integer"),
            ("type", "text"),
            ("cited_by_count", "integer"),
            ("first_author", "text"),
            ("venue", "text"),
            ("n_library_citers", "integer"),
        ],
        "pk": ["work_id"],
        "indexes": [["doi"]],
    },
    "cites_external": {
        "cols": [("citing_paper_id", "text"), ("work_id", "text")],
        "pk": ["citing_paper_id", "work_id"],
        "fks": [
            ("citing_paper_id", "papers", "paper_id"),
            ("work_id", "external_works", "work_id"),
        ],
        "indexes": [["work_id"]],
    },
    "paper_embeddings": {
        "cols": [("paper_id", "text"), ("model", "text"), ("embedding", "vector(768)")],
        "pk": ["paper_id"],
        "fks": [("paper_id", "papers", "paper_id")],
        "indexes": [],
    },
}


# --------------------------------------------------------------------------
# identifiers and names
# --------------------------------------------------------------------------


def norm_doi(doi: str) -> str:
    """Lowercase, without a resolver prefix or 'doi:'; '' when not a DOI."""
    d = (doi or "").strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d if d.startswith("10.") and "/" in d else ""


def short_id(openalex_id) -> str:
    """Last path segment of an OpenAlex URL ('' for null)."""
    return (openalex_id or "").rsplit("/", 1)[-1]


def fold(s) -> str:
    """Casefold, strip accents, keep letters only."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return "".join(c for c in s if c.isalpha())


def initials(given: str) -> str:
    """First (folded) letter of each given-name token."""
    out = ""
    for tok in re.split(r"[\s\-]+", given or ""):
        f = fold(tok)
        if f:
            out += f[0]
    return out


def _nonempty(v) -> bool:
    return v is not None and v != "" and v != [] and v != {}


def parse_year(s):
    m = re.search(r"\d{4}", str(s or ""))
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------
# OpenAlex harvest
# --------------------------------------------------------------------------


def _urlopen_json(url: str, headers: dict) -> dict:
    """The only function that touches the network."""
    with urllib.request.urlopen(
        s2_api.build_request("GET", url, headers), timeout=60
    ) as resp:
        return json.load(resp)


# Set when OpenAlex answers 401/403 to the key: the rest of the run goes without it.
_STATE = {"oa_key_rejected": False}


def get_json(url: str, api_key: str) -> dict:
    """GET with the key as the ``api_key`` header; retries 429 and 5xx, dropped
    connections, cut-short bodies and malformed JSON. A 401 or 403 while the key is sent
    is reported once, and the run continues without it."""
    for attempt in range(TRIES):
        headers = {"Accept": "application/json"}
        if api_key and not _STATE["oa_key_rejected"]:
            headers["api_key"] = api_key
        try:
            return s2_api.clean_obj(_urlopen_json(url, headers))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and "api_key" in headers:
                _STATE["oa_key_rejected"] = True
                print(
                    f"WARNING: OpenAlex answered {e.code} to the API key; continuing WITHOUT the key "
                    "for the rest of this run",
                    file=sys.stderr,
                )
                return get_json(url, "")
            if e.code not in RETRY_CODES or attempt == TRIES - 1:
                raise
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            ValueError,
        ):
            if attempt == TRIES - 1:
                raise
        time.sleep(2**attempt * 2)
    raise RuntimeError("unreachable")


def fetch_batch(dois: list[str], mailto: str, api_key: str) -> dict[str, dict]:
    """OpenAlex works for the DOIs it knows, keyed by normalized DOI."""
    params = {"select": SELECT, "per-page": "200"}
    if mailto:
        params["mailto"] = mailto
    params["filter"] = "doi:" + "|".join(dois)
    data = get_json(f"{API}?{urllib.parse.urlencode(params)}", api_key)
    wanted = set(dois)
    found = {}
    for work in data.get("results") or []:
        doi = norm_doi(work.get("doi") or "")
        if doi in wanted:
            found[doi] = work
    return found


def fetch_one(doi: str, mailto: str, api_key: str):
    """One work by DOI, or None when OpenAlex answers 404."""
    params = {"select": SELECT}
    if mailto:
        params["mailto"] = mailto
    url = (
        f"{API}/doi:{urllib.parse.quote(doi, safe='')}?{urllib.parse.urlencode(params)}"
    )
    try:
        return get_json(url, api_key)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def harvest(dois, store, mailto, api_key, refresh=False):
    """Fill ``store`` ({"works": {doi: work}, "unknown": [doi]}) for the DOIs
    it lacks. Returns {"batches", "singles", "failed"}."""
    works = store["works"]
    unknown = set(store.get("unknown") or [])
    todo = [d for d in dois if refresh or d not in works]
    # '|' and ',' separate filter values, so such a DOI is asked about alone
    plain = [d for d in todo if "|" not in d and "," not in d]
    odd = [d for d in todo if "|" in d or "," in d]
    stats = {"batches": 0, "singles": 0, "failed": 0}
    for i in range(0, len(plain), BATCH):
        batch = plain[i : i + BATCH]
        stats["batches"] += 1
        try:
            found = fetch_batch(batch, mailto, api_key)
            works.update(found)
            for d in batch:
                if d in found:
                    unknown.discard(d)
                elif d not in works:
                    unknown.add(d)
        except Exception as e:
            stats["failed"] += 1
            print(f"WARN: OpenAlex batch {i // BATCH + 1} failed: {e}", file=sys.stderr)
        time.sleep(PAUSE)
    for doi in odd:
        stats["singles"] += 1
        try:
            work = fetch_one(doi, mailto, api_key)
            if work:
                works[doi] = work
                unknown.discard(doi)
            elif doi not in works:
                unknown.add(doi)
        except Exception as e:
            stats["failed"] += 1
            print(f"WARN: OpenAlex lookup of {doi} failed: {e}", file=sys.stderr)
        time.sleep(PAUSE)
    store["unknown"] = sorted(unknown - set(works))
    return stats


def load_cache(path: str) -> dict:
    store = {"works": {}, "unknown": []}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        store["works"] = data.get("works") or {}
        store["unknown"] = data.get("unknown") or []
        for k in (
            "merged",
            "deleted",
        ):  # the outside-works store only (resolve_external_unknown)
            if k in data:
                store[k] = data[k]
    return store


def write_json(path: str, data, indent=None) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)  # no half-written .tmp is left behind
        raise  # readers see the old or the new file, never a partial one


def library_dois(metadata: dict) -> list[str]:
    """Sorted distinct normalized DOIs of the metadata's entries."""
    out = set()
    for entry in metadata.values():
        if isinstance(entry, dict):
            d = norm_doi(entry.get("doi") or "")
            if d:
                out.add(d)
    return sorted(out)


def fetch_external_batch(ids: list[str], mailto: str, api_key: str) -> dict[str, dict]:
    """OpenAlex works for the ids (``W...``) it knows, keyed by the short id."""
    params = {"select": EXT_SELECT, "per-page": "200"}
    if mailto:
        params["mailto"] = mailto
    params["filter"] = "openalex_id:" + "|".join(ids)
    data = get_json(f"{API}?{urllib.parse.urlencode(params)}", api_key)
    wanted = set(ids)
    found = {}
    for work in data.get("results") or []:
        wid = short_id(work.get("id"))
        if wid in wanted:
            found[wid] = _slim_authorships(work)
    return found


def harvest_external(ids, store, mailto, api_key, refresh=False):
    """Fill ``store`` ({"works": {W-id: work}, "unknown": [W-id]}) for the ids it
    lacks. An id OpenAlex did not return is recorded as unknown and not asked
    again (``refresh`` asks for all). Returns {"batches", "failed"}."""
    works = store["works"]
    unknown = set(store.get("unknown") or [])
    todo = [w for w in ids if refresh or (w not in works and w not in unknown)]
    stats = {"batches": 0, "failed": 0}
    for i in range(0, len(todo), EXT_BATCH):
        batch = todo[i : i + EXT_BATCH]
        stats["batches"] += 1
        try:
            found = fetch_external_batch(batch, mailto, api_key)
            works.update(found)
            for w in batch:
                if w in found:
                    unknown.discard(w)
                elif w not in works:
                    unknown.add(w)
        except Exception as e:
            stats["failed"] += 1
            print(
                f"WARN: OpenAlex external batch {i // EXT_BATCH + 1} failed: {e}",
                file=sys.stderr,
            )
        time.sleep(PAUSE)
    store["unknown"] = sorted(unknown - set(works))
    return stats


def _slim_authorships(work: dict) -> dict:
    """The cache keeps each author's id and name only: the institution lists are most of the bytes."""
    work["authorships"] = [
        {
            "author": {
                k: ((a or {}).get("author") or {}).get(k)
                for k in ("id", "display_name")
            }
        }
        for a in work.get("authorships") or []
    ]
    return work


def fetch_external_one(wid: str, mailto: str, api_key: str):
    """One OpenAlex work by id, or None when OpenAlex answers 404 (a deleted work). A merged
    work comes back under the id it was merged into."""
    params = {"select": EXT_SELECT}
    if mailto:
        params["mailto"] = mailto
    try:
        return get_json(
            f"{API}/{urllib.parse.quote(wid, safe='')}?{urllib.parse.urlencode(params)}",
            api_key,
        )
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def resolve_external_unknown(store, mailto, api_key):
    """Ask OpenAlex about each id in ``store["unknown"]`` (ids the list filter did not
    return) one at a time: a 404 goes to ``store["deleted"]``, a work served under
    another id goes to ``store["merged"]`` ({old: new}) with its record stored under
    the new id, and a work served under its own id is stored. An id whose request
    failed stays unknown and is asked again next run; after RESOLVE_MAX_FAILS
    failures in a row (OpenAlex down) the rest are left for the next run too.
    Returns the counts."""
    works = store["works"]
    merged = store.setdefault("merged", {})
    deleted = set(store.get("deleted") or [])
    stats = {"asked": 0, "found": 0, "merged": 0, "deleted": 0, "failed": 0}
    still = []
    in_a_row = 0
    todo = [
        w for w in store.get("unknown") or [] if w not in merged and w not in deleted
    ]
    for i, wid in enumerate(todo):
        if in_a_row >= RESOLVE_MAX_FAILS:
            still.extend(todo[i:])
            print(
                f"WARN: {in_a_row} outside-work lookups failed in a row; {len(todo) - i} left for the next run",
                file=sys.stderr,
            )
            break
        stats["asked"] += 1
        try:
            w = fetch_external_one(wid, mailto, api_key)
            new = short_id(w.get("id")) if w else ""
            if w and not new:
                raise ValueError("answer without an id")
        except Exception as e:
            stats["failed"] += 1
            in_a_row += 1
            still.append(wid)
            print(
                f"WARN: OpenAlex lookup of outside work {wid} failed: {e}",
                file=sys.stderr,
            )
            time.sleep(PAUSE)
            continue
        in_a_row = 0
        if w is None:
            deleted.add(wid)
            stats["deleted"] += 1
        else:
            works[new] = _slim_authorships(w)
            if new == wid:
                stats["found"] += 1
            else:
                merged[wid] = new
                stats["merged"] += 1
        time.sleep(PAUSE)
    store["deleted"] = sorted(deleted)
    store["unknown"] = sorted(still)
    return stats


# --------------------------------------------------------------------------
# assembling the tables
# --------------------------------------------------------------------------


def _best_entry(entries):
    """Entry with most non-empty fields; tie: smallest attachment key."""
    return min(
        entries, key=lambda ke: (-sum(_nonempty(v) for v in ke[1].values()), ke[0])
    )


def _zotero_creators(entry):
    """[(family, given)] from Zotero ``creators``. A creator with only one
    (single-field) name is split into given + last-token family."""
    out = []
    for c in entry.get("creators") or []:
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            continue
        family, given = (c[0] or "").strip(), (c[1] or "").strip()
        if not family and given:
            parts = given.split()
            family, given = parts[-1], " ".join(parts[:-1])
        if fold(family):
            out.append((family, given))
    return out


def _oa_name_key(display_name: str):
    """(folded family, first initial) of an OpenAlex 'Given Family' name, or
    None when the name has no usable pair."""
    toks = [t for t in (display_name or "").split() if fold(t)]
    if len(toks) < 2:
        return None
    return fold(toks[-1]), fold(toks[0])[0]


def _topic_chain(t):
    """[(topic_id, name, level)] from the topic up to its domain, skipping
    levels the object lacks."""
    chain = []
    tid = short_id(t.get("id"))
    if tid:
        chain.append((f"topic:{tid}", t.get("display_name") or "", "topic"))
    for lvl in LEVELS[1:]:
        o = t.get(lvl) or {}
        sid = short_id(o.get("id"))
        if sid:
            chain.append((f"{lvl}:{sid}", o.get("display_name") or "", lvl))
    return chain


def group_papers(metadata: dict) -> dict:
    """{paper_id: {"doi", "entries": [(attachment key, entry)]}}; entries sharing
    a DOI (or an item key when they have no DOI) are one paper."""
    groups = {}
    for akey in sorted(metadata):
        e = metadata[akey]
        if not isinstance(e, dict):
            continue
        doi = norm_doi(e.get("doi") or "")
        item_key = str(e.get("item_key") or "").strip()
        pid = f"doi:{doi}" if doi else f"zot:{item_key or akey}"
        g = groups.setdefault(pid, {"doi": doi, "entries": []})
        g["entries"].append((akey, e))
    return groups


def assemble(
    metadata: dict,
    works: dict,
    doi_overrides: dict | None = None,
    merged=None,
    deleted=(),
):
    """Build every table from the metadata and the OpenAlex works.

    ``doi_overrides`` ({paper_id: doi}) gives a DOI to papers without one (found
    by the S2 title match); the paper_id stays as it is. ``merged`` ({old W-id:
    W-id}) and ``deleted`` (W-ids) rewrite the reference lists before citations,
    coupling and outside works are taken from them (resolve_external_unknown).

    Returns (tables, aux): ``tables`` maps a table name to its rows (lists in
    column order, sorted by primary key); ``aux`` holds what the QA needs
    that the tables do not (author-merge counts, reference lists)."""
    # ---- papers and attachments
    groups = group_papers(metadata)
    for pid, doi in (doi_overrides or {}).items():  # DOIs the S2 title match found
        if pid in groups and not groups[pid]["doi"]:
            groups[pid]["doi"] = doi

    attach_rows = []
    paper_info = {}
    for pid in sorted(groups):
        g = groups[pid]
        akey, best = _best_entry(g["entries"])
        rec = works.get(g["doi"]) if g["doi"] else None
        paper_info[pid] = {
            "doi": g["doi"],
            "best": best,
            "rec": rec,
            "n_attach": len(g["entries"]),
        }
        for ak, e in g["entries"]:
            attach_rows.append([ak, str(e.get("item_key") or "").strip(), pid])

    # ---- venues
    venues = {}  # venue_id -> [name, source_id, set(papers)]
    paper_venue = {}
    for pid, info in paper_info.items():
        rec = info["rec"]
        src = ((rec or {}).get("primary_location") or {}).get("source") or {}
        sid = short_id(src.get("id"))
        if sid:
            vid, name, oa_id = f"oa:{sid}", src.get("display_name") or "", sid
        else:
            pub = str(info["best"].get("publication") or "").strip()
            f = fold(pub)
            vid, name, oa_id = (f"z:{f}", pub, "") if f else ("", "", "")
        paper_venue[pid] = vid
        if vid:
            v = venues.setdefault(vid, [name, oa_id, set()])
            if not v[0] and name:
                v[0] = name
            v[2].add(pid)

    # ---- topics, keywords
    topic_nodes = {}  # id -> [name, level, set(parents)]
    paper_topics = []
    keywords = {}  # id -> [name, set(papers)]
    paper_keywords = []
    for pid, info in paper_info.items():
        rec = info["rec"]
        if not rec:
            continue
        primary = rec.get("primary_topic") or {}
        primary_id = f"topic:{short_id(primary.get('id'))}" if primary.get("id") else ""
        seen = set()
        for t in [t for t in (rec.get("topics") or []) if t] + (
            [primary] if primary else []
        ):
            chain = _topic_chain(t)
            for i, (nid, name, lvl) in enumerate(chain):
                node = topic_nodes.setdefault(nid, ["", lvl, set()])
                if not node[0] and name:
                    node[0] = name
                if i + 1 < len(chain):
                    node[2].add(chain[i + 1][0])
            if chain and chain[0][2] == "topic" and chain[0][0] not in seen:
                tid = chain[0][0]
                seen.add(tid)
                paper_topics.append([pid, tid, t.get("score"), tid == primary_id])
        seen_kw = set()
        for k in rec.get("keywords") or []:
            kid = short_id((k or {}).get("id"))
            if not kid or kid in seen_kw:
                continue
            seen_kw.add(kid)
            kw = keywords.setdefault(f"kw:{kid}", [k.get("display_name") or "", set()])
            if not kw[0] and k.get("display_name"):
                kw[0] = k["display_name"]
            kw[1].add(pid)
            paper_keywords.append([pid, f"kw:{kid}", k.get("score")])

    # ---- authors
    oa_names = {}  # oa author id -> display name
    provisional = {}  # pid -> [(author_id, name, source)]
    z_names = {}  # z id -> (name, family, initials)
    skipped_no_id = 0
    for pid, info in paper_info.items():
        rec = info["rec"]
        lst = []
        for a in (rec or {}).get("authorships") or []:
            au = (a or {}).get("author") or {}
            aid = short_id(au.get("id"))
            if not aid:
                skipped_no_id += 1
                continue
            oa_names.setdefault(aid, au.get("display_name") or "")
            lst.append((f"oa:{aid}", au.get("display_name") or "", "openalex"))
        if not lst:
            for family, given in _zotero_creators(info["best"]):
                fam, ini = fold(family), initials(given)
                zid = f"z:{fam}|{ini}"
                name = f"{given} {family}".strip()
                z_names.setdefault(zid, (name, fam, ini))
                lst.append((zid, name, "zotero"))
        provisional[pid] = lst

    by_key = defaultdict(set)
    for aid, name in oa_names.items():
        k = _oa_name_key(name)
        if k:
            by_key[k].add(aid)
    z_target = {}
    merge_stats = {"merged": 0, "kept": 0, "kept_ambiguous": 0, "kept_no_match": 0}
    for zid, (name, fam, ini) in sorted(z_names.items()):
        cands = by_key.get((fam, ini[:1]), set()) if ini else set()
        if len(cands) == 1:
            z_target[zid] = f"oa:{next(iter(cands))}"
            merge_stats["merged"] += 1
        else:
            merge_stats["kept"] += 1
            merge_stats["kept_ambiguous" if len(cands) > 1 else "kept_no_match"] += 1

    authors = {}  # id -> [name, oa_id, set(papers)]
    paper_author_rows = []
    for pid in sorted(provisional):
        pos = 0
        seen = set()
        for aid, name, source in provisional[pid]:
            aid = z_target.get(aid, aid)
            if aid in seen:
                continue
            seen.add(aid)
            pos += 1
            if aid.startswith("oa:"):
                nm = oa_names[aid[3:]]
                a = authors.setdefault(aid, [nm, aid[3:], set()])
            else:
                a = authors.setdefault(aid, [z_names[aid][0], "", set()])
            a[2].add(pid)
            paper_author_rows.append([pid, aid, pos, source])

    # ---- citations
    oa_to_paper = {}
    for pid in sorted(paper_info):
        rec = paper_info[pid]["rec"]
        if rec and rec.get("id"):
            oa_to_paper.setdefault(short_id(rec["id"]), pid)
    cites = set()
    refs_by_paper = {}
    merged, deleted = merged or {}, set(deleted)
    for pid, info in paper_info.items():
        rec = info["rec"]
        if not rec:
            continue
        refs = {
            merged.get(short_id(w), short_id(w))
            for w in rec.get("referenced_works") or []
            if w
        } - deleted
        refs_by_paper[pid] = sorted(refs)
        for r in refs:
            tgt = oa_to_paper.get(r)
            if tgt and tgt != pid:
                cites.add((pid, tgt))

    coupled_rows = coupling(refs_by_paper)

    # ---- rows
    tables = {}
    tables["venues"] = [
        [vid, v[0], v[1], len(v[2])] for vid, v in sorted(venues.items())
    ]
    tables["topics"] = [
        [tid, n[0], n[1], min(n[2]) if n[2] else None]
        for tid, n in sorted(topic_nodes.items())
    ]
    prow = []
    for pid in sorted(paper_info):
        info = paper_info[pid]
        rec = info["rec"]
        best = info["best"]
        year = parse_year(best.get("year"))
        if year is None and rec:
            year = rec.get("publication_year")
        prow.append(
            [
                pid,
                info["doi"] or None,
                short_id(rec.get("id")) if rec and rec.get("id") else None,
                best.get("title") or None,
                year,
                paper_venue[pid] or None,
                rec.get("type") if rec else None,
                rec.get("cited_by_count") if rec else None,
                len(rec.get("referenced_works") or []) if rec else None,
                info["n_attach"],
            ]
        )
    tables["papers"] = prow
    tables["paper_attachments"] = sorted(attach_rows)
    tables["authors"] = [
        [aid, a[0], a[1] or None, len(a[2])] for aid, a in sorted(authors.items())
    ]
    tables["paper_authors"] = sorted(paper_author_rows, key=lambda r: (r[0], r[2]))
    tables["paper_topics"] = sorted(paper_topics, key=lambda r: (r[0], r[1]))
    tables["keywords"] = [[kid, k[0], len(k[1])] for kid, k in sorted(keywords.items())]
    tables["paper_keywords"] = sorted(paper_keywords, key=lambda r: (r[0], r[1]))
    tables["cites"] = [list(e) for e in sorted(cites)]
    tables["coupled"] = coupled_rows
    for name in tables:
        pk_idx = [
            i
            for i, (c, _) in enumerate(TABLES[name]["cols"])
            if c in TABLES[name]["pk"]
        ]
        tables[name].sort(key=lambda r: tuple(r[i] for i in pk_idx))
    aux = {
        "author_merge": {
            **merge_stats,
            "skipped_authorships_without_id": skipped_no_id,
        },
        "n_with_reference_list": sum(1 for r in refs_by_paper.values() if r),
        "refs_by_paper": refs_by_paper,
        "oa_to_paper": oa_to_paper,
        "doi_overrides": {
            pid: g["doi"]
            for pid, g in groups.items()
            if pid.startswith("zot:") and g["doi"]
        },
    }
    return tables, aux


def coupling(refs_by_paper: dict[str, list[str]]):
    """Bibliographic coupling rows [paper_a, paper_b, shared_refs, score]."""
    papers = sorted(p for p, r in refs_by_paper.items() if r)
    sets = [sorted(set(refs_by_paper[p])) for p in papers]
    n = len(papers)
    df = Counter()
    for s in sets:
        df.update(s)
    idf = {r: math.log(n / d) for r, d in df.items()}
    total = [sum(idf[r] for r in s) for s in sets]
    post = defaultdict(list)
    for i, s in enumerate(sets):
        for r in s:
            if MIN_DF <= df[r] <= MAX_DF:
                post[r].append(i)
    band = [[r for r in s if r in post] for s in sets]
    pairs = {}
    for a in range(n):
        cnt, sm = {}, {}
        for r in band[a]:
            w = idf[r]
            for b in post[r]:
                if b != a:
                    cnt[b] = cnt.get(b, 0) + 1
                    sm[b] = sm.get(b, 0.0) + w
        cands = []
        for b, c in cnt.items():
            if c < MIN_SHARED:
                continue
            den = math.sqrt(total[a] * total[b])
            if den > 0:
                cands.append((-sm[b] / den, b, c))
        cands.sort()
        for neg, b, c in cands[:TOP_K]:
            pairs[(a, b) if a < b else (b, a)] = (c, -neg)
    return [
        [papers[a], papers[b], c, round(s, 6)]
        for (a, b), (c, s) in sorted(pairs.items())
    ]


# --------------------------------------------------------------------------
# citation graph v2: S2 records, cites with contexts, outside works, embeddings
# --------------------------------------------------------------------------


def clean_context(text) -> tuple[str, bool]:
    """A citation context without control characters (tab, newline and the like
    become a space; NUL and lone surrogates go), cut at MAX_CONTEXT characters.
    Returns (text, was_cut)."""
    if not isinstance(text, str):
        return "", False
    out = []
    for ch in text:
        if ch in "\t\n\r\v\f":
            out.append(" ")
        elif unicodedata.category(ch) in ("Cc", "Cs"):
            continue
        else:
            out.append(ch)
    t = "".join(out).strip()
    if len(t) > MAX_CONTEXT:
        return t[:MAX_CONTEXT], True
    return t, False


def title_candidates(groups: dict, works: dict, s2_by_id: dict) -> list[dict]:
    """Papers for the S2 title gap fill: no DOI, or a DOI that neither OpenAlex
    nor S2 knows (S2 knows nothing when its cache holds a null for the DOI)."""
    out = []
    for pid in sorted(groups):
        g = groups[pid]
        doi = g["doi"]
        if doi and (doi in works or s2_by_id.get("DOI:" + doi, 0) is not None):
            continue
        _, best = _best_entry(g["entries"])
        creators = _zotero_creators(best)
        out.append(
            {
                "paper_id": pid,
                "doi": doi,
                "title": str(best.get("title") or "").strip(),
                "year": parse_year(best.get("year")),
                "first_family": creators[0][0] if creators else "",
            }
        )
    return out


def resolve_matches(candidates, matches, library_dois, library_corpus_ids=()) -> dict:
    """Decisions of the title match, from the cached answers.

    Returns {"decisions": Counter, "overrides": {paper_id: doi}, "records":
    {paper_id: S2 record}, "conflicts": [...], "examples": [...]}. Every accepted match
    is checked: when its DOI (other than the paper's own) or its corpus id already
    belongs to another library paper (``library_dois``; ``library_corpus_ids`` of the
    records found by DOI; or a match accepted earlier), the match is a conflict and
    nothing is attached. A DOI found for a paper without one is assigned."""
    res = {
        "decisions": Counter(),
        "overrides": {},
        "records": {},
        "conflicts": [],
        "examples": [],
    }
    taken = set(library_dois)
    taken_cids = set(library_corpus_ids)
    for c in sorted(candidates, key=lambda c: c["paper_id"]):
        entry = matches.get(s2_api.match_key(c["paper_id"], c["title"], c["year"]))
        if entry is None:
            res["decisions"]["unasked"] += 1
            continue
        rec = entry.get("record")
        decision = s2_api.decide_match(c, rec)
        res["decisions"][decision] += 1
        if decision != "accepted":
            continue
        found = norm_doi(((rec.get("externalIds") or {}).get("DOI")) or "")
        cid = rec.get("corpusId")
        why = None
        if found and found != c["doi"] and found in taken:
            why = "doi"
        elif isinstance(cid, int) and cid in taken_cids:
            why = "corpus_id"
        if why:
            res["conflicts"].append(
                {
                    "paper_id": c["paper_id"],
                    "doi": found,
                    "corpus_id": cid,
                    "reason": why,
                    "title": c["title"],
                }
            )
            continue
        if not c["doi"] and found:
            taken.add(found)
            res["overrides"][c["paper_id"]] = found
        if isinstance(cid, int):
            taken_cids.add(cid)
        res["records"][c["paper_id"]] = rec
        if len(res["examples"]) < 2:
            res["examples"].append(
                {
                    "paper_id": c["paper_id"],
                    "zotero_title": c["title"],
                    "zotero_year": c["year"],
                    "first_family": c["first_family"],
                    "s2_title": rec.get("title"),
                    "s2_year": rec.get("year"),
                    "s2_doi": found or None,
                    "s2_first_author": ((rec.get("authors") or [{}])[0] or {}).get(
                        "name"
                    ),
                    "match_score": rec.get("matchScore"),
                }
            )
    return res


def s2_records_by_paper(tables, s2_by_id: dict, title_records: dict):
    """({paper_id: S2 record}, {paper_id: 'doi' | 'title'}): the record asked by
    DOI when S2 has one, else the accepted title match."""
    records, via = {}, {}
    for r in tables["papers"]:
        pid, doi = r[0], r[1]
        rec = s2_by_id.get("DOI:" + doi) if doi else None
        if rec and rec.get("paperId"):
            records[pid], via[pid] = rec, "doi"
        elif title_records.get(pid, {}).get("paperId"):
            records[pid], via[pid] = title_records[pid], "title"
    return records, via


def corpus_map(records: dict):
    """({corpus id: paper_id} for ids held by one paper, {corpus id: [paper_ids]}
    for ids held by several, sorted list of all corpus ids)."""
    held = defaultdict(list)
    for pid, rec in records.items():
        cid = rec.get("corpusId")
        if isinstance(cid, int):
            held[cid].append(pid)
    unique = {c: p[0] for c, p in held.items() if len(p) == 1}
    ambiguous = {c: sorted(p) for c, p in held.items() if len(p) > 1}
    return unique, ambiguous, sorted(held)


def extend_papers(tables, records: dict, overrides: dict) -> None:
    """Append the v2 columns to every papers row."""
    for r in tables["papers"]:
        pid = r[0]
        rec = records.get(pid) or {}
        doi_source = "s2_title" if pid in overrides else ("zotero" if r[1] else None)
        r.extend(
            [
                rec.get("paperId"),
                rec.get("corpusId"),
                rec.get("citationCount"),
                rec.get("influentialCitationCount"),
                rec.get("referenceCount"),
                (rec.get("tldr") or {}).get("text") or None,
                doi_source,
            ]
        )


def merge_pair_rows(rows) -> dict:
    """One edge from the S2 rows of a (citing, cited) pair.

    ``rows``: [(citationid, isinfluential, contexts JSON, intents JSON)]. Returns
    {"influential", "intents" (sorted union), "contexts": [(text, intents)], "cut"}.
    A context repeated across rows is kept once (its intents are merged)."""
    infl, all_intents, contexts, index, cut = False, set(), [], {}, 0
    for _cid, isinf, ctx_json, int_json in sorted(rows, key=lambda r: r[0]):
        infl = infl or bool(isinf)
        ctxs = json.loads(ctx_json) if ctx_json else []
        ints = json.loads(int_json) if int_json else []
        for lst in ints if isinstance(ints, list) else []:
            all_intents.update(
                x for x in (lst if isinstance(lst, list) else []) if isinstance(x, str)
            )
        for i, raw in enumerate(ctxs if isinstance(ctxs, list) else []):
            text, was_cut = clean_context(raw)
            cut += was_cut
            if not text:
                continue
            here = (
                ints[i]
                if isinstance(ints, list)
                and i < len(ints)
                and isinstance(ints[i], list)
                else []
            )
            here = {x for x in here if isinstance(x, str)}
            if text in index:
                index[text].update(here)
            else:
                index[text] = here
                contexts.append(text)
    return {
        "influential": infl,
        "intents": sorted(all_intents),
        "contexts": [(t, sorted(index[t])) for t in contexts],
        "cut": cut,
    }


def read_s2_store(path: str, cid_to_pid: dict, library_cids=()):
    """Edges between library papers from the S2 citation store.

    Returns ({(citing_pid, cited_pid): merged edge}, info). ``info`` has the store's
    release_id and mode, its row count and how many of ``library_cids`` its
    library_ids table lacks (None when the store has no such table)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
        n_rows = con.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
        try:
            have = {r[0] for r in con.execute("SELECT corpusid FROM library_ids")}
            lacking = len(set(library_cids) - have)
        except sqlite3.OperationalError:
            lacking = None
        ids = sorted(cid_to_pid)
        by_pair = defaultdict(list)
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = (
                "SELECT citationid, citing, cited, isinfluential, contexts, intents FROM citations "
                f"WHERE citing IN ({','.join('?' * len(chunk))})"
            )
            for cid, citing, cited, isinf, ctx, ints in con.execute(q, chunk):
                a, b = cid_to_pid.get(citing), cid_to_pid.get(cited)
                if a and b and a != b:
                    by_pair[(a, b)].append((cid, isinf, ctx, ints))
    finally:
        con.close()
    edges = {pair: merge_pair_rows(rows) for pair, rows in by_pair.items()}
    info = {
        "release_id": meta.get("release_id"),
        "mode": meta.get("mode"),
        "rows": n_rows,
        "library_ids_missing": lacking,
    }
    return edges, info


def build_cites(tables, s2_edges: dict) -> dict:
    """Replace tables["cites"] (v1: OpenAlex pairs) by the union with the S2 edges,
    and fill cite_contexts. Returns the counts the QA reports."""
    oa = {(r[0], r[1]) for r in tables["cites"]}
    rows, ctx_rows = [], []
    by_source = Counter()
    with_context = n_ctx = n_cut = 0
    for pair in sorted(oa | set(s2_edges)):
        e = s2_edges.get(pair)
        source = "both" if e and pair in oa else ("s2" if e else "openalex")
        by_source[source] += 1
        if e is None:
            rows.append([pair[0], pair[1], source, None, None, None])
            continue
        rows.append(
            [
                pair[0],
                pair[1],
                source,
                e["influential"],
                e["intents"],
                len(e["contexts"]),
            ]
        )
        for k, (text, intents) in enumerate(e["contexts"], 1):
            ctx_rows.append([pair[0], pair[1], k, text, intents])
        with_context += bool(e["contexts"])
        n_ctx += len(e["contexts"])
        n_cut += e["cut"]
    tables["cites"] = rows
    tables["cite_contexts"] = ctx_rows
    return {
        "by_source": {k: by_source.get(k, 0) for k in ("openalex", "s2", "both")},
        "with_context": with_context,
        "total_contexts": n_ctx,
        "cut_contexts": n_cut,
    }


def external_citers(aux: dict) -> dict:
    """{OpenAlex work id: sorted library papers citing it} for every referenced
    work that is not a library paper; a work counts once per citing paper."""
    cited = defaultdict(set)
    oa_to_paper = aux["oa_to_paper"]
    for pid, refs in aux["refs_by_paper"].items():
        for r in refs:
            if r not in oa_to_paper:
                cited[r].add(pid)
    return {w: sorted(p) for w, p in cited.items() if len(p) >= MIN_EXTERNAL_CITERS}


def build_external(tables, citers: dict, ext_works: dict, library_dois=()) -> dict:
    """Fill external_works and cites_external; works the cache lacks keep only
    their id and count. Returns what the QA reports."""
    rows, links, with_meta = [], [], 0
    for wid in sorted(citers):
        w = ext_works.get(wid)
        n = len(citers[wid])
        if w:
            with_meta += 1
            auth = ((w.get("authorships") or [{}])[0] or {}).get("author") or {}
            src = ((w.get("primary_location") or {}).get("source")) or {}
            rows.append(
                [
                    f"oa:{wid}",
                    wid,
                    norm_doi(w.get("doi") or "") or None,
                    w.get("title") or w.get("display_name") or None,
                    w.get("publication_year"),
                    w.get("type"),
                    w.get("cited_by_count"),
                    auth.get("display_name") or None,
                    src.get("display_name") or None,
                    n,
                ]
            )
        else:
            rows.append([f"oa:{wid}", wid, None, None, None, None, None, None, None, n])
        links.extend([pid, f"oa:{wid}"] for pid in citers[wid])
    tables["external_works"] = rows
    tables["cites_external"] = sorted(links)
    top = sorted(rows, key=lambda r: (-r[9], r[0]))[:20]
    lib = set(library_dois)
    same_doi = [r for r in rows if r[2] and r[2] in lib]  # flagged, not dropped
    return {
        "doi_equals_library_paper": {
            "count": len(same_doi),
            "examples": [[r[0], r[2]] for r in same_doi[:5]],
        },
        "count": len(rows),
        "with_metadata": with_meta,
        "cites_external_rows": len(links),
        "min_library_citers": MIN_EXTERNAL_CITERS,
        "top_cited_by_library": [
            {"work_id": r[0], "title": r[3], "year": r[4], "n_library_citers": r[9]}
            for r in top
        ],
    }


def build_embeddings(tables, records: dict, emb_cache: dict) -> None:
    rows = []
    for pid in sorted(records):
        e = emb_cache.get(records[pid].get("paperId"))
        if e and e.get("vector"):
            rows.append([pid, e.get("model"), tuple(e["vector"])])
    tables["paper_embeddings"] = rows


def scan_for_keys(dirs, keys: dict) -> list[str]:
    """Violations for every file under ``dirs`` that contains the value of a key
    in ``keys`` ({env name: value}). The value itself is never reported."""
    needles = {n: v.encode("utf-8") for n, v in keys.items() if v}
    if not needles:
        return []
    viol = []
    for d in dirs:
        for root, _dirs, files in os.walk(d):
            for fn in sorted(files):
                path = os.path.join(root, fn)
                tail, found = b"", set()
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(1 << 24)
                        if not chunk:
                            break
                        buf = tail + chunk
                        found.update(n for n, v in needles.items() if v in buf)
                        keep = max(map(len, needles.values())) - 1
                        tail = buf[-keep:] if keep else b""
                for n in sorted(found):
                    viol.append(f"{path}: contains the value of {n}")
    return viol


# --------------------------------------------------------------------------
# hard checks
# --------------------------------------------------------------------------


def _null(v) -> bool:
    return v is None or v == ""


def _fk_parts(fk):
    """(columns, referenced table, referenced columns) of a foreign key given as
    (col, table, col) or (cols, table, cols)."""
    cols, ref, rcols = fk
    return (
        [cols] if isinstance(cols, str) else list(cols),
        ref,
        [rcols] if isinstance(rcols, str) else list(rcols),
    )


def hard_checks(tables: dict) -> list[str]:
    """Violations of the hard checks (empty when all pass)."""
    viol = []

    def example(items, n=3):
        return ", ".join(repr(x) for x in list(items)[:n])

    for name, spec in TABLES.items():
        rows = tables.get(name, [])
        cols = [c for c, _ in spec["cols"]]
        ix = {c: i for i, c in enumerate(cols)}
        for label, keycols in [("primary key", spec["pk"])] + [
            ("unique key", u) for u in spec.get("unique", [])
        ]:
            seen, dups = set(), []
            for r in rows:
                k = tuple(r[ix[c]] for c in keycols)
                if k in seen:
                    dups.append(k)
                seen.add(k)
            if dups:
                viol.append(
                    f"{name}: {len(dups)} duplicate {label} values, e.g. {example(dups)}"
                )
        for fk in spec.get("fks", []):
            fcols, ref, rcols = _fk_parts(fk)
            ref_ix = [[c for c, _ in TABLES[ref]["cols"]].index(c) for c in rcols]
            valid = {tuple(r[i] for i in ref_ix) for r in tables.get(ref, [])}
            bad = []
            for r in rows:
                k = tuple(r[ix[c]] for c in fcols)
                if not any(_null(v) for v in k) and k not in valid:
                    bad.append(k[0] if len(k) == 1 else k)
            if bad:
                viol.append(
                    f"{name}.{','.join(fcols)}: {len(bad)} values missing from "
                    f"{ref}.{','.join(rcols)}, e.g. {example(bad)}"
                )
    selfc = [r for r in tables.get("cites", []) if r[0] == r[1]]
    if selfc:
        viol.append(f"cites: {len(selfc)} self-citation rows, e.g. {example(selfc)}")
    unordered = [r for r in tables.get("coupled", []) if not r[0] < r[1]]
    if unordered:
        viol.append(
            f"coupled: {len(unordered)} rows with paper_a >= paper_b, e.g. {example(unordered)}"
        )
    parent = {r[0]: r[3] for r in tables.get("topics", [])}
    cyclic = []
    for start in parent:
        seen, cur = set(), start
        while not _null(cur) and cur in parent:
            if cur in seen:
                cyclic.append(start)
                break
            seen.add(cur)
            cur = parent[cur]
    if cyclic:
        viol.append(
            f"topics: {len(cyclic)} topics on a parent cycle, e.g. {example(cyclic)}"
        )
    return viol


def scan_for_nul(out: str) -> list[str]:
    """Violations for every CSV under ``out`` that holds a NUL byte (PostgreSQL text cannot)."""
    viol = []
    for fn in sorted(os.listdir(out)):
        if fn.endswith(".csv"):
            with open(os.path.join(out, fn), "rb") as f:
                while chunk := f.read(1 << 24):
                    if b"\x00" in chunk:
                        viol.append(f"{fn}: contains a NUL character")
                        break
    return viol


def hard_checks_v2(tables: dict) -> list[str]:
    """Violations in the v2 columns and tables (keys and foreign keys are in
    hard_checks): source tags, arrays, contexts, vectors."""
    viol = []

    def example(items, n=3):
        return ", ".join(repr(x) for x in list(items)[:n])

    def arrays_bad(rows, i):
        bad = []
        for r in rows:
            v = r[i]
            if v is None:
                continue
            try:
                if not isinstance(v, list) or parse_pg_array(pg_array(v)) != [
                    str(x) for x in v
                ]:
                    bad.append(v)
            except ValueError:
                bad.append(v)
        return bad

    cites = tables.get("cites", [])
    bad = [r for r in cites if len(r) > 5 and r[2] not in ("openalex", "s2", "both")]
    if bad:
        viol.append(
            f"cites: {len(bad)} rows with an invalid source, e.g. {example(r[2] for r in bad)}"
        )
    bad = [
        r
        for r in cites
        if len(r) > 5
        and (
            (
                r[2] == "openalex"
                and (r[3] is not None or r[4] is not None or r[5] is not None)
            )
            or (
                r[2] != "openalex"
                and (
                    not isinstance(r[3], bool)
                    or r[4] is None
                    or not isinstance(r[5], int)
                )
            )
        )
    ]
    if bad:
        viol.append(
            f"cites: {len(bad)} rows whose S2 columns do not match their source, e.g. {example(bad, 2)}"
        )
    bad = arrays_bad([r for r in cites if len(r) > 5], 4)
    if bad:
        viol.append(
            f"cites.intents: {len(bad)} malformed array values, e.g. {example(bad)}"
        )
    ctx = tables.get("cite_contexts", [])
    bad = arrays_bad(ctx, 4)
    if bad:
        viol.append(
            f"cite_contexts.intents: {len(bad)} malformed array values, e.g. {example(bad)}"
        )
    bad = [
        r
        for r in ctx
        if not r[3]
        or len(r[3]) > MAX_CONTEXT
        or any(unicodedata.category(c) in ("Cc", "Cs") for c in r[3])
    ]
    if bad:
        viol.append(
            f"cite_contexts.context: {len(bad)} empty, over-long or control-character values"
        )
    bad = [
        r[0]
        for r in tables.get("papers", [])
        if len(r) > 16 and r[16] not in (None, "zotero", "s2_title")
    ]
    if bad:
        viol.append(
            f"papers.doi_source: {len(bad)} invalid values, e.g. {example(bad)}"
        )
    bad = [
        r[0]
        for r in tables.get("paper_embeddings", [])
        if len(r[2]) != s2_api.EMBED_DIM
        or not all(math.isfinite(x) for x in r[2])
        or not r[1]
    ]
    if bad:
        viol.append(
            f"paper_embeddings: {len(bad)} rows without a model or {s2_api.EMBED_DIM} finite numbers, e.g. {example(bad)}"
        )
    return viol


# --------------------------------------------------------------------------
# output files
# --------------------------------------------------------------------------


def pg_array(items) -> str:
    """A PostgreSQL array literal of strings: ``{a,b}``, ``{}`` when empty. An element
    with a comma, brace, quote, backslash or whitespace (or empty, or NULL) is double-quoted."""
    out = []
    for x in items:
        x = str(x).replace("\x00", "")  # PostgreSQL text cannot hold NUL (as in _fmt)
        if x == "" or x.upper() == "NULL" or re.search(r'[\s,{}"\\]', x):
            x = '"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"'
        out.append(x)
    return "{" + ",".join(out) + "}"


def parse_pg_array(text: str) -> list[str]:
    """Inverse of pg_array for one-dimensional text arrays; ValueError when malformed."""
    if len(text) < 2 or text[0] != "{" or text[-1] != "}":
        raise ValueError(f"not an array literal: {text!r}")
    body, out, i = text[1:-1], [], 0
    if body == "":
        return out
    while True:
        if i < len(body) and body[i] == '"':
            i += 1
            cur = []
            while True:
                if i >= len(body):
                    raise ValueError(f"unterminated quote: {text!r}")
                c = body[i]
                if c == "\\":
                    if i + 1 >= len(body):
                        raise ValueError(f"dangling backslash: {text!r}")
                    cur.append(body[i + 1])
                    i += 2
                elif c == '"':
                    i += 1
                    break
                else:
                    cur.append(c)
                    i += 1
            out.append("".join(cur))
        else:
            k = i
            while i < len(body) and body[i] != ",":
                if body[i] in '{}"\\' or body[i].isspace():
                    raise ValueError(f"unquoted special character: {text!r}")
                i += 1
            if i == k:
                raise ValueError(f"empty element: {text!r}")
            out.append(body[k:i])
        if i == len(body):
            return out
        if body[i] != ",":
            raise ValueError(f"expected a comma: {text!r}")
        i += 1


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):  # a text[] column
        return pg_array(v)
    if isinstance(v, tuple):  # a vector column: pgvector's text form
        return "[" + ",".join(repr(float(x)) for x in v) + "]"
    v = str(v)
    if "\x00" in v:  # PostgreSQL text cannot hold NUL
        v = v.replace("\x00", "")
    return s2_api.clean_text(v)


def write_csv(path: str, header: list[str], rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])


SCHEMA_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def check_schema_name(name: str) -> str:
    if not SCHEMA_NAME_RE.match(name or "") or len(name) > 50:
        raise ValueError(
            f"bad schema name {name!r}: lowercase letters, digits and _ only"
        )
    return name


def _table_ddl(schema: str, name: str, spec: dict) -> list[str]:
    lines = []
    for c, t in spec["cols"]:
        nn = c in spec["pk"] or c in spec.get("notnull", [])
        lines.append(f"    {c} {t}{' NOT NULL' if nn else ''}")
    lines.append(f"    PRIMARY KEY ({', '.join(spec['pk'])})")
    for u in spec.get("unique", []):
        lines.append(f"    UNIQUE ({', '.join(u)})")
    for fk in spec.get("fks", []):
        fcols, ref, rcols = _fk_parts(fk)
        lines.append(
            f"    FOREIGN KEY ({', '.join(fcols)}) REFERENCES {schema}.{ref} ({', '.join(rcols)})"
        )
    for chk in spec.get("checks", []):
        lines.append(f"    CHECK ({chk})")
    out = [f"CREATE TABLE {schema}.{name} (", ",\n".join(lines), ");"]
    for cols in spec.get("indexes", []):
        out.append(
            f"CREATE INDEX {name}_{'_'.join(cols)}_idx ON {schema}.{name} ({', '.join(cols)});"
        )
    out.append("")
    return out


def schema_sql(schema: str = "zstruct") -> str:
    out = [
        "-- Structural layer schema. Generated by build_struct_layer.py; review before running.",
        f"CREATE SCHEMA {schema};",
        "",
    ]
    for name, spec in TABLES.items():
        out.extend(_table_ddl(schema, name, spec))
    return "\n".join(out)


def _copy_lines(schema: str) -> list[str]:
    out = []
    for name, spec in TABLES.items():
        cols = ", ".join(c for c, _ in spec["cols"])
        out.append(
            f"\\copy {schema}.{name} ({cols}) FROM '{name}.csv' WITH (FORMAT csv, HEADER true)"
        )
    return out


def load_sql(schema: str = "zstruct") -> str:
    out = [
        f"-- Load the CSV files in this directory into the {schema} schema.",
        "-- Run schema.sql first, then: psql -d DATABASE -f load.sql  (from this directory).",
        "\\set ON_ERROR_STOP on",
        "BEGIN;",
    ]
    out.extend(_copy_lines(schema))
    out.append("COMMIT;")
    out.append("")
    return "\n".join(out)


def apply_sql(schema: str, workspace: str) -> str:
    """One transaction: build ``<schema>_next`` (tables, loads, the paper_document
    view), analyze it. Nothing outside the new schema is created or changed; the
    only outside table it reads is public.lightrag_doc_status (in the view)."""
    nxt = f"{schema}_next"
    ws = workspace.replace("'", "''")
    out = [
        f"-- Build {nxt} in ONE transaction. Generated by build_struct_layer.py; review before running.",
        f"-- Run from this directory:  psql -d DATABASE -f apply_{schema}.sql",
        f"-- Then swap:                psql -d DATABASE -f swap_{schema}.sql",
        f"-- Nothing outside {nxt} is created or changed (the vector extension already exists).",
        "\\set ON_ERROR_STOP on",
        "BEGIN;",
        f"DROP SCHEMA IF EXISTS {nxt} CASCADE;",
        "",
        schema_sql(nxt).rstrip("\n"),
        "",
    ]
    out.extend(_copy_lines(nxt))
    out += [
        "",
        "-- Join to the knowledge base at read time: the processed LightRAG documents of one",
        "-- workspace. A document's file_path starts with its Zotero attachment key and two underscores.",
        f"CREATE VIEW {nxt}.paper_document AS",
        "SELECT a.paper_id, a.attachment_key, d.id AS doc_id, d.file_path, d.status",
        f"FROM {nxt}.paper_attachments a",
        "JOIN public.lightrag_doc_status d ON split_part(d.file_path, '__', 1) = a.attachment_key",
        f"WHERE d.workspace = '{ws}' AND d.status = 'processed';",
        "",
        f"COMMENT ON SCHEMA {nxt} IS 'Structural layer of the Zotero knowledge base (citation graph v2). "
        "Built by build_struct_layer.py; rebuilt, not edited.';",
        "",
    ]
    out.extend(f"ANALYZE {nxt}.{name};" for name in TABLES)
    out += ["COMMIT;", ""]
    return "\n".join(out)


def swap_sql(schema: str) -> str:
    """One transaction: the live schema becomes ``<schema>_prev``, ``<schema>_next`` goes live."""
    return "\n".join(
        [
            f"-- Swap {schema}_next into place. Generated by build_struct_layer.py; review before running.",
            f"-- Undo: ALTER SCHEMA {schema} RENAME TO {schema}_next; ALTER SCHEMA {schema}_prev RENAME TO {schema};",
            "\\set ON_ERROR_STOP on",
            "BEGIN;",
            f"DROP SCHEMA IF EXISTS {schema}_prev CASCADE;",
            "DO $swap$ BEGIN",
            f"    IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = '{schema}') THEN",
            f"        ALTER SCHEMA {schema} RENAME TO {schema}_prev;",
            "    END IF;",
            "END $swap$;",
            f"ALTER SCHEMA {schema}_next RENAME TO {schema};",
            "COMMIT;",
            "",
        ]
    )


def drop_prev_sql(schema: str) -> str:
    return "\n".join(
        [
            f"-- Removes the previous structural layer ({schema}_prev) and nothing else.",
            f"DROP SCHEMA IF EXISTS {schema}_prev CASCADE;",
            "",
        ]
    )


# --------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------


def _wcc(nodes, edges):
    """Weakly connected components. Returns (parent map, root of the largest
    component by node count)."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    sizes = Counter(find(n) for n in parent)
    largest = min(sizes, key=lambda r: (-sizes[r], r)) if sizes else None
    return find, largest


def _graph_stats(paper_ids, extra_nodes, edges):
    edges = list(edges)
    nodes = set(paper_ids) | set(extra_nodes)
    find, largest = _wcc(nodes, edges)
    papers = set(paper_ids)
    touched = {n for e in edges for n in e} & papers
    in_largest = (
        sum(1 for p in papers if find(p) == largest) if largest is not None else 0
    )
    return {
        "edges": len(edges),
        "papers_with_edge": len(touched),
        "largest_component_paper_share": round(in_largest / len(papers), 6)
        if papers
        else 0.0,
        "largest_component_papers": in_largest,
    }


def compare_reference(tables, ref: dict):
    """Compare our citation edges (as DOI pairs) with the reference file's."""
    ours_all = {
        (r[0][4:], r[1][4:])
        for r in tables["cites"]
        if r[0].startswith("doi:") and r[1].startswith("doi:")
    }
    # citing DOIs we know OpenAlex for = papers with an OpenAlex record
    ours_citing = {r[1] for r in tables["papers"] if r[1] and r[2]}
    common = ours_citing & set(ref)
    ours = {e for e in ours_all if e[0] in common}
    theirs = {(c, d) for c in common for d in (ref[c].get("cited") or [])}
    both, only_ours, only_ref = ours & theirs, ours - theirs, theirs - ours
    union = len(ours | theirs)
    jac = len(both) / union if union else 1.0
    return {
        "citing_dois_ours": len(ours_citing),
        "citing_dois_reference": len(ref),
        "citing_dois_common": len(common),
        "edges_both": len(both),
        "edges_only_ours": len(only_ours),
        "edges_only_reference": len(only_ref),
        "jaccard": round(jac, 6),
        "min_jaccard": ORACLE_MIN_JACCARD,
        "passed": jac >= ORACLE_MIN_JACCARD,
        "differing_only_ours": [list(e) for e in sorted(only_ours)[:10]],
        "differing_only_reference": [list(e) for e in sorted(only_ref)[:10]],
    }


def build_qa(
    tables, aux, harvest_stats, n_library_dois, n_unmatched, violations, oracle, v2=None
):
    P = {
        name: {c: i for i, (c, _) in enumerate(spec["cols"])}
        for name, spec in TABLES.items()
    }
    papers = tables["papers"]
    pid_ix = P["papers"]
    n = len(papers)
    paper_ids = [r[0] for r in papers]
    has = lambda name, col: len({r[P[name][col]] for r in tables[name]})  # noqa: E731

    cites = [(r[0], r[1]) for r in tables["cites"]]
    coupled = [(r[0], r[1]) for r in tables["coupled"]]
    pa = [(r[0], r[1]) for r in tables["paper_authors"]]
    pv = [(r[0], r[pid_ix["venue_id"]]) for r in papers if r[pid_ix["venue_id"]]]
    pt = [(r[0], r[1]) for r in tables["paper_topics"]]

    graph_c_edges = (
        cites
        + coupled
        + [(p, "author|" + a) for p, a in pa]
        + [(p, "venue|" + v) for p, v in pv]
        + [(p, "topic|" + t) for p, t in pt]
    )
    extra = (
        {"author|" + a for _, a in pa}
        | {"venue|" + v for _, v in pv}
        | {"topic|" + t for _, t in pt}
    )
    graph_c = _graph_stats(paper_ids, extra, graph_c_edges)
    graph_c["papers_with_no_edge_of_any_kind"] = n - graph_c["papers_with_edge"]
    graph_c["nodes"] = n + len(extra)

    indeg = Counter(r[1] for r in tables["cites"])
    meta = {r[0]: r for r in papers}
    top_cited = [
        {
            "paper_id": pid,
            "title": meta[pid][pid_ix["title"]],
            "year": meta[pid][pid_ix["year"]],
            "cited_by_library_papers": c,
        }
        for pid, c in sorted(indeg.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    ]
    topic_name = {r[0]: r for r in tables["topics"]}
    n_by_topic, n_primary = Counter(), Counter()
    for r in tables["paper_topics"]:
        n_by_topic[r[1]] += 1
        n_primary[r[1]] += 1 if r[3] else 0
    largest_topics = [
        {
            "topic_id": t,
            "display_name": topic_name[t][1],
            "n_papers": c,
            "n_primary_papers": n_primary[t],
        }
        for t, c in sorted(n_by_topic.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    ]
    paper_primary = {r[0]: r[1] for r in tables["paper_topics"] if r[3]}
    field_of = {}
    for tid, row in topic_name.items():
        cur, seen = row, set()
        while cur and cur[2] != "field" and cur[3] and cur[3] not in seen:
            seen.add(cur[3])
            cur = topic_name.get(cur[3])
        if cur and cur[2] == "field":
            field_of[tid] = cur[1] or cur[0]
    fields = Counter()
    for pid in paper_ids:
        t = paper_primary.get(pid)
        fields[field_of.get(t, "(none)") if t else "(none)"] += 1
    field_dist = [
        {"field": f, "n_papers": c}
        for f, c in sorted(fields.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    qa = {
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "coverage": {
            "attachments": len(tables["paper_attachments"]),
            "papers": n,
            "library_dois_distinct": n_library_dois,
            "papers_with_doi": sum(1 for r in papers if r[pid_ix["doi"]]),
            "papers_with_openalex_record": sum(
                1 for r in papers if r[pid_ix["openalex_id"]]
            ),
            "papers_with_author": has("paper_authors", "paper_id"),
            "papers_with_topic": has("paper_topics", "paper_id"),
            "papers_with_keyword": has("paper_keywords", "paper_id"),
            "papers_with_venue": len(pv),
            "papers_with_reference_list": aux["n_with_reference_list"],
            "unmatched_dois": n_unmatched,
        },
        "graph": {
            "a_cites_only": _graph_stats(paper_ids, (), cites),
            "b_cites_and_coupled": _graph_stats(paper_ids, (), cites + coupled),
            "c_heterogeneous": graph_c,
        },
        "top_cited_within_library": top_cited,
        "largest_topics": largest_topics,
        "field_distribution": field_dist,
        "author_merge": aux["author_merge"],
        "requests": harvest_stats,
        "hard_checks": {"passed": not violations, "violations": violations},
        "oracle": oracle,
    }
    if v2 is not None:
        qa["v2"] = v2
    return qa


def _render_v2(L, v, table) -> None:
    c = v["s2_coverage"]
    L.append("## Semantic Scholar coverage")
    table(
        ["measure", "papers"],
        [
            ["library papers", c["papers"]],
            ["found by DOI", c["found_by_doi"]],
            ["found by title", c["found_by_title"]],
            ["with a TLDR", c["with_tldr"]],
            ["with an embedding", c["with_embedding"]],
        ],
    )
    m = v["title_match"]
    L.append("## Title match")
    L.append(
        f"Candidates: {m['candidates']}. Decisions: "
        + (", ".join(f"{k} {n}" for k, n in sorted(m["decisions"].items())) or "none")
        + f". DOIs assigned: {m['dois_assigned']}. DOI conflicts (found DOI already belongs to another "
        f"library paper, not assigned): {m['doi_conflicts']}."
    )
    L.append("")
    if m["accepted_examples"]:
        L.append("Accepted examples (check the rules by eye):")
        for e in m["accepted_examples"]:
            L.append(
                f'- {e["paper_id"]}: Zotero "{e["zotero_title"]}" ({e["zotero_year"]}, first creator '
                f'{e["first_family"]}) -> S2 "{e["s2_title"]}" ({e["s2_year"]}, first author '
                f"{e['s2_first_author']}, DOI {e['s2_doi']}, matchScore {e['match_score']})"
            )
        L.append("")
    if m["conflict_examples"]:
        L.append("DOI conflicts (up to 5):")
        L.extend(
            f"- {e['paper_id']} -> {e['doi'] or e['corpus_id']} ({e['reason']})"
            for e in m["conflict_examples"]
        )
        L.append("")
    k = v["cites"]
    L.append("## Citation edges and contexts")
    table(
        ["measure", "count"],
        [
            ["edges, source openalex", k["by_source"]["openalex"]],
            ["edges, source s2", k["by_source"]["s2"]],
            ["edges, source both", k["by_source"]["both"]],
            ["edges with at least one context", k["with_context"]],
            ["contexts", k["total_contexts"]],
            ["contexts cut at 2,000 characters", k["cut_contexts"]],
        ],
    )
    a = v["ambiguous_corpus_ids"]
    L.append(
        f"Ambiguous corpus ids (shared by several library papers, mapped to none): {a['count']}."
    )
    L.append("")
    x = v["external_works"]
    L.append("## Works outside the library")
    L.append(
        f"{x['count']} works cited by at least {x['min_library_citers']} library papers "
        f"({x['with_metadata']} with metadata); {x['cites_external_rows']} citing links."
    )
    L.append(
        f"Outside works whose DOI equals a library paper's DOI (flagged, kept): "
        f"{x['doi_equals_library_paper']['count']}."
    )
    L.append("")
    table(
        ["title", "year", "library papers citing"],
        [
            [t["title"] or t["work_id"], t["year"] or "", t["n_library_citers"]]
            for t in x["top_cited_by_library"]
        ],
    )
    st = v["s2_store"]
    L.append("## S2 citation store")
    if not st["used"]:
        L.append(st["note"])
    else:
        L.append(
            f"Release {st['release_id']}, mode {st['mode']}, {st['rows']} rows; library corpus ids the store "
            f"lacks: {st['library_ids_missing']}."
        )
    L.append("")
    L.append("## Legs")
    table(
        ["leg", "ran", "complete", "details"],
        [[n, g["ran"], g["complete"], g.get("note", "")] for n, g in v["legs"].items()],
    )
    kf = v["key_fallback"]
    L.append("## Keys")
    L.append(
        f"OpenAlex key set: {kf['openalex_key_set']}, rejected (run continued without it): "
        f"{kf['openalex_key_rejected']}. S2 key set: {kf['s2_key_set']}, rejected: {kf['s2_key_rejected']}."
    )
    L.append("")


def render_qa_md(qa: dict) -> str:
    L = ["# Structural layer QA", ""]

    def table(header, rows):
        L.append("| " + " | ".join(header) + " |")
        L.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            L.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
        L.append("")

    pct = lambda x: f"{100 * x:.2f}%"  # noqa: E731
    L.append("## Hard checks")
    hc = qa["hard_checks"]
    L.append("All passed." if hc["passed"] else "FAILED:")
    L.extend(f"- {v}" for v in hc["violations"])
    L.append("")
    L.append("## Requests")
    rq = qa["requests"]
    L.append(
        f"{rq['batches']} list requests, {rq['singles']} single-work requests, "
        f"{rq['failed']} failed after retries."
    )
    L.append("")
    L.append("## Row counts")
    table(["table", "rows"], sorted(qa["row_counts"].items()))
    L.append("## Coverage")
    table(["measure", "count"], list(qa["coverage"].items()))
    L.append("## Graph connectivity")
    names = {
        "a_cites_only": "(a) cites only",
        "b_cites_and_coupled": "(b) cites + coupled",
        "c_heterogeneous": "(c) all edges (papers, authors, venues, topics)",
    }
    table(
        [
            "graph",
            "edges",
            "papers with an edge",
            "papers in largest component",
            "share of papers",
        ],
        [
            [
                names[k],
                g["edges"],
                g["papers_with_edge"],
                g["largest_component_papers"],
                pct(g["largest_component_paper_share"]),
            ]
            for k, g in qa["graph"].items()
        ],
    )
    c = qa["graph"]["c_heterogeneous"]
    L.append(f"Papers with no edge of any kind: {c['papers_with_no_edge_of_any_kind']}")
    L.append("")
    L.append("## Most cited within the library (top 15)")
    table(
        ["title", "year", "cited by"],
        [
            [t["title"], t["year"], t["cited_by_library_papers"]]
            for t in qa["top_cited_within_library"]
        ],
    )
    L.append("## Largest topics (top 15)")
    table(
        ["topic", "papers", "primary"],
        [
            [t["display_name"], t["n_papers"], t["n_primary_papers"]]
            for t in qa["largest_topics"]
        ],
    )
    L.append("## Field distribution (by primary topic)")
    table(
        ["field", "papers"],
        [[f["field"], f["n_papers"]] for f in qa["field_distribution"]],
    )
    m = qa["author_merge"]
    L.append("## Author merge")
    L.append(
        f"Zotero-keyed authors merged into an OpenAlex author: {m['merged']}. "
        f"Kept as Zotero authors: {m['kept']} "
        f"(ambiguous: {m['kept_ambiguous']}, no match: {m['kept_no_match']}). "
        f"OpenAlex authorships skipped for lack of an author id: {m['skipped_authorships_without_id']}."
    )
    L.append("")
    if "v2" in qa:
        _render_v2(L, qa["v2"], table)
    o = qa["oracle"]
    L.append("## Oracle (reference citation file)")
    if o is None:
        L.append("Not run (no --reference-check).")
    else:
        L.append(
            f"Citing DOIs in both: {o['citing_dois_common']} "
            f"(ours {o['citing_dois_ours']}, reference {o['citing_dois_reference']}). "
            f"Edges in both: {o['edges_both']}; only ours: {o['edges_only_ours']}; "
            f"only reference: {o['edges_only_reference']}; Jaccard {o['jaccard']:.6f} "
            f"(threshold {o['min_jaccard']}): {'passed' if o['passed'] else 'FAILED'}."
        )
        L.append("")
        for label, key in (
            ("Only ours", "differing_only_ours"),
            ("Only reference", "differing_only_reference"),
        ):
            if o[key]:
                L.append(f"{label} (up to 10, citing -> cited):")
                L.extend(f"- {a} -> {b}" for a, b in o[key])
                L.append("")
    return "\n".join(L).rstrip() + "\n"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _load_json_file(path: str, default):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


S2_LEGS = ("papers", "embeddings", "match")


def _leg(ran: bool, stats: dict | None, failed_key="calls_failed", note="") -> dict:
    complete = bool(
        ran and stats is not None and stats.get("complete", not stats.get(failed_key))
    )
    out = {"ran": ran, "complete": complete}
    if note:
        out["note"] = note
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the structural layer tables.")
    ap.add_argument("--metadata", required=True, help="zotero_metadata.json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument(
        "--cache",
        help="OpenAlex works cache (default <out>/../cache/openalex_works.json)",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="ask OpenAlex (and S2) about every id again",
    )
    ap.add_argument(
        "--offline", action="store_true", help="make no request to either API"
    )
    ap.add_argument(
        "--reference-check", help="zotero_references.json to compare citations with"
    )
    ap.add_argument("--mailto", default=os.environ.get("OPENALEX_MAILTO", ""))
    ap.add_argument(
        "--schema", default="zstruct", help="PostgreSQL schema name (default zstruct)"
    )
    ap.add_argument(
        "--workspace",
        default="zotero",
        help="LightRAG workspace of the paper_document view",
    )
    ap.add_argument(
        "--s2-legs",
        default=",".join(S2_LEGS),
        help="comma list from papers,embeddings,match",
    )
    ap.add_argument(
        "--s2-max-calls", type=int, help="stop the S2 legs after this many requests"
    )
    ap.add_argument("--s2-store", help="SQLite citation store of s2_citations_sync.py")
    a = ap.parse_args(argv)

    try:
        schema = check_schema_name(a.schema)
    except ValueError as e:
        ap.error(str(e))
    legs = {x.strip() for x in a.s2_legs.split(",") if x.strip()}
    if legs - set(S2_LEGS):
        ap.error(f"--s2-legs: unknown leg(s) {', '.join(sorted(legs - set(S2_LEGS)))}")
    if a.s2_store and not os.path.exists(a.s2_store):
        ap.error(f"--s2-store: no such file: {a.s2_store}")

    _STATE["oa_key_rejected"] = False
    oa_key = os.environ.get("OPENALEX_API_KEY", "")
    s2_key = os.environ.get("S2_API_KEY", "")
    today = datetime.date.today().isoformat()
    out = a.out
    cache_path = a.cache or os.path.normpath(
        os.path.join(out, "..", "cache", "openalex_works.json")
    )
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    p_papers = os.path.join(cache_dir, "s2_papers.json")
    p_emb = os.path.join(cache_dir, "s2_embeddings.json")
    p_match = os.path.join(cache_dir, "s2_title_match.json")
    p_ext = os.path.join(cache_dir, "openalex_external.json")
    os.makedirs(out, exist_ok=True)
    with open(a.metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    store = load_cache(cache_path)
    dois = library_dois(metadata)
    groups = group_papers(metadata)
    works = store["works"]

    papers_cache = _load_json_file(p_papers, s2_api.new_papers_cache())
    papers_cache.setdefault("by_id", {})
    papers_cache.setdefault("fetched", {})
    emb_cache = _load_json_file(p_emb, {})
    matches = _load_json_file(p_match, {})
    ext_store = load_cache(p_ext)

    stats = {"batches": 0, "singles": 0, "failed": 0}
    client = None if a.offline else s2_api.S2Client(s2_key, max_calls=a.s2_max_calls)
    legs_qa = {}
    failed_requests = 0

    # ---- OpenAlex works of the library DOIs
    if not a.offline:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            stats = harvest(dois, store, a.mailto, oa_key, a.refresh)
        finally:
            write_json(cache_path, store)
    legs_qa["openalex_works"] = _leg(
        not a.offline,
        {"complete": not stats["failed"]},
        note=f"{stats['batches']} list requests, {stats['singles']} single, {stats['failed']} failed",
    )
    failed_requests += stats["failed"]

    # ---- S2 legs: papers by DOI, then the title gap fill
    s2_stats = {}
    if client and "papers" in legs:
        s2_stats["papers"] = s2_api.papers_leg(
            client,
            ["DOI:" + d for d in dois],
            papers_cache,
            today,
            refresh=a.refresh,
            save=lambda: write_json(p_papers, papers_cache),
        )
    candidates = title_candidates(groups, works, papers_cache["by_id"])
    if client and "match" in legs:
        s2_stats["match"] = s2_api.match_leg(
            client,
            candidates,
            matches,
            papers_cache,
            today,
            save=lambda: (
                write_json(p_match, matches),
                write_json(p_papers, papers_cache),
            ),
        )
    doi_cids = {
        r["corpusId"]
        for d in dois
        for r in [papers_cache["by_id"].get("DOI:" + d)]
        if r and isinstance(r.get("corpusId"), int)
    }
    res = resolve_matches(candidates, matches, dois, doi_cids)
    overrides = res["overrides"]

    # ---- OpenAlex records of the DOIs the title match found
    if not a.offline and overrides:
        try:
            for pid, doi in sorted(overrides.items()):
                if doi in works:
                    continue
                stats["singles"] += 1
                try:
                    w = fetch_one(doi, a.mailto, oa_key)
                    if w:
                        works[doi] = w
                except Exception as e:
                    stats["failed"] += 1
                    failed_requests += 1
                    print(
                        f"WARN: OpenAlex lookup of {doi} failed: {e}", file=sys.stderr
                    )
                time.sleep(PAUSE)
        finally:
            write_json(cache_path, store)

    if a.refresh:  # ask about every outside work again, the merged and deleted ones too
        ext_store["merged"], ext_store["deleted"] = {}, []

    def assemble_all():
        return assemble(
            metadata,
            works,
            overrides,
            ext_store.get("merged"),
            ext_store.get("deleted") or (),
        )

    tables, aux = assemble_all()
    unmatched = [d for d in dois if d not in works]

    # ---- OpenAlex works outside the library; the ids the list filter does not return are asked
    # one at a time, and when that finds merged or deleted works the tables are assembled again
    citers = external_citers(aux)
    ext_stats = {"batches": 0, "failed": 0}
    res_stats = {"asked": 0, "found": 0, "merged": 0, "deleted": 0, "failed": 0}
    if not a.offline:
        try:
            ext_stats = harvest_external(
                sorted(citers), ext_store, a.mailto, oa_key, a.refresh
            )
            res_stats = resolve_external_unknown(ext_store, a.mailto, oa_key)
        finally:
            write_json(p_ext, ext_store)
        if res_stats["merged"] or res_stats["deleted"]:
            tables, aux = assemble_all()
            citers = external_citers(aux)
    legs_qa["openalex_external"] = _leg(
        not a.offline,
        {"complete": not (ext_stats["failed"] or res_stats["failed"])},
        note=f"{ext_stats['batches']} list requests, {ext_stats['failed']} failed; "
        f"{res_stats['asked']} single lookups of unknown ids ({res_stats['found']} found, "
        f"{res_stats['merged']} merged, {res_stats['deleted']} deleted, {res_stats['failed']} failed)",
    )
    failed_requests += ext_stats["failed"] + res_stats["failed"]

    oracle = None
    if a.reference_check:  # OpenAlex edges only: compared before the S2 edges join
        with open(a.reference_check, encoding="utf-8") as f:
            oracle = compare_reference(tables, json.load(f))

    # ---- S2 records per paper, embeddings, corpus ids
    records, via = s2_records_by_paper(tables, papers_cache["by_id"], res["records"])
    if client and "embeddings" in legs:
        s2_stats["embeddings"] = s2_api.embeddings_leg(
            client,
            sorted({r["paperId"] for r in records.values()}),
            emb_cache,
            save=lambda: write_json(p_emb, emb_cache),
        )
    for name in S2_LEGS:
        st = s2_stats.get(name)
        if st is None:
            legs_qa["s2_" + name] = _leg(
                False, None, note="offline" if a.offline else "not requested"
            )
        else:
            legs_qa["s2_" + name] = _leg(
                True,
                st,
                note=f"{st['asked']} asked, {st['calls_failed']} failed"
                + (", request budget used up" if st["budget_exhausted"] else ""),
            )
            failed_requests += st["calls_failed"]

    extend_papers(tables, records, overrides)
    unique_cids, ambiguous, all_cids = corpus_map(records)

    # ---- cites: OpenAlex pairs plus the S2 store's
    if a.s2_store:
        edges, info = read_s2_store(a.s2_store, unique_cids, all_cids)
        store_qa = {"used": True, **info}
    else:
        edges = {}
        store_qa = {
            "used": False,
            "note": "No --s2-store given: cites carry no S2 data (every edge has source openalex).",
        }
    cite_qa = build_cites(tables, edges)
    ext_qa = build_external(
        tables, citers, ext_store["works"], {r[1] for r in tables["papers"] if r[1]}
    )
    ext_qa[
        "ids_rewritten"
    ] = {  # applied to every reference list (citations, coupling, outside works)
        "merged": len(ext_store.get("merged") or {}),
        "deleted": len(ext_store.get("deleted") or []),
        "still_unknown": len(ext_store.get("unknown") or []),
    }
    build_embeddings(tables, records, emb_cache)

    violations = hard_checks(tables) + hard_checks_v2(tables)

    # ---- files
    for name, spec in TABLES.items():
        write_csv(
            os.path.join(out, f"{name}.csv"), [c for c, _ in spec["cols"]], tables[name]
        )

    def put(name, text):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="") as f:
            f.write(text)

    put("unmatched_dois.txt", "".join(d + "\n" for d in unmatched))
    put("library_corpus_ids.txt", "".join(f"{c}\n" for c in all_cids))
    put("schema.sql", schema_sql(schema))
    put("load.sql", load_sql(schema))
    put(f"apply_{schema}.sql", apply_sql(schema, a.workspace))
    put(f"swap_{schema}.sql", swap_sql(schema))
    put(f"drop_prev_{schema}.sql", drop_prev_sql(schema))

    keys = {"OPENALEX_API_KEY": oa_key, "S2_API_KEY": s2_key}
    violations += scan_for_keys([out, cache_dir], keys)
    violations += scan_for_nul(out)

    P = {c: i for i, (c, _) in enumerate(TABLES["papers"]["cols"])}
    conflicts = res["conflicts"]
    v2 = {
        "s2_coverage": {
            "papers": len(tables["papers"]),
            "found_by_doi": sum(1 for v in via.values() if v == "doi"),
            "found_by_title": sum(1 for v in via.values() if v == "title"),
            "with_tldr": sum(1 for r in tables["papers"] if r[P["tldr"]]),
            "with_embedding": len(tables["paper_embeddings"]),
        },
        "title_match": {
            "candidates": len(candidates),
            "decisions": dict(res["decisions"]),
            "dois_assigned": len(overrides),
            "doi_conflicts": len(conflicts),
            "conflict_examples": conflicts[:5],
            "accepted_examples": res["examples"],
        },
        "cites": cite_qa,
        "ambiguous_corpus_ids": {
            "count": len(ambiguous),
            "examples": [
                {"corpus_id": c, "papers": p} for c, p in sorted(ambiguous.items())[:10]
            ],
        },
        "external_works": ext_qa,
        "s2_store": store_qa,
        "legs": legs_qa,
        "key_fallback": {
            "openalex_key_set": bool(oa_key),
            "openalex_key_rejected": _STATE["oa_key_rejected"],
            "s2_key_set": bool(s2_key),
            "s2_key_rejected": bool(client and client.key_rejected),
        },
    }
    qa = build_qa(tables, aux, stats, len(dois), len(unmatched), violations, oracle, v2)
    write_json(os.path.join(out, "qa.json"), qa, indent=1)
    put("QA.md", render_qa_md(qa))
    late = [
        v for v in scan_for_keys([out], keys) if v not in violations
    ]  # the QA files themselves
    qa["hard_checks"]["late_key_scan"] = {"passed": not late, "violations": late}
    if late:
        qa["hard_checks"]["passed"] = False
        qa["hard_checks"]["violations"] = qa["hard_checks"]["violations"] + late
    write_json(os.path.join(out, "qa.json"), qa, indent=1)
    if late:
        put("QA.md", render_qa_md(qa))
    violations += late

    cov = qa["coverage"]
    print(
        f"wrote {out}: {cov['papers']} papers ({cov['papers_with_openalex_record']} with an "
        f"OpenAlex record), {qa['row_counts']['cites']} citation edges "
        f"({cite_qa['by_source']['s2']} S2-only, {cite_qa['by_source']['both']} in both), "
        f"{qa['row_counts']['coupled']} coupling edges, {len(tables['external_works'])} outside works, "
        f"{len(tables['paper_embeddings'])} embeddings, {failed_requests} failed requests"
    )
    status = 0
    if failed_requests:
        status = 1
    for v in violations:
        print(f"HARD CHECK FAILED: {v}", file=sys.stderr)
        status = 1
    if oracle is not None and not oracle["passed"]:
        print(
            f"ORACLE FAILED: Jaccard {oracle['jaccard']:.6f} is below {ORACLE_MIN_JACCARD}",
            file=sys.stderr,
        )
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
