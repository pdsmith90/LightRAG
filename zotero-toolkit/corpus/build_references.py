#!/usr/bin/env python3
"""
build_references.py — (Re)generate zotero_references.json: which library papers
each library paper cites.

Reads the DOIs in zotero_metadata.json (build_metadata.py), asks OpenAlex for
each one's work id and reference list (`referenced_works`), and keeps the
references that are themselves library papers. The output maps a citing DOI
to the DOIs it cites:

  {"10.1002/2013jb010900": {"n_refs": 45, "cited": ["10.1029/98jb02844", ...]}}

`n_refs` is the length of OpenAlex's whole reference list, `cited` the part of
it in the library. A DOI OpenAlex does not know is absent; a known one with no
library references maps to an empty `cited`. The LightRAG server reads this
file (ZOTERO_REFERENCES) for the paper-and-references graph view.

OpenAlex answers are cached (--cache), so a rerun only asks about DOIs it has
not seen. A paper's reference list rarely changes after publication; --refresh
asks about every DOI again. DOIs OpenAlex did not know are asked again on every
run (they may be indexed later).

Usage:
  python3 build_references.py --metadata zotero_metadata.json --out zotero_references.json

--out defaults to $ZOTERO_REFERENCES, else zotero_references.json beside
--metadata; --cache to openalex_references_cache.json beside --out.
OPENALEX_MAILTO (or --mailto) joins OpenAlex's polite pool; OPENALEX_API_KEY,
if set, is sent as a header (never in the URL, which reaches logs).
Exit status: 0 when every request succeeded, 1 when some failed (the output
then holds what the cache and the successful requests know).
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.openalex.org/works"
BATCH = 50  # DOIs per filter request
PAUSE = 0.2  # seconds between requests (OpenAlex allows 10/s)
TRIES = 4


def norm_doi(doi: str) -> str:
    """Lowercase, without a resolver prefix or 'doi:'; '' when not a DOI."""
    d = (doi or "").strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d if d.startswith("10.") and "/" in d else ""


def _short_id(openalex_id: str) -> str:
    return (openalex_id or "").rsplit("/", 1)[-1]


def _get(url: str, api_key: str) -> dict:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["api_key"] = api_key
    for attempt in range(TRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == TRIES - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == TRIES - 1:
                raise
        time.sleep(2**attempt * 2)
    raise RuntimeError("unreachable")


def fetch_works(dois: list[str], mailto: str, api_key: str) -> dict[str, dict]:
    """OpenAlex id and reference list for each DOI it knows, keyed by DOI."""
    params = {
        "select": "id,doi,referenced_works",
        "per-page": str(BATCH),
    }
    if mailto:
        params["mailto"] = mailto
    params["filter"] = "doi:" + "|".join(dois)
    data = _get(f"{API}?{urllib.parse.urlencode(params)}", api_key)
    found = {}
    for work in data.get("results") or []:
        doi = norm_doi(work.get("doi") or "")
        if doi:
            found[doi] = {
                "id": _short_id(work.get("id")),
                "refs": [_short_id(w) for w in work.get("referenced_works") or []],
            }
    return found


def fetch_one(doi: str, mailto: str, api_key: str) -> dict | None:
    params = {"select": "id,doi,referenced_works"}
    if mailto:
        params["mailto"] = mailto
    url = (
        f"{API}/doi:{urllib.parse.quote(doi, safe='')}?{urllib.parse.urlencode(params)}"
    )
    try:
        work = _get(url, api_key)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return {
        "id": _short_id(work.get("id")),
        "refs": [_short_id(w) for w in work.get("referenced_works") or []],
    }


def harvest(
    dois: list[str], cache: dict, mailto: str, api_key: str, refresh: bool
) -> int:
    """Fill ``cache`` (DOI -> {id, refs}) for the DOIs it lacks; return the
    number of failed requests."""
    todo = [d for d in dois if refresh or d not in cache]
    # '|' and ',' separate filter values, so such a DOI is asked about alone
    plain = [d for d in todo if "|" not in d and "," not in d]
    odd = [d for d in todo if "|" in d or "," in d]
    failed = 0
    for i in range(0, len(plain), BATCH):
        batch = plain[i : i + BATCH]
        try:
            cache.update(fetch_works(batch, mailto, api_key))
        except Exception as e:
            failed += 1
            print(f"WARN: OpenAlex batch {i // BATCH + 1} failed: {e}", file=sys.stderr)
        time.sleep(PAUSE)
    for doi in odd:
        try:
            work = fetch_one(doi, mailto, api_key)
            if work:
                cache[doi] = work
        except Exception as e:
            failed += 1
            print(f"WARN: OpenAlex lookup of {doi} failed: {e}", file=sys.stderr)
        time.sleep(PAUSE)
    return failed


def library_dois(metadata: dict) -> list[str]:
    """Distinct DOIs of the metadata's entries, in first-seen order."""
    seen = {}
    for entry in metadata.values():
        doi = norm_doi(entry.get("doi") or "") if isinstance(entry, dict) else ""
        if doi:
            seen.setdefault(doi, None)
    return list(seen)


def build(dois: list[str], cache: dict) -> dict:
    """Citing DOI -> {n_refs, cited}, keeping only references in ``dois``."""
    by_work = {}
    for doi in dois:
        work = cache.get(doi)
        if work and work.get("id"):
            by_work.setdefault(work["id"], doi)
    out = {}
    for doi in dois:
        work = cache.get(doi)
        if not work:
            continue
        refs = work.get("refs") or []
        cited = []
        for w in refs:
            ref_doi = by_work.get(w)
            if ref_doi and ref_doi != doi and ref_doi not in cited:
                cited.append(ref_doi)
        out[doi] = {"n_refs": len(refs), "cited": cited}
    return out


def _write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)  # readers see the old or the new file, never a partial one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="zotero_metadata.json")
    ap.add_argument("--out", default=os.environ.get("ZOTERO_REFERENCES"))
    ap.add_argument("--cache")
    ap.add_argument("--mailto", default=os.environ.get("OPENALEX_MAILTO", ""))
    ap.add_argument("--refresh", action="store_true", help="ask about every DOI again")
    a = ap.parse_args()
    out = a.out or os.path.join(
        os.path.dirname(os.path.abspath(a.metadata)), "zotero_references.json"
    )
    cache_path = a.cache or os.path.join(
        os.path.dirname(os.path.abspath(out)), "openalex_references_cache.json"
    )
    with open(a.metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    dois = library_dois(metadata)
    failed = harvest(
        dois, cache, a.mailto, os.environ.get("OPENALEX_API_KEY", ""), a.refresh
    )
    _write_json(cache_path, cache)
    refs = build(dois, cache)
    _write_json(out, refs)
    citing = sum(1 for v in refs.values() if v["cited"])
    edges = sum(len(v["cited"]) for v in refs.values())
    print(
        f"wrote {out}: {len(dois)} library DOIs, {len(refs)} known to OpenAlex, "
        f"{citing} cite library papers ({edges} references), {failed} failed requests"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
