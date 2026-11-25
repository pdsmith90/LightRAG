#!/usr/bin/env python3
"""
s2_api.py - Semantic Scholar Graph API client and the three legs of the
citation-graph builder: papers (metadata + TLDR), embeddings (SPECTER2) and the
title gap fill. Standard library only; no file I/O (the builder owns the caches
and hands in a ``save`` callback).

The key, if any, is read by the caller from S2_API_KEY and sent only as the
request header "x-api-key", and only to api.semanticscholar.org: never in a URL,
never to the pre-signed S3 hosts of the Datasets API. No email or other personal
detail is sent; the User-Agent is generic.

One pacer per client. The clock, the sleep and the opener can be injected, so the
tests run without sleeping and without the network.
"""

from __future__ import annotations

import datetime
import hashlib
import http.client
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

HOST = "api.semanticscholar.org"
BASE = f"https://{HOST}/graph/v1"
USER_AGENT = "zotero-citegraph/2"

PACE_KEY = 1.1  # seconds between requests with a key
PACE_NOKEY = 2.0
PACE_MAX = 10.0
PACE_SLOW = 1.5
PACE_DECAY_AFTER = 20  # successes in a row
BACKOFF0 = 4.0
BACKOFF_MAX = 120.0
RETRIES = 6
RETRY_CODES = (429, 500, 502, 503, 504)

PAPER_FIELDS = "paperId,corpusId,externalIds,title,year,citationCount,influentialCitationCount,referenceCount,tldr"
MATCH_FIELDS = (
    "paperId,corpusId,externalIds,title,year,authors,citationCount,influentialCitationCount,"
    "referenceCount,tldr"
)
EMBED_FIELDS = "paperId,embedding.specter_v2"
PAPER_BATCH = 500
EMBED_BATCH = 100  # the 10 MB response cap
EMBED_DIM = 768
RECHECK_DAYS = 30  # unknown ids are asked again after this long
SAVE_EVERY = 10  # batches between cache saves


class S2Error(Exception):
    """A request failed after its retries (the message carries no URL)."""


class NotFound(S2Error):
    """HTTP 404."""


class BudgetExhausted(Exception):
    """The --s2-max-calls budget is used up."""


def fold(s) -> str:
    """Casefold, strip accents, keep letters only (same as the builder's)."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return "".join(c for c in s if c.isalpha())


def tfold(s) -> str:
    """Casefold, strip accents, keep letters AND digits: for titles, where "Part 1" and
    "Part 2" must stay different."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return "".join(c for c in s if c.isalnum())


_SURROGATE = re.compile("[\ud800-\udfff]")


def clean_text(s: str) -> str:
    """A lone surrogate (a JSON \\ud835 escape without its pair) becomes U+FFFD, so the
    text can be written as UTF-8."""
    return s if s.isascii() else _SURROGATE.sub("\ufffd", s)


def clean_obj(o):
    """clean_text on every string of a parsed JSON answer."""
    if isinstance(o, str):
        return clean_text(o)
    if isinstance(o, list):
        return [clean_obj(x) for x in o]
    if isinstance(o, dict):
        return {
            clean_text(k) if isinstance(k, str) else k: clean_obj(v)
            for k, v in o.items()
        }
    return o


def tokens(s) -> set:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return set(re.findall(r"[^\W_]+", s))


SECRET_HEADERS = ("x-api-key", "api_key")


def build_request(
    method: str, url: str, headers: dict, body=None
) -> urllib.request.Request:
    """A Request whose key header is unredirected: urllib copies the ordinary headers to
    the target of a redirect, but not these."""
    plain = {k: v for k, v in headers.items() if k.lower() not in SECRET_HEADERS}
    req = urllib.request.Request(url, data=body, headers=plain, method=method)
    for k, v in headers.items():
        if k.lower() in SECRET_HEADERS:
            req.add_unredirected_header(k, v)
    return req


def _urlopen_json(method: str, url: str, headers: dict, body):
    """The only function that touches the network."""
    with urllib.request.urlopen(
        build_request(method, url, headers, body), timeout=60
    ) as resp:
        return json.load(resp)


def s2_headers(url: str, api_key: str, json_body: bool = False) -> dict:
    """Headers for one request. The key goes only to api.semanticscholar.org."""
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if json_body:
        headers["Content-Type"] = "application/json"
    if api_key and urllib.parse.urlsplit(url).hostname == HOST:
        headers["x-api-key"] = api_key
    return headers


class Pacer:
    """Keeps requests at least ``interval`` seconds apart; slows on trouble and
    decays back to the base after PACE_DECAY_AFTER successes in a row."""

    def __init__(self, base: float, clock, sleep):
        self.base = self.interval = base
        self.clock, self.sleep = clock, sleep
        self.last = None
        self.run = 0

    def wait(self) -> None:
        if self.last is not None:
            due = self.last + self.interval
            now = self.clock()
            if now < due:
                self.sleep(due - now)
        self.last = self.clock()

    def success(self) -> None:
        self.run += 1
        if self.run >= PACE_DECAY_AFTER:
            self.run = 0
            self.interval = max(self.base, self.interval / PACE_SLOW)

    def slow(self) -> None:
        self.run = 0
        self.interval = min(PACE_MAX, self.interval * PACE_SLOW)

    def rebase(self, base: float) -> None:
        self.base = base
        self.interval = max(self.interval, base)


class S2Client:
    def __init__(
        self,
        api_key="",
        opener=None,
        clock=None,
        sleep=None,
        max_calls=None,
        warn=None,
        retries=None,
        backoff0=None,
        backoff_max=None,
    ):
        import time

        self.key = api_key or ""
        self.key_given = bool(self.key)
        self.key_rejected = False
        self.opener = opener
        self.sleep = sleep or time.sleep
        self.pacer = Pacer(
            PACE_KEY if self.key else PACE_NOKEY, clock or time.monotonic, self.sleep
        )
        self.max_calls = max_calls
        # None = the module's value at request time (the builder's defaults)
        self.retries, self.backoff0, self.backoff_max = retries, backoff0, backoff_max
        self.calls = 0
        self.warn = warn or (lambda msg: print(msg, file=sys.stderr))

    def request(self, method: str, url: str, body=None):
        """JSON answer of one request. Raises NotFound (404), S2Error (after
        retries) or BudgetExhausted."""
        data = None if body is None else json.dumps(body).encode("utf-8")
        retries = RETRIES if self.retries is None else self.retries
        cap = BACKOFF_MAX if self.backoff_max is None else self.backoff_max
        attempt = 0
        delay = BACKOFF0 if self.backoff0 is None else self.backoff0
        while True:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExhausted()
            self.pacer.wait()
            self.calls += 1
            headers = s2_headers(url, self.key, json_body=data is not None)
            try:
                result = clean_obj(
                    (self.opener or _urlopen_json)(method, url, headers, data)
                )
                self.pacer.success()
                return result
            except urllib.error.HTTPError as e:
                code = e.code
                if code in (401, 403) and self.key:
                    self.key_rejected = True
                    self.key = ""
                    self.pacer.rebase(PACE_NOKEY)
                    self.warn(
                        f"WARNING: Semantic Scholar answered {code} to the API key; "
                        "continuing WITHOUT the key for the rest of this run"
                    )
                    continue
                if code == 404:
                    raise NotFound("HTTP 404") from None
                if code not in RETRY_CODES:
                    raise S2Error(f"HTTP {code}") from None
                reason = f"HTTP {code}"
                retry_after = _retry_after(e)
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
                http.client.HTTPException,  # IncompleteRead, RemoteDisconnected, ...
                ValueError,  # a malformed or truncated JSON body
            ) as e:
                reason = type(e).__name__
                retry_after = 0.0
            if attempt >= retries:
                raise S2Error(f"{reason} after {retries} retries")
            attempt += 1
            self.pacer.slow()
            self.sleep(min(cap, max(delay, retry_after)))
            delay = min(cap, delay * 2)

    def get(self, path: str, params: dict):
        return self.request("GET", f"{BASE}{path}?{urllib.parse.urlencode(params)}")

    def post(self, path: str, params: dict, body):
        return self.request(
            "POST", f"{BASE}{path}?{urllib.parse.urlencode(params)}", body
        )


def _retry_after(e) -> float:
    try:
        return float(e.headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------
# caches (plain dicts; the builder loads and saves them)
# --------------------------------------------------------------------------


def new_papers_cache() -> dict:
    return {"by_id": {}, "fetched": {}}


def _age_days(iso: str, today: str) -> float:
    try:
        return (
            datetime.date.fromisoformat(today) - datetime.date.fromisoformat(iso)
        ).days
    except (TypeError, ValueError):
        return float("inf")


def _leg_stats() -> dict:
    return {"complete": True, "asked": 0, "calls_failed": 0, "budget_exhausted": False}


# --------------------------------------------------------------------------
# papers leg: S2 records and TLDRs by DOI
# --------------------------------------------------------------------------


def papers_leg(client, ids, cache, today, refresh=False, save=None) -> dict:
    """POST /paper/batch for the ids ("DOI:<doi>" or "CorpusId:<n>") that the cache
    lacks (null answers are asked again after RECHECK_DAYS). Returns the leg's stats."""
    st = _leg_stats()
    by_id, fetched = cache["by_id"], cache["fetched"]
    todo, seen = [], set()
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        if (
            refresh
            or i not in by_id
            or (by_id[i] is None and _age_days(fetched.get(i), today) > RECHECK_DAYS)
        ):
            todo.append(i)
    n_batches = 0
    try:
        for k in range(0, len(todo), PAPER_BATCH):
            batch = todo[k : k + PAPER_BATCH]
            try:
                resp = client.post(
                    "/paper/batch", {"fields": PAPER_FIELDS}, {"ids": batch}
                )
            except BudgetExhausted:
                st["budget_exhausted"], st["complete"] = True, False
                break
            except S2Error as e:
                _fail(st, f"papers batch {k // PAPER_BATCH + 1}: {e}")
                continue
            if not isinstance(resp, list) or len(resp) != len(batch):
                _fail(
                    st,
                    f"papers batch {k // PAPER_BATCH + 1}: answer not aligned with the request",
                )
                continue
            for i, rec in zip(batch, resp):
                by_id[i] = rec if isinstance(rec, dict) else None
                fetched[i] = today
            st["asked"] += len(batch)
            n_batches += 1
            if save and n_batches % SAVE_EVERY == 0:
                save()
    finally:
        if save:
            save()
    return st


def _fail(st: dict, msg: str) -> None:
    st["complete"] = False
    st["calls_failed"] += 1
    print(f"WARN: Semantic Scholar {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# embeddings leg: SPECTER2 vectors by S2 paper id
# --------------------------------------------------------------------------


def embeddings_leg(client, paper_ids, cache, save=None) -> dict:
    """Fetch SPECTER2 vectors for S2 paper ids the cache lacks. ``cache`` is
    {paperId: {"model", "vector"}}. A paper with no vector, or one that is not
    768-dimensional, is not cached (it is asked again on the next run)."""
    st = _leg_stats()
    st["no_vector"] = st["bad_dimension"] = 0
    todo = sorted({p for p in paper_ids if p and p not in cache})
    n_batches = 0
    try:
        for k in range(0, len(todo), EMBED_BATCH):
            batch = todo[k : k + EMBED_BATCH]
            try:
                resp = client.post(
                    "/paper/batch", {"fields": EMBED_FIELDS}, {"ids": batch}
                )
            except BudgetExhausted:
                st["budget_exhausted"], st["complete"] = True, False
                break
            except S2Error as e:
                _fail(st, f"embeddings batch {k // EMBED_BATCH + 1}: {e}")
                continue
            if not isinstance(resp, list) or len(resp) != len(batch):
                _fail(
                    st,
                    f"embeddings batch {k // EMBED_BATCH + 1}: answer not aligned with the request",
                )
                continue
            for pid, rec in zip(batch, resp):
                emb = (rec or {}).get("embedding") if isinstance(rec, dict) else None
                vec = (emb or {}).get("vector")
                if not vec:
                    st["no_vector"] += 1
                elif len(vec) != EMBED_DIM:
                    st["bad_dimension"] += 1
                else:
                    cache[pid] = {
                        "model": emb.get("model") or "specter_v2",
                        "vector": [round(x, 6) for x in vec],
                    }
            st["asked"] += len(batch)
            n_batches += 1
            if save and n_batches % SAVE_EVERY == 0:
                save()
    finally:
        if save:
            save()
    return st


# --------------------------------------------------------------------------
# title gap fill: S2 title match for papers without a usable DOI
# --------------------------------------------------------------------------


def year_range(year):
    return f"{year - 1}-{year + 1}" if isinstance(year, int) else ""


def match_key(paper_id: str, title: str, year) -> str:
    """Cache key: the paper plus a hash of the query, so a changed title is asked again."""
    q = f"{title.strip()}\n{year_range(year)}"
    return f"{paper_id}#{hashlib.sha1(q.encode('utf-8')).hexdigest()[:12]}"


# Words an edition suffix may consist of ("..., 2nd Edition, revised"). A volume or a part is a
# different work with its own DOI, so vol, part, bare numbers, roman numerals and years are not here.
_EDITION_WORD = re.compile(
    r"^(edition|editions|ed|revised|rev|expanded|updated|first|second|third|fourth|fifth|sixth|seventh"
    r"|eighth|ninth|tenth|\d+(st|nd|rd|th))$"
)


def _word_seq(s) -> list[str]:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return re.findall(r"[^\W_]+", s)


def edition_suffix_only(a: str, b: str, min_words: int = 3) -> bool:
    """True when one title is the other (at least ``min_words`` words) plus only edition
    words: "Numerical Recipes in C" vs "Numerical Recipes in C, 2nd Edition"."""
    x, y = _word_seq(a), _word_seq(b)
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    if (
        len(short) < min_words
        or len(long_) == len(short)
        or long_[: len(short)] != short
    ):
        return False
    return all(_EDITION_WORD.match(w) for w in long_[len(short) :])


def decide_match(cand: dict, rec) -> str:
    """'accepted', or 'reject:none|title|year|author'. ``cand`` has title, year
    (int or None) and first_family (str or ''); ``rec`` is the S2 match record.
    Titles agree when they fold equal (letters and digits), share >= 0.9 of their words,
    or differ only by an edition suffix (at least 3 words in the shorter title, or 2 when
    the Zotero entry has a year and a first creator and both checks pass); year and
    author still decide."""
    if not rec:
        return "reject:none"
    zt, st_ = cand.get("title") or "", rec.get("title") or ""
    zy, sy = cand.get("year"), rec.get("year")
    fam = fold(cand.get("first_family") or "")
    short_suffix = False  # the title passed only through the 2-word edition rule
    if tfold(zt) != tfold(st_) and not edition_suffix_only(zt, st_):
        a, b = tokens(zt), tokens(st_)
        if a and b and len(a & b) / len(a | b) >= 0.9:
            pass
        elif isinstance(zy, int) and fam and edition_suffix_only(zt, st_, 2):
            short_suffix = True
        else:
            return "reject:title"
    if isinstance(zy, int) and not (isinstance(sy, int) and abs(zy - sy) <= 1):
        return "reject:title" if short_suffix else "reject:year"
    if fam:
        lasts = set()
        for a in rec.get("authors") or []:
            parts = ((a or {}).get("name") or "").split()
            if parts:
                lasts.add(fold(parts[-1]))
        if fam not in lasts:
            return "reject:title" if short_suffix else "reject:author"
    return "accepted"


def match_leg(client, candidates, matches, papers_cache, today, save=None) -> dict:
    """GET /paper/search/match for each candidate not yet in ``matches``.

    ``candidates``: [{"paper_id", "title", "year", "first_family"}]. ``matches``:
    {key: {"paper_id", "query", "year_range", "asked", "decision", "record"}}.
    An accepted record is also kept in ``papers_cache`` under "CorpusId:<n>"."""
    st = _leg_stats()
    n = 0
    try:
        for c in sorted(candidates, key=lambda c: c["paper_id"]):
            title = (c.get("title") or "").strip()
            if not tfold(title):
                continue
            key = match_key(c["paper_id"], title, c.get("year"))
            if key in matches:
                continue
            params = {"query": title, "fields": MATCH_FIELDS}
            if year_range(c.get("year")):
                params["year"] = year_range(c["year"])
            try:
                resp = client.get("/paper/search/match", params)
                rec = ((resp or {}).get("data") or [None])[0]
            except NotFound:
                rec = None
            except BudgetExhausted:
                st["budget_exhausted"], st["complete"] = True, False
                break
            except S2Error as e:
                _fail(st, f"title match of {c['paper_id']}: {e}")
                continue
            decision = decide_match(c, rec)
            matches[key] = {
                "paper_id": c["paper_id"],
                "query": title,
                "year_range": year_range(c.get("year")),
                "asked": today,
                "decision": decision,
                "record": rec,
            }
            if decision == "accepted" and rec.get("corpusId") is not None:
                cid = f"CorpusId:{rec['corpusId']}"
                papers_cache["by_id"][cid] = rec
                papers_cache["fetched"][cid] = today
            st["asked"] += 1
            n += 1
            if save and n % (SAVE_EVERY * 10) == 0:
                save()
    finally:
        if save:
            save()
    return st
