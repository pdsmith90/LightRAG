"""Add an uploaded PDF to a Zotero library instead of indexing it directly.

With ``ZOTERO_UPLOAD=true`` the WebUI upload (``POST /documents/upload``) calls
:func:`add_pdf_to_zotero`: the PDF becomes a child attachment of a Zotero item,
created through the Zotero Web API, with the file bytes stored on a WebDAV file
store (the layout the Zotero desktop client uses for WebDAV syncing). The
document reaches LightRAG later, through the corpus builder, under its Zotero
key; nothing is written to the input directory. A configured
``ZOTERO_SYNC_COMMAND`` can then be launched detached to run that sync and build
(:func:`trigger_sync`).

Order of operations, with the failure each step reports (never silently):

1. md5 of the file against the upload ledger and, if configured, the local
   ``zotero.sqlite`` (``ZOTERO_SQLITE``): a known file is refused (409).
2. DOI (PDF metadata, first pages) and a title query are looked up on CrossRef;
   a candidate is accepted only when its title is found in the PDF's own text.
   CrossRef trouble is not an error: the item becomes a titled stub.
3. A DOI already in the library is refused (409) when the item holds a PDF, and
   gets the PDF attached when it does not; no second item is created.
4. Parent item (CrossRef metadata or stub), attachment row, WebDAV
   ``<KEY>.zip`` + ``<KEY>.prop``, then md5/mtime onto the attachment row. A
   failure from the attachment row on deletes what this call created (502) and
   says whether that cleanup worked.

All settings come from the environment (see :func:`load_config`); the module
holds no host, path or credential. Python 3.10+, httpx only; ``pypdf`` is
imported when a PDF is read.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import httpx

from lightrag.utils import logger

ZOTERO_API = "https://api.zotero.org"
CROSSREF_API = "https://api.crossref.org"
UPLOAD_EXTENSIONS = (".pdf",)

_sleep = time.sleep
_TRUE = {"1", "true", "yes", "on"}
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>{}|\\^`]+", re.IGNORECASE)
_JUNK_TITLE_RE = re.compile(
    r"^(microsoft word|untitled|slide \d|\s*$)|\.(docx?|pdf|tex|indd|pptx?)$",
    re.IGNORECASE,
)
# how much of a title must sit on page 0 for the record to be this paper's
_TITLE_ON_PAGE0 = 0.85
_CROSSREF_FIELDS = (
    "DOI,title,author,editor,type,issued,container-title,short-container-title,"
    "volume,issue,page,publisher,publisher-location,ISSN,ISBN,URL,abstract,event"
)

_CROSSREF_TYPES = {
    "journal-article": "journalArticle",
    "book": "book",
    "monograph": "book",
    "edited-book": "book",
    "reference-book": "book",
    "book-chapter": "bookSection",
    "book-section": "bookSection",
    "book-part": "bookSection",
    "proceedings-article": "conferencePaper",
    "posted-content": "preprint",
    "report": "report",
    "dissertation": "thesis",
    "dataset": "dataset",
}

# item field -> record reader, applied only to fields the item template has
_FIELD_READERS: dict[str, Any] = {
    "title": lambda r: _first(r.get("title")),
    "abstractNote": lambda r: _strip_tags(r.get("abstract")),
    "date": lambda r: _date(r),
    "language": lambda r: r.get("language"),
    "url": lambda r: r.get("URL"),
    "publicationTitle": lambda r: _first(r.get("container-title")),
    "journalAbbreviation": lambda r: _first(r.get("short-container-title")),
    "bookTitle": lambda r: _first(r.get("container-title")),
    "proceedingsTitle": lambda r: _first(r.get("container-title")),
    "conferenceName": lambda r: (r.get("event") or {}).get("name"),
    "volume": lambda r: r.get("volume"),
    "issue": lambda r: r.get("issue"),
    "pages": lambda r: r.get("page"),
    "publisher": lambda r: r.get("publisher"),
    "institution": lambda r: r.get("publisher"),
    "university": lambda r: r.get("publisher"),
    "repository": lambda r: r.get("publisher"),
    "place": lambda r: r.get("publisher-location"),
    "ISSN": lambda r: _first(r.get("ISSN")),
    "ISBN": lambda r: _first(r.get("ISBN")),
    "edition": lambda r: r.get("edition-number"),
}


class ZoteroUploadError(Exception):
    """A refused or failed upload; ``status_code`` is the HTTP status to answer."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ZoteroUploadConfig:
    user_id: str
    library_type: str
    api_key: str = field(repr=False)
    webdav_url: str
    webdav_user: str
    webdav_pass: str = field(repr=False)
    crossref_mailto: str = ""
    collection: str = ""
    tag: str = "lightrag-upload"
    ledger_path: str = "zotero_upload_ledger.json"
    sqlite_path: str = ""
    sync_command: str = ""
    sync_lock_file: str = ""
    sync_log: str = ""

    @property
    def library(self) -> str:
        return f"{ZOTERO_API}/{self.library_type}s/{self.user_id}"


@dataclass
class UploadResult:
    outcome: str  # "created" (new item) | "attached" (to an item already there)
    item_key: str
    attachment_key: str
    title: str
    doi: str
    metadata: str  # "crossref" | "stub" | "existing"
    md5: str
    filename: str
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        where = (
            "attached to the existing Zotero item"
            if self.outcome == "attached"
            else "added to the Zotero library as item"
        )
        how = {
            "crossref": "metadata from CrossRef",
            "stub": "no CrossRef match, titled stub; edit its metadata in Zotero",
            "existing": "metadata already in the library",
        }[self.metadata]
        doi = f", DOI {self.doi}" if self.doi else ""
        text = f"'{self.filename}' {where} {self.item_key} ({how}{doi})."
        return " ".join([text, *self.warnings]) if self.warnings else text


@dataclass
class SyncOutcome:
    triggered: bool
    state: str  # "started" | "already_running" | "not_configured" | "failed"
    message: str


def upload_enabled(env: dict[str, str] | None = None) -> bool:
    """True when ``ZOTERO_UPLOAD`` is switched on; reads no file."""
    value = (env if env is not None else os.environ).get("ZOTERO_UPLOAD", "")
    return value.strip().lower() in _TRUE


def _read_secret_file(path: str) -> str:
    return Path(os.path.expanduser(path)).read_text(encoding="utf-8").strip()


def _read_kv_file(path: str) -> dict[str, str]:
    conf: dict[str, str] = {}
    for line in Path(os.path.expanduser(path)).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            conf[k.strip()] = v.strip()
    return conf


def load_config(env: dict[str, str] | None = None) -> ZoteroUploadConfig:
    """Build the configuration from the environment; 503 when incomplete.

    Required: ``ZOTERO_USER_ID``; the API key from ``ZOTERO_API_KEY`` or the file
    named by ``ZOTERO_API_KEY_FILE``; the WebDAV settings from
    ``ZOTERO_WEBDAV_URL`` / ``ZOTERO_WEBDAV_USER`` / ``ZOTERO_WEBDAV_PASS``
    and/or a ``KEY=VALUE`` file (``WEBDAV_URL``, ``WEBDAV_USER``, ``WEBDAV_PASS``,
    optional ``WEBDAV_ENABLED``) named by ``ZOTERO_WEBDAV_FILE``. Optional:
    ``ZOTERO_LIBRARY_TYPE`` (user|group), ``ZOTERO_CROSSREF_MAILTO``,
    ``ZOTERO_UPLOAD_COLLECTION``, ``ZOTERO_UPLOAD_TAG``, ``ZOTERO_UPLOAD_LEDGER``,
    ``ZOTERO_SQLITE``, ``ZOTERO_SYNC_COMMAND``, ``ZOTERO_SYNC_LOCK_FILE``,
    ``ZOTERO_SYNC_LOG``. Files are read on every call, so a rotated key needs no
    restart.
    """
    env = dict(os.environ if env is None else env)
    problems: list[str] = []

    def secret(var: str, file_var: str) -> str:
        if env.get(var):
            return env[var].strip()
        if env.get(file_var):
            try:
                return _read_secret_file(env[file_var])
            except OSError as e:
                problems.append(f"{file_var} unreadable ({e.strerror})")
                return ""
        problems.append(f"{var} or {file_var}")
        return ""

    user_id = env.get("ZOTERO_USER_ID", "").strip()
    if not user_id:
        problems.append("ZOTERO_USER_ID")
    api_key = secret("ZOTERO_API_KEY", "ZOTERO_API_KEY_FILE")

    kv: dict[str, str] = {}
    if env.get("ZOTERO_WEBDAV_FILE"):
        try:
            kv = _read_kv_file(env["ZOTERO_WEBDAV_FILE"])
        except OSError as e:
            problems.append(f"ZOTERO_WEBDAV_FILE unreadable ({e.strerror})")
    if kv.get("WEBDAV_ENABLED", "1").strip() != "1":
        raise ZoteroUploadError(
            "Zotero upload is enabled but WEBDAV_ENABLED is not 1 in "
            "ZOTERO_WEBDAV_FILE; this module only stores files on WebDAV.",
            503,
        )
    webdav_url = env.get("ZOTERO_WEBDAV_URL") or kv.get("WEBDAV_URL", "")
    webdav_user = env.get("ZOTERO_WEBDAV_USER") or kv.get("WEBDAV_USER", "")
    webdav_pass = env.get("ZOTERO_WEBDAV_PASS") or kv.get("WEBDAV_PASS", "")
    for name, value in (
        ("ZOTERO_WEBDAV_URL", webdav_url),
        ("ZOTERO_WEBDAV_USER", webdav_user),
        ("ZOTERO_WEBDAV_PASS", webdav_pass),
    ):
        if not value:
            problems.append(f"{name} (or the matching key in ZOTERO_WEBDAV_FILE)")

    library_type = env.get("ZOTERO_LIBRARY_TYPE", "user").strip().lower() or "user"
    if library_type not in ("user", "group"):
        problems.append("ZOTERO_LIBRARY_TYPE must be user or group")
    if problems:
        raise ZoteroUploadError(
            "Zotero upload is enabled (ZOTERO_UPLOAD) but not configured: "
            + "; ".join(problems),
            503,
        )
    return ZoteroUploadConfig(
        user_id=user_id,
        library_type=library_type,
        api_key=api_key,
        webdav_url=webdav_url,
        webdav_user=webdav_user,
        webdav_pass=webdav_pass,
        crossref_mailto=env.get("ZOTERO_CROSSREF_MAILTO", "").strip(),
        collection=env.get("ZOTERO_UPLOAD_COLLECTION", "").strip(),
        tag=env.get("ZOTERO_UPLOAD_TAG", "lightrag-upload").strip(),
        ledger_path=os.path.expanduser(
            env.get("ZOTERO_UPLOAD_LEDGER", "zotero_upload_ledger.json")
        ),
        sqlite_path=os.path.expanduser(env.get("ZOTERO_SQLITE", "")),
        sync_command=env.get("ZOTERO_SYNC_COMMAND", "").strip(),
        sync_lock_file=os.path.expanduser(env.get("ZOTERO_SYNC_LOCK_FILE", "")),
        sync_log=os.path.expanduser(env.get("ZOTERO_SYNC_LOG", "")),
    )


# ---------------------------------------------------------------- small helpers


def _first(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip() if value else ""


def _strip_tags(value: Any) -> str:
    return re.sub(r"<[^>]+>", "", str(value)).strip() if value else ""


def _date(rec: dict) -> str:
    for key in ("issued", "published", "published-print", "published-online"):
        parts = ((rec.get(key) or {}).get("date-parts") or [[]])[0]
        parts = [p for p in parts if p]
        if parts:
            return "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts))
    return ""


def normalize_doi(value: str | None) -> str:
    """Lower-case bare DOI, or ``""``."""
    if not value:
        return ""
    value = re.sub(r"^\s*(https?://(dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I)
    m = _DOI_RE.search(value)
    if not m:
        return ""
    doi = m.group(0).rstrip(".,;:]}>\"'")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1]
    return doi.lower()


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", _fold(text))


def title_coverage(title: str, text: str) -> float:
    """Share of the title's words (3+ characters) that occur in ``text``."""
    words = _words(title)
    if not words:
        return 0.0
    have = set(_words(text))
    return sum(w in have for w in words) / len(words)


def clean_filename(name: str) -> str:
    """Basename without control characters or separators; never empty."""
    name = os.path.basename((name or "").replace("\\", "/"))
    name = "".join(c for c in name if c.isprintable()).strip().strip(".")
    return name[:180] or "upload.pdf"


def _md5(path: str) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------- ledger


@contextmanager
def _ledger_lock(cfg: ZoteroUploadConfig) -> Iterator[None]:
    """One upload at a time across workers: the duplicate check and the write."""
    try:
        fd = os.open(cfg.ledger_path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as e:
        raise ZoteroUploadError(
            f"The upload ledger location is not writable ({e.strerror}); "
            "set ZOTERO_UPLOAD_LEDGER.",
            503,
        ) from e
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _ledger_read(cfg: ZoteroUploadConfig) -> dict:
    try:
        with open(cfg.ledger_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    data.setdefault("by_md5", {})
    data.setdefault("by_doi", {})
    return data


def _ledger_write(cfg: ZoteroUploadConfig, data: dict) -> None:
    tmp = f"{cfg.ledger_path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, cfg.ledger_path)


def _sqlite_has_md5(cfg: ZoteroUploadConfig, md5: str) -> str:
    """Key of a live attachment with this storage hash in zotero.sqlite, else ''."""
    if not cfg.sqlite_path:
        return ""
    try:
        con = sqlite3.connect(
            Path(cfg.sqlite_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1
        )
        try:
            row = con.execute(
                "SELECT i.key FROM itemAttachments ia JOIN items i "
                "ON i.itemID = ia.itemID WHERE ia.storageHash = ? AND ia.itemID "
                "NOT IN (SELECT itemID FROM deletedItems) LIMIT 1",
                (md5,),
            ).fetchone()
        finally:
            con.close()
    except sqlite3.Error as e:
        logger.warning(
            f"zotero_add: ZOTERO_SQLITE not readable ({e}); md5 check skipped"
        )
        return ""
    return row[0] if row else ""


# -------------------------------------------------------------- PDF inspection


@dataclass
class PdfInfo:
    pages: list[str] = field(default_factory=list)
    meta_title: str = ""
    meta_dois: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)


def read_pdf_info(path: str, max_pages: int = 3) -> PdfInfo:
    """First pages' text and metadata; an unreadable PDF gives an empty result."""
    info = PdfInfo()
    try:
        from pypdf import PdfReader

        reader = PdfReader(path, strict=False)
        if reader.is_encrypted:
            reader.decrypt("")
        meta = reader.metadata or {}
        info.meta_title = str(meta.get("/Title") or "").strip()
        blob = " ".join(str(v) for v in dict(meta).values())
        info.meta_dois = [d for d in map(normalize_doi, _DOI_RE.findall(blob)) if d]
        for page in reader.pages[:max_pages]:
            try:
                info.pages.append(page.extract_text() or "")
            except Exception:
                info.pages.append("")
    except Exception as e:
        logger.warning(f"zotero_add: cannot read PDF text ({type(e).__name__}: {e})")
    return info


def doi_candidates(info: PdfInfo, limit: int = 4) -> list[str]:
    """Metadata DOIs, then DOIs labelled ``doi``, then the most frequent others."""
    out: list[str] = list(dict.fromkeys(info.meta_dois))
    labelled: list[str] = []
    counts: dict[str, int] = {}
    for m in _DOI_RE.finditer(info.text):
        doi = normalize_doi(m.group(0))
        if not doi:
            continue
        counts[doi] = counts.get(doi, 0) + 1
        if re.search(r"doi", info.text[max(0, m.start() - 24) : m.start()], re.I):
            labelled.append(doi)
    out += labelled
    out += sorted(counts, key=lambda d: -counts[d])
    return list(dict.fromkeys(out))[:limit]


def _title_queries(info: PdfInfo) -> list[str]:
    queries = []
    if len(info.meta_title) >= 12 and not _JUNK_TITLE_RE.search(info.meta_title):
        queries.append(info.meta_title)
    head = re.sub(r"\s+", " ", info.pages[0] if info.pages else "").strip()[:300]
    if len(head) >= 40:
        queries.append(head)
    return queries


# --------------------------------------------------------------------- CrossRef


class _CrossrefDown(Exception):
    pass


def _crossref_get(http: httpx.Client, cfg: ZoteroUploadConfig, path: str, params=None):
    agent = "lightrag-zotero-upload/1"
    if cfg.crossref_mailto:
        agent += f" (mailto:{cfg.crossref_mailto})"
    try:
        r = http.get(
            CROSSREF_API + path,
            params=params,
            headers={"User-Agent": agent},
            timeout=20,
        )
    except httpx.HTTPError as e:
        raise _CrossrefDown(type(e).__name__) from e
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise _CrossrefDown(f"HTTP {r.status_code}")
    return r.json().get("message")


def _title_on_first_page(title: str, info: PdfInfo) -> bool:
    """The title's words are on page 0 — where a paper's own title lives.

    Pages 1-2 can hold a reference list, so they must not count: a cited work's
    title there does not make the record the paper being uploaded.
    """
    return title_coverage(title, info.pages[0] if info.pages else "") >= _TITLE_ON_PAGE0


def _verified(rec: dict, info: PdfInfo, by_search: bool) -> bool:
    title = _first(rec.get("title"))
    words = _words(title)
    if by_search:
        if len(words) < 4 or not _title_on_first_page(title, info):
            return False
        names = [a.get("family", "") for a in rec.get("author", []) if a.get("family")]
        return not names or _fold(names[0]) in _fold(info.text)
    if not words:  # no title to check the PDF against, so nothing is verified
        return False
    return _title_on_first_page(title, info)


def lookup_crossref(
    http: httpx.Client, cfg: ZoteroUploadConfig, info: PdfInfo
) -> tuple[dict | None, str, list[str]]:
    """``(record, unverified DOI candidate, warnings)``; never raises."""
    candidates = doi_candidates(info)
    try:
        for doi in candidates:
            rec = _crossref_get(http, cfg, "/works/" + quote(doi, safe="/()"))
            if rec and _verified(rec, info, by_search=False):
                return rec, "", []
        for query in _title_queries(info):
            found = _crossref_get(
                http,
                cfg,
                "/works",
                {"query.bibliographic": query, "rows": 5, "select": _CROSSREF_FIELDS},
            )
            for rec in (found or {}).get("items", []):
                if _verified(rec, info, by_search=True):
                    return rec, "", []
    except _CrossrefDown as e:
        return (
            None,
            candidates[0] if candidates else "",
            [f"CrossRef was unreachable ({e}), so no metadata was looked up."],
        )
    return None, candidates[0] if candidates else "", []


# ------------------------------------------------------------------ Zotero API


def _retry_after(value: str | None, default: float = 2.0) -> float:
    """Seconds from a ``Retry-After`` header; an HTTP-date value uses ``default``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class _Zotero:
    def __init__(self, cfg: ZoteroUploadConfig, http: httpx.Client):
        self.cfg, self.http = cfg, http
        self.headers = {"Zotero-API-Key": cfg.api_key, "Zotero-API-Version": "3"}

    def request(self, method: str, url: str, *, headers=None, **kw) -> httpx.Response:
        hdrs = {**self.headers, **(headers or {})}
        for attempt in (1, 2, 3):
            try:
                r = self.http.request(method, url, headers=hdrs, timeout=60, **kw)
            except httpx.HTTPError as e:
                if attempt == 3:
                    raise ZoteroUploadError(
                        f"Zotero API unreachable ({type(e).__name__})"
                    ) from e
                _sleep(2)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                _sleep(min(_retry_after(r.headers.get("Retry-After")), 20))
                continue
            return r
        raise AssertionError("unreachable")

    def template(self, item_type: str) -> dict | None:
        r = self.request(
            "GET", f"{ZOTERO_API}/items/new", params={"itemType": item_type}
        )
        return r.json() if r.status_code == 200 else None

    def create(self, item: dict) -> tuple[str, int]:
        r = self.request(
            "POST",
            f"{self.cfg.library}/items",
            json=[item],
            headers={"Zotero-Write-Token": uuid.uuid4().hex},
        )
        body = r.json() if r.status_code == 200 else {}
        if r.status_code != 200 or body.get("failed") or not body.get("successful"):
            detail = body.get("failed") or r.text[:200]
            raise ZoteroUploadError(
                f"Zotero refused the item ({r.status_code}): {detail}"
            )
        done = body["successful"]["0"]
        return done["key"], int(done.get("version") or 0)

    def get_item(self, key: str) -> dict | None:
        r = self.request("GET", f"{self.cfg.library}/items/{key}")
        return r.json() if r.status_code == 200 else None

    def delete(self, key: str) -> None:
        item = self.get_item(key)
        if item is None:
            return
        r = self.request(
            "DELETE",
            f"{self.cfg.library}/items/{key}",
            headers={"If-Unmodified-Since-Version": str(item["version"])},
        )
        if r.status_code not in (200, 204, 404):
            raise ZoteroUploadError(f"delete {key} -> {r.status_code}")

    def set_file_metadata(self, key: str, mtime_ms: int, md5: str) -> None:
        item = self.get_item(key)
        if item is None:
            raise ZoteroUploadError(f"attachment {key} not found after creation")
        r = self.request(
            "PATCH",
            f"{self.cfg.library}/items/{key}",
            json={"mtime": mtime_ms, "md5": md5},
            headers={"If-Unmodified-Since-Version": str(item["version"])},
        )
        if r.status_code not in (200, 204):
            raise ZoteroUploadError(f"setting md5/mtime on {key} -> {r.status_code}")

    def find_by_doi(self, doi: str, title: str) -> dict | None:
        """Top-level item with this DOI. The API cannot search the DOI field, so
        it searches the title and filters the hits on the DOI."""
        if not title:
            return None
        r = self.request(
            "GET",
            f"{self.cfg.library}/items/top",
            params={"q": title[:200], "limit": 25, "format": "json"},
        )
        if r.status_code != 200:
            raise ZoteroUploadError(f"title search -> {r.status_code}")
        for hit in r.json():
            if normalize_doi(hit["data"].get("DOI")) == doi:
                return hit
        return None

    def pdf_children(self, key: str) -> list[dict]:
        r = self.request("GET", f"{self.cfg.library}/items/{key}/children")
        if r.status_code != 200:
            raise ZoteroUploadError(f"children of {key} -> {r.status_code}")
        return [
            c["data"]
            for c in r.json()
            if c["data"].get("itemType") == "attachment"
            and c["data"].get("contentType") == "application/pdf"
            and not c["data"].get("deleted")
        ]


# ----------------------------------------------------------------------- WebDAV


class _WebDAV:
    """The ``<KEY>.zip`` + ``<KEY>.prop`` layout of the Zotero client's WebDAV storage.

    ``.prop`` is removed first and written last, so the store never holds a
    ``.prop`` without a matching ``.zip`` (the client deletes such a ``.prop``).
    """

    def __init__(self, cfg: ZoteroUploadConfig, http: httpx.Client):
        self.http = http
        self.base = cfg.webdav_url.rstrip("/") + "/zotero/"
        self.auth = (cfg.webdav_user, cfg.webdav_pass)

    def _send(self, method: str, name: str, **kw) -> httpx.Response:
        try:
            return self.http.request(
                method,
                self.base + name if name else self.base,
                auth=self.auth,
                timeout=180,
                **kw,
            )
        except httpx.HTTPError as e:
            raise ZoteroUploadError(f"WebDAV unreachable ({type(e).__name__})") from e

    def _put(self, name: str, content, content_type: str, length: int) -> None:
        r = self._send(
            "PUT",
            name,
            content=content,
            headers={"Content-Type": content_type, "Content-Length": str(length)},
        )
        if r.status_code == 507:
            raise ZoteroUploadError("WebDAV store is full (507)")
        if r.status_code not in (200, 201, 204):
            raise ZoteroUploadError(f"WebDAV PUT {name} -> {r.status_code}")

    def delete(self, name: str) -> None:
        r = self._send("DELETE", name)
        if r.status_code not in (200, 204, 404):
            raise ZoteroUploadError(f"WebDAV DELETE {name} -> {r.status_code}")

    def put_attachment(self, key: str, pdf_path: str, arcname: str, md5: str) -> int:
        """Store one file; returns its mtime in integer milliseconds."""
        r = self._send("MKCOL", "")
        if r.status_code not in (201, 301, 405):
            raise ZoteroUploadError(f"WebDAV MKCOL -> {r.status_code}")
        mtime_ms = int(time.time() * 1000)
        prop = (
            f'<properties version="1"><mtime>{mtime_ms}</mtime>'
            f"<hash>{md5}</hash></properties>"
        ).encode("ascii")
        with tempfile.TemporaryDirectory(prefix="zotero_zip_") as tmp:
            zpath = os.path.join(tmp, key + ".zip")
            with zipfile.ZipFile(
                zpath, "w", zipfile.ZIP_DEFLATED, allowZip64=False
            ) as zf:
                info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(pdf_path, "rb") as fh:
                    zf.writestr(info, fh.read())
            self.delete(key + ".prop")
            with open(zpath, "rb") as fh:
                self._put(
                    key + ".zip",
                    iter(lambda: fh.read(1 << 20), b""),
                    "application/zip",
                    os.path.getsize(zpath),
                )
        self._put(key + ".prop", prop, "text/xml", len(prop))
        return mtime_ms

    def remove_attachment(self, key: str) -> None:
        self.delete(key + ".prop")
        self.delete(key + ".zip")


# ------------------------------------------------------------------- item build


def _creators(rec: dict, item_type: str) -> list[dict]:
    editors_ok = item_type in (
        "book",
        "bookSection",
        "conferencePaper",
        "journalArticle",
    )
    out = []
    for role, key in (("author", "author"), ("editor", "editor")):
        for person in rec.get(key) or []:
            kind = role if role == "author" or editors_ok else "contributor"
            if person.get("family"):
                out.append(
                    {
                        "creatorType": kind,
                        "lastName": person["family"],
                        "firstName": person.get("given", ""),
                    }
                )
            elif person.get("name"):
                out.append({"creatorType": kind, "name": person["name"]})
    return out


def _fill(item: dict, extra: list[str], cfg: ZoteroUploadConfig) -> dict:
    if cfg.tag:
        item["tags"] = [{"tag": cfg.tag}]
    if cfg.collection and "collections" in item:
        item["collections"] = [cfg.collection]
    item["extra"] = "\n".join(extra)
    return item


def build_crossref_item(zot: _Zotero, rec: dict) -> dict:
    item_type = _CROSSREF_TYPES.get(rec.get("type", ""), "document")
    template = zot.template(item_type)
    if template is None and item_type != "document":
        item_type, template = "document", zot.template("document")
    if template is None:
        raise ZoteroUploadError("Zotero API gave no item template")
    for name, read in _FIELD_READERS.items():
        value = read(rec)
        if name in template and value:
            template[name] = value
    doi = normalize_doi(rec.get("DOI"))
    extra = [f"Added via LightRAG upload on {time.strftime('%Y-%m-%d')}"]
    if "DOI" in template:
        template["DOI"] = doi
    elif doi:
        extra.append(f"DOI: {doi}")
    if "creators" in template:
        template["creators"] = _creators(rec, item_type)
    return _fill(template, extra, zot.cfg)


def build_stub_item(zot: _Zotero, filename: str, candidate_doi: str) -> dict:
    template = zot.template("document")
    if template is None:
        raise ZoteroUploadError("Zotero API gave no item template")
    template["title"] = f"Uploaded: {filename}"
    extra = [f"Added via LightRAG upload on {time.strftime('%Y-%m-%d')}"]
    if candidate_doi:
        extra.append(f"Candidate DOI (unverified): {candidate_doi}")
    item = _fill(template, extra, zot.cfg)
    item["tags"].append({"tag": "unmatched-upload"})
    return item


def _attachment_item(parent: str, filename: str) -> dict:
    return {
        "itemType": "attachment",
        "parentItem": parent,
        "linkMode": "imported_file",
        "title": filename,
        "accessDate": "",
        "note": "",
        "tags": [],
        "relations": {},
        "contentType": "application/pdf",
        "charset": "",
        "filename": filename,
    }


# ------------------------------------------------------------------- main entry


def _refuse(reason: str) -> ZoteroUploadError:
    return ZoteroUploadError(reason, 409)


def _check_doi_duplicate(
    zot: _Zotero, ledger: dict, doi: str, title: str, warns: list[str]
) -> tuple[str, str] | None:
    """``None`` when the DOI is new; ``(parent key, empty attachment key)`` to
    attach to an item that has no PDF yet; raises 409 when a PDF is already there."""
    seen = ledger["by_doi"].get(doi)
    if seen:
        raise _refuse(
            f"Already in the Zotero library: item {seen['item_key']} (DOI {doi}), "
            "added by an earlier upload."
        )
    from lightrag import zotero_citations

    for key, entry in zotero_citations._load().items():
        if normalize_doi(entry.get("doi")) == doi:
            raise _refuse(
                f"Already in the Zotero library: DOI {doi} is attachment {key} "
                f"('{entry.get('title', '')[:80]}')."
            )
    try:
        hit = zot.find_by_doi(doi, title)
        if hit is None:
            return None
        pdfs = zot.pdf_children(hit["key"])
    except ZoteroUploadError as e:
        warns.append(f"Duplicate check against Zotero was not possible ({e}).")
        return None
    if any(p.get("md5") for p in pdfs):
        raise _refuse(
            f"Already in the Zotero library: item {hit['key']} (DOI {doi}) holds a PDF."
        )
    empty = next((p["key"] for p in pdfs if not p.get("md5")), "")
    return hit["key"], empty


def add_pdf_to_zotero(
    cfg: ZoteroUploadConfig,
    pdf_path: str,
    filename: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> UploadResult:
    """Blocking; run it in a worker thread. See the module docstring."""
    filename = clean_filename(filename)
    if not filename.lower().endswith(UPLOAD_EXTENSIONS):
        raise ZoteroUploadError("Only PDF files can be added to Zotero.", 400)
    with open(pdf_path, "rb") as fh:
        if not fh.read(1024).lstrip().startswith(b"%PDF"):
            raise ZoteroUploadError("The file is not a PDF.", 400)
    md5 = _md5(pdf_path)

    with _ledger_lock(cfg), httpx.Client(transport=transport) as http:
        ledger = _ledger_read(cfg)
        known = ledger["by_md5"].get(md5)
        key = known["item_key"] if known else _sqlite_has_md5(cfg, md5)
        if key:
            raise _refuse(
                f"This exact file is already in the Zotero library ({key}); nothing added."
            )
        zot, dav = _Zotero(cfg, http), _WebDAV(cfg, http)
        info = read_pdf_info(pdf_path)
        rec, candidate, warns = lookup_crossref(http, cfg, info)
        doi = normalize_doi(rec.get("DOI")) if rec else ""
        title = _first(rec.get("title")) if rec else ""

        parent, empty_row, created_parent = "", "", False
        if doi:
            found = _check_doi_duplicate(zot, ledger, doi, title, warns)
            if found:
                parent, empty_row = found
        metadata = "existing" if parent else ("crossref" if rec else "stub")
        created: list[str] = []
        att_key = empty_row
        try:
            if not parent:
                item = (
                    build_crossref_item(zot, rec)
                    if rec
                    else build_stub_item(zot, filename, candidate)
                )
                parent, _ = zot.create(item)
                created_parent = True
                created.append(parent)
            if not att_key:
                att_key, _ = zot.create(_attachment_item(parent, filename))
                created.append(att_key)
            mtime_ms = dav.put_attachment(att_key, pdf_path, filename, md5)
            zot.set_file_metadata(att_key, mtime_ms, md5)
        except Exception as e:
            cleanup = _rollback(zot, dav, created, att_key, keep_row=bool(empty_row))
            if isinstance(e, ZoteroUploadError):
                reason = str(e)
            else:
                logger.exception("zotero_add: unexpected failure")
                reason = f"unexpected {type(e).__name__}"
            raise ZoteroUploadError(
                f"Adding to Zotero failed: {reason}. {cleanup}"
            ) from e

        record = {"item_key": parent, "attachment_key": att_key, "filename": filename}
        ledger["by_md5"][md5] = record
        if doi:
            ledger["by_doi"][doi] = record
        try:
            _ledger_write(cfg, ledger)
        except OSError as e:
            warns.append(f"The upload ledger could not be written ({e.strerror}).")
        return UploadResult(
            outcome="created" if created_parent else "attached",
            item_key=parent,
            attachment_key=att_key,
            title=title or f"Uploaded: {filename}",
            doi=doi,
            metadata=metadata,
            md5=md5,
            filename=filename,
            warnings=warns,
        )


def _rollback(
    zot: _Zotero, dav: _WebDAV, created: list[str], att_key: str, keep_row: bool
) -> str:
    """Delete what a failed call created; return an honest one-line account."""
    left: list[str] = []
    if att_key:
        try:
            dav.remove_attachment(att_key)
        except ZoteroUploadError as e:
            left.append(f"WebDAV files for {att_key} ({e})")
    for key in reversed(created):
        try:
            zot.delete(key)
        except ZoteroUploadError as e:
            left.append(f"Zotero item {key} ({e})")
    if left:
        logger.error(f"zotero_add: rollback incomplete: {left}")
        return "Cleanup was incomplete; remove by hand: " + "; ".join(left) + "."
    if keep_row:
        return "Nothing was left behind except the empty attachment row that already existed."
    return "Everything this upload created was removed again."


# ------------------------------------------------------------------------- sync

_sync_proc: subprocess.Popen | None = None
_sync_guard = threading.Lock()


def _lock_is_held(path: str) -> bool:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False


def trigger_sync(cfg: ZoteroUploadConfig) -> SyncOutcome:
    """Launch ``ZOTERO_SYNC_COMMAND`` detached unless a sync is already running.

    Single-flight: nothing starts while a command this process launched is still
    alive, or while ``ZOTERO_SYNC_LOCK_FILE`` (the build lock) is held by anyone.
    The command should itself take that lock without waiting (for example
    ``flock -n -o <lock file> <script>``), which covers two workers racing.
    """
    global _sync_proc
    if not cfg.sync_command:
        return SyncOutcome(
            False,
            "not_configured",
            "No sync was triggered (ZOTERO_SYNC_COMMAND is not set): the document "
            "reaches LightRAG with the next scheduled Zotero sync.",
        )
    with _sync_guard:
        busy = _sync_proc is not None and _sync_proc.poll() is None
        try:
            busy = busy or (
                bool(cfg.sync_lock_file) and _lock_is_held(cfg.sync_lock_file)
            )
            argv = shlex.split(cfg.sync_command)
            if not busy:
                out = open(cfg.sync_log, "ab") if cfg.sync_log else subprocess.DEVNULL
                try:
                    _sync_proc = subprocess.Popen(
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    if out is not subprocess.DEVNULL:
                        out.close()
                threading.Thread(target=_sync_proc.wait, daemon=True).start()
        except (OSError, ValueError) as e:
            logger.error(f"zotero_add: sync command failed to start: {e}")
            return SyncOutcome(
                False, "failed", f"The Zotero sync could not be started ({e})."
            )
    if busy:
        return SyncOutcome(
            False,
            "already_running",
            "No new sync was started because one is already running; if it is past "
            "its Zotero sync step, this file waits for the next run.",
        )
    return SyncOutcome(
        True,
        "started",
        "A Zotero sync and corpus build was started on the server; the document "
        "appears in LightRAG when it finishes.",
    )
