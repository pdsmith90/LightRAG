"""Fork (DROP_JUNK_ENTITIES): entity names that are bibliography or document
apparatus rather than subject matter.

A paper corpus hands the extraction LLM reference lists, in-text citations and
captions. Whatever the entity-type guidance says, the model turns cited authors
("Doe, J."), citations ("Roe et al. (2001)"), journal names, DOIs, numbered
figure/table/equation labels and generic placeholder words ("Concept", "Table
Name") into entities. Each becomes a hub linking unrelated papers, and because
the local query leg ranks an entity's relations by the summed degree of their
endpoints, those hubs crowd real neighbours out of the relation budget.

``junk_entity_class(name, raw_type)`` returns the class a name belongs to, or
None to keep it. ``raw_type`` is the type the LLM wrote, before any allow-list
mapping (ENTITY_TYPE_STRICT stores "person" as "other", so the mapped type no
longer says the name is a person); pass None for a relation endpoint, which
carries no type -- the name-shape rules alone apply then.

Two rules consult the Zotero metadata that ``zotero_citations`` loads (surnames
of the library's authors, and the journals it publishes in) so a shape such as
"J. Doe" is only taken for a person when "Doe" is a known author. Without the
metadata those two rules simply never fire.
"""

from __future__ import annotations

import re
import unicodedata

PERSON_TYPES = frozenset(
    {"person", "people", "author", "authors", "researcher", "scientist"}
)
# Types under which a person-shaped name is still a real entity ("N. America" as a
# location, "U.S. Navy" as an organization is handled by the surname check).
PROTECTED_TYPES = frozenset(
    {
        "location",
        "mission",
        "instrument",
        "dataset",
        "software",
        "method",
        "model",
        "artifact",
        "naturalobject",
        "event",
        "quantity",
        "data",
        "concept",
        "equipment",
        "technology",
        "phenomenon",
    }
)
# A model or product named after a year ("Hayford (1909)" as an ellipsoid) is kept
# under these types.
AUTHOR_YEAR_PROTECTED_TYPES = frozenset(
    {"event", "mission", "dataset", "model", "software", "instrument"}
)
# A named archive or service that quotes its address stays an entity.
URL_PROTECTED_TYPES = frozenset(
    {
        "dataset",
        "organization",
        "software",
        "model",
        "mission",
        "instrument",
        "data",
        "location",
        "artifact",
        "website",
    }
)
# Types a journal name arrives under when the LLM extracts a reference's venue.
VENUE_TYPES = frozenset(
    {
        "journal",
        "content",
        "publication",
        "other",
        "organization",
        "unknown",
        "",
        "periodical",
        "source",
    }
)


BIBLIOGRAPHIC_TYPES = frozenset(
    {"journal", "publication", "content", "periodical", "source"}
)
_JOURNAL_WORD = re.compile(
    r"\b(journal|letters?|transactions|proceedings|bulletin|annals|annales|reviews?|reports"
    r"|acta|advances|notices|communications|magazine|international)\b",
    re.IGNORECASE,
)


def fold(name: str) -> str:
    """Case- and punctuation-insensitive key that keeps letters of every script."""
    s = unicodedata.normalize("NFKC", name or "").casefold()
    return "".join(ch for ch in s if ch.isalnum())


_GENERIC_WORDS = (
    "Table",
    "Tables",
    "Table Name",
    "Table of Contents",
    "Figure",
    "Figures",
    "Fig",
    "Section",
    "Chapter",
    "Equation",
    "Equations",
    "Appendix",
    "Author",
    "Authors",
    "Author(s)",
    "The Authors",
    "Concept",
    "Method",
    "Methods",
    "Data",
    "Introduction",
    "Conclusion",
    "Conclusions",
    "Abstract",
    "Reference",
    "References",
    "Acknowledgments",
    "Acknowledgements",
    "Other",
    "Unknown",
    "Entity",
    "Entities",
    "Chapter Contents",
    "Chapter Summary",
    "Chapter Structure",
    "Chapter Content Index",
    "Keywords",
    "Supplementary Material",
    "Results",
    "Discussion",
    "Summary",
    "Paper",
    "Article",
    "Study",
    "This Study",
    "This Paper",
    "Authors' Contributions",
)
GENERIC_FOLDS = frozenset(fold(w) for w in _GENERIC_WORDS)

_UP = r"[A-ZÀ-ÞĀ-ſ]"
_LO = r"[a-zß-ÿĀ-ſ'’\-]"
_PARTICLE = r"(?:(?:van|von|de|der|den|da|di|du|la|le|del|dos)\s)*"
_SURNAME = rf"{_PARTICLE}{_UP}{_LO}+(?:[\s\-]{_UP}{_LO}+)?"
_SURNAME_ANY_CASE = rf"{_PARTICLE}{_UP}[A-Za-zÀ-ſ'’\-]+"
_YEAR = r"(?:1[89]|20)\d\d[a-z]?"

_SURNAME_INITIALS = re.compile(rf"^{_SURNAME},?\s{_UP}\.(?:\s?-?{_UP}\.){{0,2}}$")
_INITIALS_SURNAME = re.compile(rf"^(?:{_UP}\.\s?-?){{1,3}}\s?{_SURNAME}$")
_SURNAME_BARE_INITIALS = re.compile(rf"^{_SURNAME}\s{_UP}{{1,3}}$")
_ET_AL = re.compile(r"\bet\.?\s?al\b", re.IGNORECASE)
_AUTHOR_YEAR = re.compile(
    rf"^{_SURNAME}(?:\s(?:and|&)\s{_SURNAME}|\set\.?\sal\.?)?"
    rf"(?:,\s?{_YEAR}|\s?[\(\[]{_YEAR}(?:,\s?{_YEAR})*[\)\]])$"
)
# "Roe and Moe 2001" -- a bare year is a citation only when two names are joined.
_PAIR_BARE_YEAR = re.compile(
    rf"^{_SURNAME_ANY_CASE}\s(?:and|&)\s{_SURNAME_ANY_CASE},?\s{_YEAR}$",
    re.IGNORECASE,
)
_PAIR = re.compile(rf"^({_SURNAME})\s(?:and|&)\s({_SURNAME})$")
# "NRLMSISE-00 (Doe et al., 2002)": a real name with its citation appended. Judged
# by the part before the parenthesis, unless that part is only a list of surnames.
_TRAILING_CITATION = re.compile(
    rf"^(?P<pre>.+?)\s*[\(\[][^()\[\]]*(?:\bet\.?\s?al\b|{_YEAR})[^()\[\]]*[\)\]]\s*$",
    re.IGNORECASE,
)
_SURNAME_LIST = re.compile(
    rf"^{_SURNAME}(?:(?:,\s?|\s(?:and|&)\s){_SURNAME})*(?:\set\.?\sal\.?)?,?$"
)
# A list of two or more cited authors ("J. Doe and K. Roe", "Doe, J., and Roe, K.",
# "A. Doe, B. Roe, and C. Moe") is the citation itself, whatever type it arrived under.
_INIT_SURNAME_ITEM = rf"(?:{_UP}\.\s?-?){{1,3}}\s?{_SURNAME}"
_SURNAME_INIT_ITEM = rf"{_SURNAME},?\s{_UP}\.(?:\s?-?{_UP}\.){{0,2}}"
_AUTHOR_LIST = re.compile(
    rf"^(?:{_INIT_SURNAME_ITEM}|{_SURNAME_INIT_ITEM})"
    rf"(?:(?:,\s?|,?\s(?:and|&)\s)(?:{_INIT_SURNAME_ITEM}|{_SURNAME_INIT_ITEM}))+,?$"
)
# An untyped relation endpoint shaped like a cited author -- initials and a surname ("J. Doe",
# "J. K. Doe", "Doe JK") -- is taken for a person even when the surname is not in the library:
# relations name cited authors far more often than people a paper is about, and the merge
# would otherwise create the endpoint as an untyped placeholder node (on one corpus 8.8 % of
# all vertices were such placeholders, "R. K. Doe" among them). Two look-alikes stay: an
# abbreviation spelled with periods before a place or body ("N. America", "U.S. Geological
# Survey") and an initial before an organisation or object word ("B. Corporation", "P. Wave").
_INITIALS_ABBREVIATIONS = frozenset(
    {
        "US",
        "UK",
        "UN",
        "EU",
        "DC",
        "NY",
        "LA",
        "SA",
        "NA",
        "PR",
        "USA",
        "USSR",
        "UAE",
        "NZ",
        "BC",
        "NE",
        "NW",
        "SE",
        "SW",
        "ST",
        "MT",
        "FT",
        "CA",
        "CO",
    }
)
_PLACE_OR_BODY_WORDS = frozenset(
    fold(w)
    for w in (
        "America",
        "Americas",
        "American",
        "Africa",
        "African",
        "Europe",
        "European",
        "Asia",
        "Asian",
        "Australia",
        "Antarctica",
        "Antarctic",
        "Arctic",
        "Atlantic",
        "Pacific",
        "Indian",
        "Hemisphere",
        "Pole",
        "Polar",
        "Equatorial",
        "Tropical",
        "Ocean",
        "Sea",
        "Coast",
        "Island",
        "Islands",
        "Shetland",
        "Georgia",
        "Korea",
        "Vietnam",
        "Sudan",
        "Ireland",
        "Ossetia",
        "Cyprus",
        "Yorkshire",
        "Wales",
        "Zealand",
        "Tibet",
        "China",
        "Siberia",
        "Greenland",
        "Alaska",
        "California",
        "Carolina",
        "Dakota",
        "Virginia",
        "Jersey",
        "Hampshire",
        "Mexico",
        "Texas",
        "Cascadia",
        "Andes",
        "Alps",
        "Himalaya",
        "Himalayas",
        "Anatolia",
        "Sahara",
        "Sahel",
        "Plains",
        "Plateau",
        "Basin",
        "Trench",
        "Ridge",
        "Rise",
        "Rift",
        "Fault",
        "Shelf",
        "Slope",
        "Current",
        "Gyre",
        "Front",
        "Monsoon",
        "Jet",
        "Navy",
        "Army",
        "Air",
        "Force",
        "Marines",
        "Survey",
        "Service",
        "Department",
        "Agency",
        "Bureau",
        "Institute",
        "Office",
        "Corps",
        "Guard",
        "Station",
        "University",
        "College",
        "Center",
        "Centre",
        "Laboratory",
        "Observatory",
        "Society",
        "Union",
        "Committee",
        "Commission",
        "Council",
        "Program",
        "Programme",
        "Project",
        "Mission",
        "Standard",
        "Standards",
        "Code",
        "Rule",
        "Law",
        "Theorem",
        "Model",
        "Method",
        "Function",
        "Index",
        "Number",
        "Constant",
        "Equation",
        "Congress",
        "Senate",
        "Government",
        "Fleet",
        "Command",
        "Administration",
        "Authority",
        "Board",
        "Foundation",
        "Academy",
        "Patent",
        "Treasury",
        "Mint",
        "Steel",
        "Bank",
        "Mail",
        "Postal",
        "Highway",
        "Route",
        "Interstate",
        "Dollar",
        "Dollars",
        "Geological",
        "Corporation",
        "Company",
        "Wave",
        "Waves",
        "Field",
        "Band",
        "Shell",
        "Layer",
        "Region",
        "Zone",
        "Belt",
    )
)


def _abbreviated_place_or_body(name: str) -> bool:
    """ "N. America", "U.S. Geological Survey", "B. Corporation": an abbreviation before a place
    or a body word -- not a cited author."""
    initials = "".join(re.findall(rf"({_UP})\.", name))
    if len(initials) >= 2 and initials.upper() in _INITIALS_ABBREVIATIONS:
        return True
    return any(
        fold(t) in _PLACE_OR_BODY_WORDS
        for t in re.split(r"[\s\-\.]+", name)
        if len(t) > 1
    )


_URL_ANYWHERE = re.compile(r"(\b10\.\d{4,9}/|^doi\b|https?://|\bwww\.)", re.IGNORECASE)
_URL_LEADING = re.compile(r"^\W*(10\.\d{4,9}/|doi\b|https?://|www\.)", re.IGNORECASE)
# "Geophys.", "Res." -- at least two letters, so a person's initials ("A. B.") never
# make a journal abbreviation.
_ABBREV_TOKEN = re.compile(rf"^{_UP}[a-z]{{1,9}}\.$")
# The keyword is case-insensitive, the id is not: "Table 2", "Fig. 3a", "Lemma B.I",
# "Appendix E" -- never "Equator", "Equinox", "Equation of state" or "Tablet".
_LABEL = re.compile(
    r"^(?i:fig(?:ure)?s?|tab(?:le)?s?|eqs?|equations?|sect(?:ion)?s?|chapters?|appendix"
    r"|appendices|theorems?|lemmas?|propositions?|corollar(?:y|ies)|definitions?|remarks?"
    r"|algorithms?)"
    r"\.?\s*(?:\(?\d+(?:\.\d+)*[a-z]?\)?|\(?[A-Z]\d*(?:\.(?:\d+|[A-Z]+))*\)?|[IVX]+)"
    r"(?:\s*(?:,|and|&|[-–])\s*\(?\d+(?:\.\d+)*[a-z]?\)?)*\.?:?$"
)
_MARKDOWN = str.maketrans("", "", "_*`")


_lists: tuple | None = (
    None  # (zotero_citations index it was built from, surnames, venues)
)


def _library_lists() -> tuple[frozenset[str], frozenset[str]]:
    """(known author surnames, folded journal names) from the Zotero metadata.

    Rebuilt when ``zotero_citations`` reloads the file; empty when it is absent.
    """
    global _lists
    from lightrag import zotero_citations

    index = zotero_citations._author_year_index()
    if _lists is not None and _lists[0] is index:
        return _lists[1], _lists[2]
    surnames = frozenset(index[1].keys())
    venues = frozenset(
        f
        for entry in zotero_citations._load().values()
        if isinstance(entry, dict)
        for f in [fold(str(entry.get("publication") or ""))]
        if f
    )
    _lists = (index, surnames, venues)
    return surnames, venues


def _surname_known(name: str, surnames: frozenset[str]) -> bool:
    tokens = [t for t in re.split(r"[\s.]+", name) if len(t) > 1 and not t.isupper()]
    return any(t.casefold() in surnames for t in tokens)


def _journal_abbreviation(name: str) -> bool:
    tokens = name.split()
    return (
        len(tokens) >= 2
        and all(t[:1].isupper() for t in tokens)
        and sum(bool(_ABBREV_TOKEN.match(t)) for t in tokens) >= 2
    )


def junk_entity_class(name: str, raw_type: str | None = None) -> str | None:
    """'person', 'citation', 'label', 'generic' or 'numeric' -- or None to keep.

    ``raw_type`` is the LLM's own type, spaces removed and lower-cased; None for
    an untyped relation endpoint.
    """
    n = (name or "").strip()
    if len(n) < 3:  # one or two characters belong to DROP_SYMBOL_ENTITIES
        return None
    typ = raw_type if raw_type is not None else ""
    f = fold(n)
    if f in GENERIC_FOLDS:
        return "generic"
    if not any(ch.isalpha() for ch in n):
        return "numeric"
    plain = n.translate(_MARKDOWN).strip()
    trailing = _TRAILING_CITATION.match(plain)
    if trailing:
        prefix = trailing.group("pre").strip(" ,;:-")
        if prefix and not _SURNAME_LIST.match(prefix):
            return junk_entity_class(prefix, raw_type)
    if _LABEL.match(plain):
        return "label"
    if _ET_AL.search(plain):
        return "citation"
    if _AUTHOR_LIST.match(plain):
        return "citation"
    if typ not in AUTHOR_YEAR_PROTECTED_TYPES and (
        _AUTHOR_YEAR.match(plain) or _PAIR_BARE_YEAR.match(plain)
    ):
        return "citation"
    if _URL_LEADING.match(plain) or (
        _URL_ANYWHERE.search(plain) and typ not in URL_PROTECTED_TYPES
    ):
        return "citation"
    surnames, venues = _library_lists()
    # "Doe and Roe" with no year is often a model or data set named after its
    # authors, so the pair counts only under a type that is not one of those, and
    # never for an untyped endpoint.
    pair = _PAIR.match(plain)
    if (
        pair
        and raw_type is not None
        and typ not in PROTECTED_TYPES
        and all(_surname_known(pair.group(i), surnames) for i in (1, 2))
    ):
        return "citation"
    if _journal_abbreviation(plain):
        if raw_type is None or typ in VENUE_TYPES:
            return "citation"
    elif f in venues and (
        typ in BIBLIOGRAPHIC_TYPES
        or (typ in VENUE_TYPES and _JOURNAL_WORD.search(plain))
    ):
        # A journal title can also name a field ("Inverse Problems", "Space
        # Weather"): spelled out, it counts only when typed as a publication or
        # when the title carries a journal word.
        return "citation"
    if typ in PERSON_TYPES:
        return "person"
    if typ not in PROTECTED_TYPES:
        if _SURNAME_INITIALS.match(plain):
            return "person"
        if _INITIALS_SURNAME.match(plain) or _SURNAME_BARE_INITIALS.match(plain):
            if _surname_known(plain, surnames):
                return "person"
            if raw_type is None and not _abbreviated_place_or_body(plain):
                return "person"
    return None
