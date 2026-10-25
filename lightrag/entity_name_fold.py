"""Fork (ENTITY_NAME_FOLD): one graph node for names that differ only in case,
spacing, punctuation or the number of their last word ("Kalman Filter",
"Kalman filter", "KalmanFilter", "Kalman-Filter", "Kalman Filters").

The extraction LLM spells the same entity several ways across papers and the
merge keys on the exact string, so a concept's relations scatter over several
nodes and none of them collects its whole neighbourhood. With the switch on,
every name an extraction produces is mapped to the spelling the graph already
uses for its fold key -- the best-ranked one (``canonical_rank``) -- or becomes
that spelling when the key is new. It is applied where extraction results leave
``extract_entities`` (so the write-ahead anchors and the merge see the same
names) and, as an alias, where the knowledge rebuild re-parses cached results.

The key (``fold_key``):

- NFKC + casefold + the letters and digits of every script
  (``entity_name_guard.fold``). Keys under four characters never fold
  ("CO"/"Co"); keys of four or five characters carry an all-capitals/mixed case
  class, so a short acronym and a word stay apart ("TIDE"/"tide").
- The LAST word is singularised first ("Glorbon Models" = "glorbon
  model"): regular -s/-es/-ies plurals, acronym plurals ("DEMs" -> "DEM"), a
  "(s)" suffix, Latin/Greek and English irregulars, and -ics only after another
  word. All-capitals words in a mixed-case name are acronyms and never fold;
  words ending -ss/-ous, or -is/-us/-as/-os unless listed, never fold. If the
  singular's key would be under four characters the name keeps its plain key.
- A "symbolic" name -- one holding an operator, comparison or prime outside
  HTML tags (= + ^ * < > | and their Unicode relatives, <sup>, or an apostrophe
  used as a prime) -- gets a key that keeps those symbols, ignores whitespace and
  TeX/markdown noise and keeps the case of short variable tokens; it starts with
  '§' so it never meets a word key, and is never plural-folded ("x + y" and
  "x - y" stay apart). Minus/hyphen and '~' are not markers.

``legacy_fold_key`` is the key before the plural and symbol rules, kept for
tools that must reproduce maps built by the earlier code.

Exception lists. The code carries generic linguistic lists only (structural
words that end like plurals, irregular plurals, spelling rules). A deployment
adds its own -- surnames and place names that look like plurals, words that do
not fold when they are the whole name, -is/-us/-as/-os plurals, irregulars of
its field -- in a JSON file (env ``ENTITY_NAME_FOLD_LISTS``, ``load_fold_lists``),
merged into the defaults::

    {
      "version": 1,
      "no_fold": ["word", ...],          # never singularised
      "no_fold_alone": ["word", ...],    # not singularised when the whole name
      "irregular": {"plural": "singular", ...},
      "irregular_suffix": [["plural ending", "singular ending"], ...],
      "es_after_s": ["singular", ...],   # singulars in -s whose plural adds -es
      "oes_to_o": ["singular", ...],     # -oes plurals that drop -es
      "che": ["singular", ...],          # -ches plurals that drop only the s
      "ie": ["singular", ...],           # -ies plurals of -ie singulars
      "s_after_vowel_ok": ["plural", ...]  # -is/-us/-as/-os words that are plurals
    }

All words are lowercase ASCII; keys starting with "_" are comments. Unknown
keys are reported and ignored.

Alias table. A CSV of ``variant,canonical`` rows (env ``ENTITY_NAME_ALIASES``,
``load_aliases``) maps names whose fold key differs from their canonical's --
spellings a bulk merge joined by rules the key does not know. It is keyed by
``fold_key(variant)`` and consulted before the map, so new extractions and the
rebuild alias both land on the canonical. Rows whose two keys are equal, or
whose variant has no key, are skipped (the map already covers them).

Names the apparatus guard (``entity_name_guard``) would drop never become the
canonical spelling.

The map is built from the graph once the storages are up (Postgres graph
storage only) and is armed when complete; until then names pass unchanged.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict

from lightrag.constants import (
    DEFAULT_ENTITY_NAME_ALIASES,
    DEFAULT_ENTITY_NAME_FOLD_LISTS,
)
from lightrag.entity_name_guard import fold, junk_entity_class
from lightrag.utils import get_env_value, logger

MIN_KEY = 4
SHORT_KEY = 5

_ARMED = False
_MAP: dict[str, str] = {}
_ALIASES: dict[str, str] = {}
_LISTS_SOURCE = "defaults"


def _case_class(name: str) -> str:
    cased = [c for c in name if c.isupper() or c.islower()]
    return "U" if cased and all(c.isupper() for c in cased) else "m"


def legacy_fold_key(name: str) -> str | None:
    """The key before the plural and symbol rules (case, spacing and punctuation
    only)."""
    f = fold(name)
    if len(f) < MIN_KEY:
        return None
    return f"{f}|{_case_class(name)}" if len(f) <= SHORT_KEY else f


# ------------------------------------------------------------------ symbol class

# '~' is left out (mostly OCR noise and markdown strike-through in extracted
# names), and so is the whole minus/hyphen family ('-', U+2212, superscript
# minus): a hyphen or a title dash far more often than a minus, and an expression
# with a real minus nearly always carries another operator ("1 - i" vs "1 + i":
# the '+' side is symbolic).
_OPERATORS = frozenset("=+^*<>|±∓≤≥≠≈∑∫√∂∇′″‴⁺∗")
_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9]*)\s*/?>")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")  # markdown **bold**
_FOOTNOTE_STAR = re.compile(r"(?<=[A-Za-z]{4})\*+(?![\w(])")  # "Quuxville*"
_PRIME_APOS = re.compile(r"(?<=[A-Za-z0-9)\]}])'+(?![A-Za-z])")
_POSSESSIVE = re.compile(r"(?<=[a-z]s)'(?![A-Za-z])")  # "Students'" is no prime
_QUOTED = re.compile(r"^\s*(['\"`])(.*)\1\s*$", re.S)
_SYM_MAP = str.maketrans(
    {
        "−": "-",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "∗": "*",
        "′": "'",
        "’": "'",
        "‘": "'",
        "´": "'",
        "″": "''",
        "‴": "'''",
        "⁺": "+",
        "⁻": "-",
    }
)
_SYM_DROP = re.compile(r"[\s_$`{}\\]")
_LONG_RUN = re.compile(r"[^\W\d_]{3,}")


def _unwrap(name: str) -> str:
    """Strip one pair of wrapping quotes, markdown bold and HTML tags; <sup>
    becomes '^' (a superscript is an exponent, prime or star index)."""
    s = name or ""
    m = _QUOTED.match(s)
    if m:
        s = m.group(2)
    s = _BOLD.sub(r"\1", s)
    return _TAG.sub(lambda m: "^" if m.group(0).lower() == "<sup>" else "", s)


def is_symbolic(name: str) -> bool:
    """True when ``name`` holds an operator, comparison or prime."""
    s = _FOOTNOTE_STAR.sub("", _unwrap(name).translate(_SYM_MAP))
    if any(c in _OPERATORS for c in s):
        return True
    return bool(_PRIME_APOS.search(s)) and not _POSSESSIVE.search(s)


def symbolic_key(name: str) -> str | None:
    """The key of a symbolic name: '§' + its symbols and casefolded prose words."""
    s = _FOOTNOTE_STAR.sub("", _unwrap(name).translate(_SYM_MAP))
    s = _SYM_DROP.sub("", unicodedata.normalize("NFKC", s).translate(_SYM_MAP))
    s = _LONG_RUN.sub(lambda m: m.group(0).casefold(), s).rstrip(".,;:")
    if sum(ch.isalnum() for ch in s) < 2 or len(s) < MIN_KEY:
        return None
    return "§" + s


# ------------------------------------------------------------------- last word

_TAG_END = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\s*/?>$")
_ACR_PLURAL = re.compile(r"[A-Z]{2,}s")


def last_word(name: str):
    """(start, end, word, shape) of the name's last word, or None.

    The last word is the final run of ASCII letters after trailing punctuation
    and HTML tags are peeled, split at the last lower->Upper CamelCase boundary
    (or before a capital that starts a Titlecase word after an acronym:
    "PermanentZQXStations", "ZQValues"). It may follow a digit but not a
    non-ASCII letter. shape: 'lower' | 'title' | 'upper' | 'acr_s' (an acronym
    plural such as 'DEMs') | 'mixed'.
    """
    i = len(name)
    while i > 0:
        m = _TAG_END.search(name, 0, i)
        if m:
            i = m.start()
            continue
        if not name[i - 1].isalnum():
            i -= 1
            continue
        break
    j = i
    while j > 0 and name[j - 1].isascii() and name[j - 1].isalpha():
        j -= 1
    if j == i or (j > 0 and name[j - 1].isalpha()):  # glued to a non-ASCII letter
        return None
    run = name[j:i]
    k0 = 0
    for k in range(len(run) - 1, 0, -1):
        # CamelCase: split at the last lower->Upper ("GlorbonModels"), or before
        # the last capital of an acronym of two or more that starts a Titlecase word
        # ("PermanentZQXStations"); "ZQobs" (one capital before) is not split.
        if run[k].isupper() and (
            run[k - 1].islower()
            or (
                k >= 2
                and run[k - 2 : k].isupper()
                and run[k + 1 : k + 4].islower()
                and len(run[k + 1 : k + 4]) == 3
            )
            or (
                run[k - 1].isupper()
                and run[k + 1 : k + 5].islower()
                and len(run[k + 1 : k + 5]) == 4
            )  # "ZQValues", not "ZQobs"
        ):
            k0 = k
            break
    w = run[k0:]
    if _ACR_PLURAL.fullmatch(w):
        shape = "acr_s"
    elif w.islower():
        shape = "lower"
    elif w[0].isupper() and (len(w) == 1 or w[1:].islower()):
        shape = "title"
    elif w.isupper():
        shape = "upper"
    else:
        shape = "mixed"
    return j + k0, i, w, shape


# ---------------------------------------------------------------- default lists

# Words that end like a plural but are singular (or plural-only) nouns.
_DEFAULT_NO_FOLD = frozenset(
    """
    series species news lens bias gas atlas alias canvas chaos cosmos ethos pathos
    logos thermos kudos corps biceps triceps forceps mumps measles diabetes herpes
    rabies scabies facies caries odds remains whereabouts headquarters crossroads
    barracks savings earnings surroundings outskirts
    """.split()
)

# Not folded when the word is the whole name (a deployment lists its own).
_DEFAULT_NO_FOLD_ALONE: frozenset[str] = frozenset()

# plural -> singular for the whole last word (lowercase).
_DEFAULT_IRREGULAR = {
    # Latin/Greek -a <- -um / -on
    "spectra": "spectrum",
    "media": "medium",
    "minima": "minimum",
    "maxima": "maximum",
    "extrema": "extremum",
    "optima": "optimum",
    "momenta": "momentum",
    "equilibria": "equilibrium",
    "symposia": "symposium",
    "strata": "stratum",
    "septa": "septum",
    "phyla": "phylum",
    "quanta": "quantum",
    "continua": "continuum",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "taxa": "taxon",
    "perihelia": "perihelion",
    "aphelia": "aphelion",
    "automata": "automaton",
    # -ae <- -a
    "formulae": "formula",
    "antennae": "antenna",
    "aurorae": "aurora",
    "nebulae": "nebula",
    "hyperbolae": "hyperbola",
    "parabolae": "parabola",
    "laminae": "lamina",
    "lamellae": "lamella",
    "lacunae": "lacuna",
    "larvae": "larva",
    "vertebrae": "vertebra",
    "algae": "alga",
    "medusae": "medusa",
    "pinnae": "pinna",
    # -i <- -us
    "radii": "radius",
    "moduli": "modulus",
    "nuclei": "nucleus",
    "stimuli": "stimulus",
    "foci": "focus",
    "loci": "locus",
    "termini": "terminus",
    "cumuli": "cumulus",
    "tori": "torus",
    "fungi": "fungus",
    # -ices <- -ex / -ix
    "apices": "apex",
    "helices": "helix",
    "cortices": "cortex",
    "appendices": "appendix",
    "codices": "codex",
    "simplices": "simplex",
    "vortices": "vortex",
    # -es <- -is
    "axes": "axis",
    "crises": "crisis",
    "oases": "oasis",
    "diagnoses": "diagnosis",
    "prognoses": "prognosis",
    "metastases": "metastasis",
    # Greek -mata <- -ma
    "schemata": "schema",
    "stomata": "stoma",
    "lemmata": "lemma",
    # English
    "plateaux": "plateau",
    "tableaux": "tableau",
    "leaves": "leaf",
    "sheaves": "sheaf",
    "halves": "half",
    "wolves": "wolf",
    "calves": "calf",
    "knives": "knife",
    "wives": "wife",
    "lives": "life",
    "women": "woman",
    "children": "child",
    "feet": "foot",
    "teeth": "tooth",
    "mice": "mouse",
    "geese": "goose",
}

# Endings that compound (bispectra, submatrices, icosahedra, reanalyses ...).
_DEFAULT_IRREGULAR_SUFFIX = (
    ("matrices", "matrix"),
    ("indices", "index"),
    ("vertices", "vertex"),
    ("spectra", "spectrum"),
    ("hedra", "hedron"),
    ("lyses", "lysis"),
    ("theses", "thesis"),
    ("novae", "nova"),
    ("shelves", "shelf"),
)

# -ics: folded only after another word, whatever the word. After another word a
# count noun and a field noun alike name the same concept as their -ic form
# ("Sample Statistics", "Wobble Dynamics"); alone both kinds part ways
# ("Statistics" the field vs a statistic).

# Singulars ending in s whose plural adds -es (biases, lenses, viruses).
_DEFAULT_ES_AFTER_S = frozenset(
    """
    bias gas alias atlas canvas lens virus sinus bus focus status census bonus
    campus corpus consensus radius plexus cirrus nexus apparatus octopus genius
    abacus callus isthmus rhombus torus bolus fetus chorus walrus prospectus
    syllabus thesaurus platypus citrus discus stylus minus plus iris
    """.split()
)
_DEFAULT_OES_TO_O = frozenset(
    """
    echo volcano tornado zero hero potato tomato torpedo mosquito domino veto
    embargo cargo buffalo mango motto halo grotto
    """.split()
)
_DEFAULT_CHE = frozenset(
    "cache niche avalanche headache moustache psyche cliche quiche creche microniche".split()
)
_DEFAULT_IE = frozenset(
    """
    movie cookie pie tie lie die prairie calorie zombie rookie goalie hippie
    brownie smoothie sortie coterie menagerie reverie rotisserie genie pixie
    auntie birdie boogie hoagie bootie
    """.split()
)
# Plurals in -is / -us / -as / -os. The generic rule skips those endings (most
# such words are singulars or names); a deployment lists the true plurals it has
# seen next to their singular.
_DEFAULT_S_AFTER_VOWEL_OK: frozenset[str] = frozenset()

# The active lists (defaults plus a deployment's file; see load_fold_lists).
NO_FOLD = _DEFAULT_NO_FOLD
NO_FOLD_ALONE = _DEFAULT_NO_FOLD_ALONE
IRREGULAR = dict(_DEFAULT_IRREGULAR)
IRREGULAR_SUFFIX = _DEFAULT_IRREGULAR_SUFFIX
ES_AFTER_S = _DEFAULT_ES_AFTER_S
_ES_AFTER_S_LONG = tuple(sorted(x for x in ES_AFTER_S if len(x) >= 4))
OES_TO_O = _DEFAULT_OES_TO_O
CHE = _DEFAULT_CHE
IE = _DEFAULT_IE
S_AFTER_VOWEL_OK = _DEFAULT_S_AFTER_VOWEL_OK

_SET_LISTS = (
    "no_fold",
    "no_fold_alone",
    "es_after_s",
    "oes_to_o",
    "che",
    "ie",
    "s_after_vowel_ok",
)


def _words_of(value, key: str) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(w, str) for w in value):
        raise ValueError(f"fold lists: {key!r} must be a list of strings")
    return frozenset(w.strip().lower() for w in value if w.strip())


def load_fold_lists(path=None) -> dict:
    """Install the exception lists: the code's defaults plus, when ``path`` names
    a JSON file (schema in the module docstring), that file's entries. ``None``
    restores the defaults. Returns the size of each active list and the source.
    Raises on a missing or malformed file."""
    global NO_FOLD, NO_FOLD_ALONE, IRREGULAR, IRREGULAR_SUFFIX, ES_AFTER_S
    global _ES_AFTER_S_LONG, OES_TO_O, CHE, IE, S_AFTER_VOWEL_OK, _LISTS_SOURCE
    data: dict = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("fold lists: the file must hold a JSON object")
    known = set(_SET_LISTS) | {"irregular", "irregular_suffix", "version"}
    unknown = sorted(k for k in data if k not in known and not k.startswith("_"))
    if unknown:
        logger.warning(f"ENTITY_NAME_FOLD lists: ignored unknown keys {unknown}")
    sets = {k: _words_of(data.get(k, []), k) for k in _SET_LISTS}
    irregular = data.get("irregular", {})
    if not isinstance(irregular, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in irregular.items()
    ):
        raise ValueError("fold lists: 'irregular' must map strings to strings")
    suffix = data.get("irregular_suffix", [])
    if not isinstance(suffix, list) or not all(
        isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p)
        for p in suffix
    ):
        raise ValueError(
            "fold lists: 'irregular_suffix' must be [ending, ending] pairs"
        )

    NO_FOLD = _DEFAULT_NO_FOLD | sets["no_fold"]
    NO_FOLD_ALONE = _DEFAULT_NO_FOLD_ALONE | sets["no_fold_alone"]
    IRREGULAR = {
        **_DEFAULT_IRREGULAR,
        **{k.strip().lower(): v.strip().lower() for k, v in irregular.items()},
    }
    extra = tuple(
        (a.strip().lower(), b.strip().lower())
        for a, b in suffix
        if (a.strip().lower(), b.strip().lower()) not in _DEFAULT_IRREGULAR_SUFFIX
    )
    IRREGULAR_SUFFIX = _DEFAULT_IRREGULAR_SUFFIX + extra
    ES_AFTER_S = _DEFAULT_ES_AFTER_S | sets["es_after_s"]
    _ES_AFTER_S_LONG = tuple(sorted(x for x in ES_AFTER_S if len(x) >= 4))
    OES_TO_O = _DEFAULT_OES_TO_O | sets["oes_to_o"]
    CHE = _DEFAULT_CHE | sets["che"]
    IE = _DEFAULT_IE | sets["ie"]
    S_AFTER_VOWEL_OK = _DEFAULT_S_AFTER_VOWEL_OK | sets["s_after_vowel_ok"]
    _LISTS_SOURCE = str(path) if path else "defaults"
    return {
        "source": _LISTS_SOURCE,
        "no_fold": len(NO_FOLD),
        "no_fold_alone": len(NO_FOLD_ALONE),
        "irregular": len(IRREGULAR),
        "irregular_suffix": len(IRREGULAR_SUFFIX),
        "es_after_s": len(ES_AFTER_S),
        "oes_to_o": len(OES_TO_O),
        "che": len(CHE),
        "ie": len(IE),
        "s_after_vowel_ok": len(S_AFTER_VOWEL_OK),
    }


# -------------------------------------------------------------------- singulars


def singular(word: str, alone: bool, before: str = "") -> tuple[str, str] | None:
    """(singular, rule) for a lowercase ASCII word, or None when it does not fold.

    ``alone``: nothing alphanumeric precedes the word. ``before``: fold() of the
    text before it (only "data bases" uses it).
    """
    w = word
    if len(w) < 4 or w in NO_FOLD or (alone and w in NO_FOLD_ALONE):
        return None
    if w in IRREGULAR:
        return IRREGULAR[w], "irregular"
    if w == "bases":  # "Flarn Bases" -> Basis; "Zorp Data Bases" -> Base
        return ("base", "s") if before.endswith("data") else ("basis", "irregular")
    for suf, rep in IRREGULAR_SUFFIX:
        if w.endswith(suf):
            return w[: -len(suf)] + rep, "irregular"
    if w.endswith("ics"):
        return None if alone else (w[:-1], "ics")
    if not w.endswith("s") or w.endswith(("ss", "ous")):
        return None
    if w.endswith("ies"):
        if len(w) == 4 or w[:-1] in IE:  # ties; movies, prairies
            return w[:-1], "ies"
        return w[:-3] + "y", "ies"
    if w.endswith("es"):
        stem = w[:-2]
        if w.endswith("sses") or stem in ES_AFTER_S or stem.endswith(_ES_AFTER_S_LONG):
            return stem, "es"
        if w.endswith(("ches", "shes")):
            return (w[:-1], "s") if w[:-1] in CHE else (stem, "es")
        if w.endswith(("xes", "zzes")):
            return stem, "es"
        if w.endswith("oes"):
            return (stem, "es") if stem in OES_TO_O else (w[:-1], "s")
        return w[:-1], "s"
    if w.endswith(("is", "us", "as", "os")):
        return (w[:-1], "s") if w in S_AFTER_VOWEL_OK else None
    return w[:-1], "s"


# -------------------------------------------------------------------------- key

# "Flarn Function(s)", but not "Flarn Function H(s)" (an argument).
_PAREN_S = re.compile(r"(?<=[A-Za-z]{2}[a-z])\((?:s|es)\)(?=\W*$)")


def _singularised(name: str):
    """(name with its last word singularised, rule) or None."""
    m = _PAREN_S.search(name)
    if m:  # "Flarn Function(s)"
        return name[: m.start()] + name[m.end() :], "paren_s"
    lw = last_word(name)
    if lw is None:
        return None
    start, end, word, shape = lw
    before = fold(name[:start])
    if (
        shape == "upper"
        and before
        and len(word) >= 5
        and not any(c.islower() for c in name)
    ):
        # An all-capitals name carries no case information: "FLUMPH SECONDS" folds
        # like "Flumph Seconds". Alone, or under five letters, the word is taken for
        # an acronym.
        shape = "lower"
        word = word.lower()
    if shape == "acr_s":
        sg, rule = word[:-1], "acronym"
    elif shape in ("lower", "title"):
        r = singular(word.lower(), not before, before)
        if r is None:
            return None
        sg, rule = r
        if shape == "title":
            sg = sg[:1].upper() + sg[1:]
    else:
        return None
    return name[:start] + sg + name[end:], rule


def fold_key(name: str) -> str | None:
    """The map key for ``name``, or None when the name is never folded."""
    if not name:
        return None
    if is_symbolic(name):
        return symbolic_key(name)
    f = fold(name)
    if len(f) < MIN_KEY:
        return None
    cc_name = name
    s = _singularised(name)
    if s is not None:
        f2 = fold(s[0])
        if len(f2) >= MIN_KEY:
            f, cc_name = f2, s[0]
    return f"{f}|{_case_class(cc_name)}" if len(f) <= SHORT_KEY else f


def plural_rule(name: str) -> str | None:
    """The plural rule ``fold_key`` applied to ``name`` (None: no plural fold)."""
    if not name or is_symbolic(name) or len(fold(name)) < MIN_KEY:
        return None
    s = _singularised(name)
    if s is None or len(fold(s[0])) < MIN_KEY:
        return None
    return s[1]


# ---------------------------------------------------------------- canonical rank

_MARKUP = re.compile(r"[_*`$<>{}~\\]|^[^\w(\[]")
_BAD_SPACE = re.compile(r"\s{2,}|\n|\t")
_CAMEL = re.compile(r"[a-z][A-Z][a-z]")
_WORD_SPLIT = re.compile(r"[\s_\-/]+|(?<=[a-z])(?=[A-Z])")


def _is_junk(name: str, raw_type) -> bool:
    return (
        junk_entity_class(name, (raw_type or "").replace(" ", "").lower()) is not None
    )


def _flawed(name: str, siblings=None) -> bool:
    """A spelling no member of a cluster should keep when a clean one exists:
    markup characters or a leading non-word character, stray whitespace, or
    CamelCase glue of three or more words while a spaced sibling with at least as
    many words exists. Symbolic names are never flawed (their symbols are
    content)."""
    if is_symbolic(name):
        return False
    if _MARKUP.search(name):
        return True
    if name != name.strip() or _BAD_SPACE.search(name):
        return True
    if siblings and " " not in name and _CAMEL.search(name):
        n_words = len([w for w in _WORD_SPLIT.split(name) if w])
        if n_words >= 3 and any(
            o != name and " " in o and len(o.split()) >= n_words for o in siblings
        ):
            return True
    return False


def canonical_rank(name: str, raw_type, degree, siblings=None) -> tuple:
    """Sort key of a cluster member; the smallest is the canonical spelling.

    (apparatus name, flawed spelling, -degree, plural, name): apparatus names
    last, then flawed spellings, then the best-connected, a singular before a
    plural, and plain code-point order of the name last (Title Case before
    lowercase), so the choice is deterministic. ``siblings``: the cluster's other
    names, for the CamelCase-glue flaw; without them a name is judged alone.
    """
    return (
        _is_junk(name, raw_type),
        _flawed(name, siblings),
        -int(degree or 0),
        plural_rule(name) is not None,
        name,
    )


# ------------------------------------------------------------------ alias table


def load_aliases(path_or_rows) -> int:
    """Install the alias table from a ``variant,canonical`` CSV path or from
    ``(variant, canonical)`` rows; ``None`` empties it. Keyed by
    ``fold_key(variant)`` under the active lists, so load the lists first. Rows
    whose variant has no key or shares its canonical's key are skipped; of rows
    with the same key the first wins. Returns the number of aliases installed."""
    global _ALIASES
    if path_or_rows is None:
        _ALIASES = {}
        return 0
    if isinstance(path_or_rows, (str, bytes)) or hasattr(path_or_rows, "__fspath__"):
        csv.field_size_limit(sys.maxsize)
        with open(path_or_rows, encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        if rows and [c.strip().lower() for c in rows[0][:2]] == [
            "variant",
            "canonical",
        ]:
            rows = rows[1:]
    else:
        rows = list(path_or_rows)
    new: dict[str, str] = {}
    skipped = conflicts = 0
    for row in rows:
        if not row:
            continue
        variant, canon = row[0], row[1]
        if not variant or not canon:
            skipped += 1
            continue
        key = fold_key(variant)
        if key is None or key == fold_key(canon):
            skipped += 1
            continue
        if key in new:
            if new[key] != canon:
                conflicts += 1
            continue
        new[key] = canon
    if conflicts:
        logger.warning(
            f"ENTITY_NAME_FOLD aliases: {conflicts} rows name a second canonical "
            "for a key already aliased; the first row wins"
        )
    if skipped:
        logger.debug(f"ENTITY_NAME_FOLD aliases: skipped {skipped} rows")
    _ALIASES = new
    return len(_ALIASES)


# ------------------------------------------------------------------------- map


def canonical_name(name: str, register: bool = True) -> str:
    """The spelling the graph uses for ``name``'s key -- the alias table first,
    then the map; with ``register``, a name whose key is new becomes that
    spelling."""
    if not _ARMED:
        return name
    key = fold_key(name)
    if key is None:
        return name
    alias = _ALIASES.get(key)
    if alias is not None:
        return alias
    canon = _MAP.get(key)
    if canon is None:
        if register:
            _MAP[key] = name
        return name
    return canon


def load_fold_map(rows) -> int:
    """Fill the map from ``(name, stored type, degree)`` rows; arm it. Returns the
    number of keys. Each key's spelling is the member with the smallest
    ``canonical_rank`` among the names sharing it; apparatus names are
    skipped."""
    global _ARMED
    groups: dict[str, list] = defaultdict(list)
    for name, typ, degree in rows:
        if not name:
            continue
        key = fold_key(name)
        if key is None:
            continue
        groups[key].append((name, typ, degree))
    new: dict[str, str] = {}
    for key, members in groups.items():
        if len(members) == 1:
            name, typ, _degree = members[0]
            if not _is_junk(name, typ):
                new[key] = name
            continue
        names = [m[0] for m in members]
        best = min(canonical_rank(n, t, d, names) for n, t, d in members)
        if not best[0]:  # the best is junk only when every member is
            new[key] = best[-1]
    _MAP.clear()
    _MAP.update(new)
    _ARMED = True
    return len(_MAP)


def disarm() -> None:
    global _ARMED, _ALIASES
    _ARMED = False
    _MAP.clear()
    _ALIASES = {}


def is_armed() -> bool:
    return _ARMED


async def build_fold_map(graph) -> int | None:
    """Load the exception lists (env ``ENTITY_NAME_FOLD_LISTS``) and the alias
    table (env ``ENTITY_NAME_ALIASES``), read every node's name, type and degree
    from a Postgres/AGE graph storage and arm the map. Returns the key count, or
    None when the storage is not supported (folding then stays off)."""
    graph_name = getattr(graph, "graph_name", None)
    if graph_name is None or not hasattr(graph, "_query"):
        logger.warning(
            "ENTITY_NAME_FOLD ignored: it needs the Postgres (AGE) graph storage"
        )
        return None
    started = time.monotonic()
    lists_path = get_env_value(
        "ENTITY_NAME_FOLD_LISTS", DEFAULT_ENTITY_NAME_FOLD_LISTS, str
    ).strip()
    aliases_path = get_env_value(
        "ENTITY_NAME_ALIASES", DEFAULT_ENTITY_NAME_ALIASES, str
    ).strip()
    lists = load_fold_lists(lists_path or None)
    n_aliases = load_aliases(aliases_path or None)
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
        f"ENTITY_NAME_FOLD armed: {keys} name keys, {n_aliases} aliases, "
        f"lists {lists['source']}, from {len(rows or [])} nodes "
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
