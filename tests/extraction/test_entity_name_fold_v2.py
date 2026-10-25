"""ENTITY_NAME_FOLD key rules: the last word's plural, the symbol class, the
deployment's exception lists, the canonical rank and the alias table.

All names below are invented.
"""

import json
import logging
import random

import pytest

import lightrag.entity_name_fold as fold_mod
import lightrag.entity_name_guard as guard
from lightrag.entity_name_fold import (
    alias_chunk_result,
    canonical_name,
    canonical_rank,
    fold_chunk_result,
    fold_key,
    is_symbolic,
    legacy_fold_key,
    load_aliases,
    load_fold_lists,
    load_fold_map,
    plural_rule,
    symbolic_key,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(guard, "_library_lists", lambda: (frozenset(), frozenset()))
    monkeypatch.delenv("ENTITY_NAME_FOLD_LISTS", raising=False)
    monkeypatch.delenv("ENTITY_NAME_ALIASES", raising=False)
    fold_mod.disarm()
    load_fold_lists(None)
    yield
    fold_mod.disarm()
    load_fold_lists(None)


# A deployment's own lists, as a JSON file would hold them.
LISTS = {
    "_comment": "invented test words",
    "version": 1,
    "no_fold": ["zorbels"],
    "no_fold_alone": ["quibbles"],
    "irregular": {"glorbi": "glorbus"},
    "irregular_suffix": [["zorbae", "zorba"]],
    "es_after_s": ["zorbus"],
    "oes_to_o": ["zorpo"],
    "che": ["flumche"],
    "ie": ["flonkie"],
    "s_after_vowel_ok": ["zorbas"],
}


@pytest.fixture
def lists_file(tmp_path):
    path = tmp_path / "fold_lists.json"
    path.write_text(json.dumps(LISTS), encoding="utf-8")
    return path


def _same(a, b):
    ka, kb = fold_key(a), fold_key(b)
    return ka is not None and ka == kb


# (a, b, same key?, why) under the code's default lists.
DEFAULT_CASES = [
    # case, spacing and punctuation, as before
    ("Imaginary Filter", "imaginary-filter", True, "case and punctuation"),
    ("ZORP", "zorp", False, "4-5 character keys keep the case class"),
    ("Model 2", "Model 3", False, "digits stay in the key"),
    # regular plural of the last word, every name shape
    ("Glorbon Models", "glorbon model", True, "-s on a spaced last word"),
    ("GlorbonModels", "Glorbon Model", True, "CamelCase last word"),
    ("glorbon_models", "Glorbon Model", True, "underscored last word"),
    ("Glorbon-Models", "Glorbon Model", True, "hyphenated last word"),
    ("**Glorbon Models**", "Glorbon Model", True, "markdown-wrapped name"),
    ("<i>Glorbon Models</i>", "<i>Glorbon Model</i>", True, "HTML tags peeled"),
    ("PermanentZQXStations", "Permanent ZQX Station", True, "acronym + Titlecase"),
    ("ZQValues", "ZQ Value", True, "a capital before a 4-letter Titlecase word"),
    ("Region2Currents", "Region 2 Current", True, "word glued to a digit"),
    ("FLUMPH SECONDS", "Flumph Second", True, "all-capitals multi-word name"),
    ("Models of Quuxland", "Model of Quuxland", False, "only the LAST word"),
    ("Wobble Delays", "Wobble Delay", True, "-ys is a regular plural"),
    ("Flarn Caps", "Flarn Cap", True, "-ps is a regular plural"),
    ("Flarn Function(s)", "Flarn Function", True, "'(s)' suffix"),
    ("Flarn Function H(s)", "Flarn Function H", False, "'(s)' as argument"),
    # -ies / -es
    ("Zib Anomalies", "Zib Anomaly", True, "-ies -> -y"),
    ("Quux Prairies", "Quux Prairie", True, "-ies -> -ie for listed singulars"),
    ("Flarn Ties", "Flarn Tie", True, "4-letter -ies word -> -ie"),
    ("Wobble Processes", "Wobble Process", True, "-sses -> -ss"),
    ("Wobble Approaches", "Wobble Approach", True, "-ches -> -ch"),
    ("Wobble Brushes", "Wobble Brush", True, "-shes -> -sh"),
    ("Wobble Fluxes", "Wobble Flux", True, "-xes -> -x"),
    ("Wobble Buzzes", "Wobble Buzz", True, "-zzes -> -zz"),
    ("Wobble Biases", "Wobble Bias", True, "listed -s singular + es"),
    (
        "Wobble Megaviruses",
        "Wobble Megavirus",
        True,
        "compound of a listed -s singular",
    ),
    ("Wobble Phases", "Wobble Phase", True, "other -es drops only the s"),
    ("Volcanoes", "Volcano", True, "listed -oes -> -o"),
    ("Wobble Shoes", "Wobble Shoe", True, "other -oes drops only the s"),
    ("Zorp Caches", "Zorp Cache", True, "listed -ches drops only the s"),
    # irregular
    ("Flarn Matrices", "Flarn Matrix", True, "matrices -> matrix"),
    ("Flarn Submatrices", "Flarn Submatrix", True, "irregular ending compounds"),
    ("Flarn Indices", "Flarn Index", True, "indices -> index"),
    ("Flarn Polyhedra", "Flarn Polyhedron", True, "-hedra -> -hedron"),
    ("Flarn Bispectra", "Flarn Bispectrum", True, "-spectra -> -spectrum"),
    ("Flarn Criteria", "Flarn Criterion", True, "criteria"),
    ("Flarn Radii", "Flarn Radius", True, "radii -> radius"),
    ("Flarn Reanalyses", "Flarn Reanalysis", True, "-lyses -> -lysis"),
    ("Flarn Hypotheses", "Flarn Hypothesis", True, "-theses -> -thesis"),
    ("Flarn Axes", "Flarn Axis", True, "axes -> axis"),
    ("Flarn Bases", "Flarn Basis", True, "bases -> basis"),
    ("Zorp Data Bases", "Zorp Data Base", True, "data bases -> base"),
    ("Flarn Ice Shelves", "Flarn Ice Shelf", True, "shelves -> shelf"),
    ("Flarn Maxima", "Flarn Maximum", True, "maxima -> maximum"),
    ("Flarn Media", "Flarn Medium", True, "media -> medium"),
    ("Zorp Data", "Zorp Datum", False, "data never -> datum"),
    ("Flarn Ellipses", "Flarn Ellipse", True, "ellipses -> ellipse"),
    # acronym plurals and acronyms
    ("Zorp Elevation Model ZEMs", "Zorp Elevation Model ZEM", True, "acronym plural"),
    ("QXTIDs", "QXTID", True, "acronym plural, 5-character key"),
    ("ZEMs", "ZEM", False, "singular key under 4 characters: plain key kept"),
    ("ZORPS", "Zorp", False, "an all-capitals single word is an acronym"),
    ("Blorft QPS", "Blorft QP", False, "all-capitals last word in a mixed name"),
    ("ZQobs", "ZQob", False, "mixed-case word never folds"),
    # -ics by position
    ("Wobble Statistics", "Wobble Statistic", True, "-ics after another word"),
    ("Statistics", "Statistic", False, "-ics alone never folds"),
    # words that only look plural
    ("Flarn Series", "Flarn Serie", False, "series is singular"),
    ("Wobble Lens", "Wobble Len", False, "lens is singular"),
    ("Flarn Glass", "Flarn Glas", False, "-ss never folds"),
    ("Flarn Nervous", "Flarn Nervou", False, "-ous never folds"),
    ("Flarnis", "Flarni", False, "-is folds only for listed plurals"),
    ("Flarnus", "Flarnu", False, "-us folds only for listed plurals"),
    ("Flarnas", "Flarna", False, "-as folds only for listed plurals"),
    ("Flarnos", "Flarno", False, "-os folds only for listed plurals"),
    # symbol class
    ("q_7 = 2 - j*sqrt(3)", "q_7 = 2 + j*sqrt(3)", False, "the sign is kept"),
    ("Wobble Number k2'", "Wobble Number k2", False, "a prime is kept"),
    ("Wobble Number k2′", "Wobble Number k2'", True, "prime variants unified"),
    ("Matrix A+", "Matrix A", False, "plus kept"),
    ("Matrix A⁺", "Matrix A+", True, "superscript plus = plus"),
    ("S+ and S-", "Sands", False, "a symbolic key never meets a word key"),
    ("x^T A x", "X^T A x", False, "short variable tokens keep their case"),
    ("Load Wobble Number k′", "load wobble number k′", True, "prose casefolds"),
    ("τ_q<sup>2</sup>", "τ_q^2", True, "<sup> = ^"),
    ("Flarn ∇g(y)", "Flarn ∇g (y)", True, "whitespace ignored"),
    ("Panel+Z", "Panel-Z", False, "plus vs minus"),
    ("Panel−Z", "Panel-Z", True, "U+2212 is not a marker"),
    ("Zorpsat-1", "Zorpsat 1", True, "a hyphen is not a marker"),
    ("Quuxville*", "Quuxville", True, "a footnote star is not an operator"),
    ("Flarn's Functions", "Flarn's Function", True, "possessive is not a prime"),
    ("Imaginary Society **(IMS)**", "Imaginary Society (IMS)", True, "bold"),
    ("Glorbon Models = P", "Glorbon Model = P", False, "symbolic: no plural fold"),
    ("Glorbon Models'", "Glorbon Model", True, "plural possessive is not a prime"),
]


@pytest.mark.parametrize("a,b,same,why", DEFAULT_CASES)
def test_default_key_rules(a, b, same, why):
    assert _same(a, b) is same, f"{a!r} {fold_key(a)!r} vs {b!r} {fold_key(b)!r}: {why}"


@pytest.mark.parametrize(
    "name,rule",
    [
        ("Glorbon Models", "s"),
        ("Zib Anomalies", "ies"),
        ("Wobble Fluxes", "es"),
        ("Flarn Radii", "irregular"),
        ("Flarn Bases", "irregular"),
        ("Zorp Data Bases", "s"),
        ("QXTIDs", "acronym"),
        ("Wobble Statistics", "ics"),
        ("Flarn Function(s)", "paren_s"),
        ("Glorbon Model", None),
        ("Glorbon Models = P", None),  # symbolic
        ("ZEMs", None),  # the singular's key would be too short
        ("Abs", None),  # no key at all
    ],
)
def test_plural_rule_names_the_rule(name, rule):
    assert plural_rule(name) == rule


def test_symbol_gate():
    assert is_symbolic("x + y") and is_symbolic("a <= b") and is_symbolic("Q200'")
    assert is_symbolic("F<sup>-1</sup>")  # <sup> is an exponent
    assert not is_symbolic("a - b") and not is_symbolic("x ~ y")  # not markers
    assert not is_symbolic("Students'") and not is_symbolic("Quuxville*")
    assert not is_symbolic("<i>Glorbon</i>")  # other tags are dropped
    assert not is_symbolic("'Glorbon Model'")  # wrapping quotes
    assert fold_key("x + yz").startswith("§")
    assert symbolic_key("x+y") is None  # under four characters
    assert fold_key("+ +") is None  # fewer than two alphanumerics
    assert fold_key("F<sup>-1</sup>") == fold_key("F^{-1}")


def test_legacy_key_is_the_plain_fold():
    assert legacy_fold_key("Glorbon Models") == "glorbonmodels"
    assert legacy_fold_key("Glorbon Models") != legacy_fold_key("Glorbon Model")
    assert legacy_fold_key("ZORP") == "zorp|U" and legacy_fold_key("Zorp") == "zorp|m"
    assert legacy_fold_key("Co") is None
    # the plain fold ignores symbols
    assert legacy_fold_key("q_7 = 2 - j*sqrt(3)") == legacy_fold_key(
        "q_7 = 2 + j*sqrt(3)"
    )


# ------------------------------------------------------------------- lists file


@pytest.mark.parametrize(
    "a,b,same_default,same_listed",
    [
        ("Zorbels", "Zorbel", True, False),  # no_fold
        ("Quibbles", "Quibble", True, False),  # no_fold_alone, alone
        ("Sample Quibbles", "Sample Quibble", True, True),  # ... after a word
        ("Flarn Glorbi", "Flarn Glorbus", False, True),  # irregular
        ("Megazorbae", "Megazorba", False, True),  # irregular_suffix
        ("Big Zorbuses", "Big Zorbus", False, True),  # es_after_s
        ("Big Zorpoes", "Big Zorpo", False, True),  # oes_to_o
        ("Big Flumches", "Big Flumche", False, True),  # che
        ("Big Flonkies", "Big Flonkie", False, True),  # ie
        ("Big Flonkies", "Big Flonky", True, False),
        ("Wire Zorbas", "Wire Zorba", False, True),  # s_after_vowel_ok
    ],
)
def test_lists_file_extends_the_defaults(lists_file, a, b, same_default, same_listed):
    assert _same(a, b) is same_default
    load_fold_lists(lists_file)
    assert _same(a, b) is same_listed
    load_fold_lists(None)  # back to the defaults
    assert _same(a, b) is same_default


def test_lists_file_keeps_the_defaults_and_reports_counts(lists_file):
    base = load_fold_lists(None)
    assert base["source"] == "defaults" and base["no_fold_alone"] == 0
    counts = load_fold_lists(lists_file)
    assert counts["source"] == str(lists_file)
    assert counts["no_fold"] == base["no_fold"] + 1
    assert counts["irregular_suffix"] == base["irregular_suffix"] + 1
    assert counts["s_after_vowel_ok"] == 1
    assert not _same("Flarn Series", "Flarn Serie")  # a default is still there
    assert _same("Flarn Matrices", "Flarn Matrix")


def test_lists_file_errors(tmp_path, caplog, monkeypatch):
    with pytest.raises(FileNotFoundError):
        load_fold_lists(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"no_fold": "zorbels"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_fold_lists(bad)
    bad.write_text(json.dumps({"irregular_suffix": [["zorbae"]]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_fold_lists(bad)
    bad.write_text(json.dumps(["zorbels"]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_fold_lists(bad)
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps({"no_fold": ["zorbels"], "nofold": []}), encoding="utf-8")
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        assert load_fold_lists(odd)["source"] == str(odd)
    assert "nofold" in caplog.text  # reported, not silently dropped
    assert not _same("Zorbels", "Zorbel")


# --------------------------------------------------------------- canonical rank


def test_canonical_rank_orders_junk_flaws_degree_plural_name():
    members = [
        ("Table 2", "other", 900),  # apparatus: last whatever its degree
        ("_Wobble Rate", "concept", 50),  # markup
        ("Wobble Rates", "concept", 7),
        ("Wobble Rate", "concept", 7),
        ("Wobble rate", "concept", 7),
        ("Wobble Rate", "concept", 3),
    ]
    order = sorted(members, key=lambda m: canonical_rank(*m))
    assert order == [
        ("Wobble Rate", "concept", 7),  # degree, then singular, then code point
        ("Wobble rate", "concept", 7),
        ("Wobble Rates", "concept", 7),
        ("Wobble Rate", "concept", 3),
        ("_Wobble Rate", "concept", 50),
        ("Table 2", "other", 900),
    ]
    assert canonical_rank("Wobble Rate", None, None) == (
        False,
        False,
        0,
        False,
        "Wobble Rate",
    )


def test_canonical_rank_degree_beats_singular():
    assert canonical_rank("Wobble Harmonics", "concept", 900) < canonical_rank(
        "Wobble Harmonic", "concept", 16
    )


@pytest.mark.parametrize(
    "name",
    ["Wobble Rate ", " Wobble Rate", "Wobble  Rate", "Wobble\nRate", "Wobble\tRate"]
    + ["-Wobble Rate", "Wobble `Rate`", "Wobble {Rate}", "Wobble\\Rate", "~Wobble"],
)
def test_flawed_spellings_rank_after_clean_ones(name):
    assert canonical_rank(name, "concept", 100)[1] is True
    assert canonical_rank("Wobble Rate", "concept", 1) < canonical_rank(
        name, "concept", 100
    )


def test_unflawed_shapes():
    assert canonical_rank("(Wobble) Rate", "concept", 1)[1] is False
    assert canonical_rank("x_1 + y", "variable", 1)[1] is False  # symbolic content


def test_camelcase_glue_is_a_flaw_only_beside_a_spaced_sibling():
    siblings = ["RootMeanZorpError", "Root Mean Zorp Error"]
    assert fold_key(siblings[0]) == fold_key(siblings[1])
    assert canonical_rank("RootMeanZorpError", "concept", 79, siblings)[1] is True
    assert canonical_rank("RootMeanZorpError", "concept", 79)[1] is False  # alone
    # two words of CamelCase are brand-like and never a flaw
    assert canonical_rank("ZorpSat", "mission", 50, ["ZorpSat", "Zorp Sat"])[1] is False


def test_load_fold_map_uses_the_rank_with_siblings():
    rows = [
        ("RootMeanZorpError", "concept", 79),
        ("Root Mean Zorp Error", "concept", 3),
        ("ZorpSat", "mission", 50),
        ("Zorpsat", "mission", 1),
        ("Glorbon Models", "model", 4),
        ("Glorbon Model", "model", 4),
        ("glorbon model", "model", 4),
        ("Table 2", "other", 90),  # apparatus ...
        ("table-2", "other", 5),  # ... while this spelling is not
        ("Figure 3", "other", 9),  # every spelling apparatus
        ("figure 3", "other", 2),
    ]
    assert load_fold_map(rows) == 4  # the all-apparatus key gets no entry
    assert canonical_name("root mean zorp error") == "Root Mean Zorp Error"
    assert canonical_name("Zorp-Sat") == "ZorpSat"
    assert canonical_name("glorbon models") == "Glorbon Model"
    assert canonical_name("table 2") == "table-2"
    assert canonical_name("FIGURE 3") == "FIGURE 3"


def test_load_fold_map_is_independent_of_row_order():
    rows = [
        (f"{a} {b}", "concept", d)
        for a in ("Wobble", "wobble", "WOBBLE")
        for b in ("Rate", "Rates", "rate", "rates")
        for d in (1, 5)
    ]
    maps = set()
    for seed in range(20):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        load_fold_map(shuffled)
        maps.add(tuple(sorted(fold_mod._MAP.items())))
    assert len(maps) == 1
    # plain code-point order: capitals before lowercase letters
    assert dict(next(iter(maps))) == {"wobblerate": "WOBBLE Rate"}


# ------------------------------------------------------------------ alias table

NODES = [
    ("Zorp Analysis Kit", "software", 40),
    ("Glorbon Model", "model", 12),
    ("Quux Basin", "location", 7),
]
ALIASES = [
    ("ZAK", "Zorp Analysis Kit"),  # key "zak": too short, skipped
    ("ZAK Toolkit", "Zorp Analysis Kit"),
    ("Glorbon Models", "Glorbon Model"),  # same key: skipped
    ("zak-toolkit", "Quux Basin"),  # same key as a row above: first wins
    ("Flarn Basin", "Quux Basin"),
]


def test_load_aliases_skips_same_key_and_keyless_rows(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        assert load_aliases(ALIASES) == 2
    assert "first row wins" in caplog.text
    assert fold_mod._ALIASES == {
        fold_key("ZAK Toolkit"): "Zorp Analysis Kit",
        fold_key("Flarn Basin"): "Quux Basin",
    }
    assert load_aliases(None) == 0 and fold_mod._ALIASES == {}


def test_load_aliases_reads_a_csv(tmp_path):
    path = tmp_path / "aliases.csv"
    path.write_text(
        'variant,canonical\nZAK Toolkit,Zorp Analysis Kit\n"Flarn, the\nBasin",Quux Basin\n',
        encoding="utf-8",
    )
    assert load_aliases(path) == 2
    assert load_aliases(str(path)) == 2
    assert fold_mod._ALIASES[fold_key("Flarn, the\nBasin")] == "Quux Basin"


def test_load_aliases_keys_under_the_active_lists(lists_file):
    rows = [("Zorbels", "Zorbel Index")]
    assert load_aliases(rows) == 1
    load_fold_lists(lists_file)
    assert load_aliases(rows) == 1
    assert fold_key("Zorbels") in fold_mod._ALIASES


def test_canonical_name_consults_the_aliases_first():
    load_aliases(ALIASES)
    assert canonical_name("zak toolkit") == "zak toolkit"  # disarmed: unchanged
    load_fold_map(NODES + [("ZAK Toolkit", "software", 99)])
    assert canonical_name("zak toolkit") == "Zorp Analysis Kit"  # beats the map
    assert canonical_name("ZAK-Toolkit", register=False) == "Zorp Analysis Kit"
    assert canonical_name("glorbon models") == "Glorbon Model"  # the map's own key
    assert canonical_name("ZAK") == "ZAK"  # no key


def test_chunk_results_honour_the_aliases():
    load_aliases(ALIASES)
    load_fold_map(NODES)
    nodes = {
        "ZAK toolkit": [{"entity_name": "ZAK toolkit", "description": "a"}],
        "Zorp Analysis Kit": [{"entity_name": "Zorp Analysis Kit", "description": "b"}],
        "flarn basin": [{"entity_name": "flarn basin", "description": "c"}],
    }
    edges = {
        ("ZAK toolkit", "flarn basin"): [
            {"src_id": "ZAK toolkit", "tgt_id": "flarn basin"}
        ],
        ("ZAK toolkit", "Zorp Analysis Kit"): [
            {"src_id": "ZAK toolkit", "tgt_id": "Zorp Analysis Kit"}
        ],
    }
    n, e = alias_chunk_result(
        {k: [dict(r) for r in v] for k, v in nodes.items()},
        {k: [dict(r) for r in v] for k, v in edges.items()},
    )
    assert set(n) == {"ZAK toolkit", "Zorp Analysis Kit", "flarn basin", "Quux Basin"}
    assert [r["description"] for r in n["Zorp Analysis Kit"]] == ["a", "b"]
    assert ("Zorp Analysis Kit", "Quux Basin") in e
    assert ("Zorp Analysis Kit", "Zorp Analysis Kit") not in e

    n, e = fold_chunk_result(nodes, edges)
    assert set(n) == {"Zorp Analysis Kit", "Quux Basin"}
    assert all(r["entity_name"] == "Zorp Analysis Kit" for r in n["Zorp Analysis Kit"])
    assert set(e) == {("Zorp Analysis Kit", "Quux Basin")}  # the self-loop is gone


def test_disarm_clears_the_aliases():
    load_aliases(ALIASES)
    load_fold_map(NODES)
    fold_mod.disarm()
    load_fold_map(NODES)
    assert canonical_name("zak toolkit") == "zak toolkit"


# --------------------------------------------------------------- build_fold_map


class _Graph:
    graph_name = "g"

    def __init__(self, nodes):
        self.nodes = nodes

    async def _query(self, sql):
        assert '"g".base' in sql
        return [{"name": n, "type": t, "degree": d} for n, t, d in self.nodes]


async def test_build_fold_map_loads_lists_and_aliases_from_env(
    tmp_path, lists_file, monkeypatch, caplog
):
    aliases = tmp_path / "aliases.csv"
    aliases.write_text(
        "variant,canonical\nZAK Toolkit,Zorp Analysis Kit\nZorbels,Zorbel\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENTITY_NAME_FOLD_LISTS", str(lists_file))
    monkeypatch.setenv("ENTITY_NAME_ALIASES", str(aliases))
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)
    graph = _Graph(NODES + [("Zorbel", "concept", 3)])
    with caplog.at_level(logging.INFO, logger="lightrag"):
        assert await fold_mod.build_fold_map(graph) == 4
    assert fold_mod.is_armed()
    assert (
        f"ENTITY_NAME_FOLD armed: 4 name keys, 2 aliases, lists {lists_file}, "
        "from 4 nodes" in caplog.text
    )
    assert canonical_name("zak toolkit") == "Zorp Analysis Kit"
    # "zorbels" is on the lists' no_fold list, so only the alias joins it
    assert fold_key("Zorbels") != fold_key("Zorbel")
    assert canonical_name("zorbels") == "Zorbel"


async def test_build_fold_map_without_env_uses_the_defaults(caplog, monkeypatch):
    load_aliases(ALIASES)  # left over from an earlier load
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="lightrag"):
        assert await fold_mod.build_fold_map(_Graph(NODES)) == 3
    assert "3 name keys, 0 aliases, lists defaults" in caplog.text
    assert canonical_name("zak toolkit") == "zak toolkit"


async def test_build_fold_map_fails_on_a_missing_lists_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTITY_NAME_FOLD_LISTS", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        await fold_mod.build_fold_map(_Graph(NODES))
    assert not fold_mod.is_armed()
