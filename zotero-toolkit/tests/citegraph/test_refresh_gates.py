"""Offline tests for citegraph/refresh_gates.py: the checks refresh.sh and load_zstruct.sh make on a qa.json."""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

CITEGRAPH = Path(__file__).resolve().parents[2] / "citegraph"
spec = importlib.util.spec_from_file_location(
    "refresh_gates", CITEGRAPH / "refresh_gates.py"
)
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)

COUNTS = {
    "papers": 1000,
    "cites": 500,
    "cite_contexts": 300,
    "external_works": 200,
    "paper_embeddings": 900,
}


def qa_file(tmp_path, hard=True, oracle=True, counts=COUNTS, drop=()):
    qa = {
        "hard_checks": {"passed": hard},
        "oracle": {"passed": oracle},
        "row_counts": dict(counts),
    }
    for key in drop:
        qa.pop(key)
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(qa))
    return str(path)


def run(capsys, monkeypatch, *argv, stdin=""):
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    rc = gates.main(list(argv))
    out, err = capsys.readouterr()
    return rc, out, err


def live_lines(counts):
    return "".join(f"{t} {n}\n" for t, n in counts.items())


class TestQaGate:
    def test_both_checks_pass(self, tmp_path, capsys, monkeypatch):
        assert run(capsys, monkeypatch, "qa-gate", qa_file(tmp_path))[0] == 0

    def test_failed_hard_checks(self, tmp_path, capsys, monkeypatch):
        rc, _, err = run(capsys, monkeypatch, "qa-gate", qa_file(tmp_path, hard=False))
        assert rc == 1 and "hard checks" in err

    def test_failed_reference_check(self, tmp_path, capsys, monkeypatch):
        rc, _, err = run(
            capsys, monkeypatch, "qa-gate", qa_file(tmp_path, oracle=False)
        )
        assert rc == 1 and "reference check" in err

    def test_missing_reference_check(self, tmp_path, capsys, monkeypatch):
        rc, _, err = run(
            capsys, monkeypatch, "qa-gate", qa_file(tmp_path, drop=("oracle",))
        )
        assert rc == 1 and "reference check" in err

    def test_null_reference_check(self, tmp_path, capsys, monkeypatch):
        path = tmp_path / "qa.json"
        path.write_text(
            json.dumps(
                {"hard_checks": {"passed": True}, "oracle": None, "row_counts": COUNTS}
            )
        )
        assert run(capsys, monkeypatch, "qa-gate", str(path))[0] == 1

    def test_missing_hard_checks(self, tmp_path, capsys, monkeypatch):
        assert (
            run(
                capsys, monkeypatch, "qa-gate", qa_file(tmp_path, drop=("hard_checks",))
            )[0]
            == 1
        )

    def test_unreadable_or_malformed_file_is_bad_input(
        self, tmp_path, capsys, monkeypatch
    ):
        assert run(capsys, monkeypatch, "qa-gate", str(tmp_path / "none.json"))[0] == 2
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert run(capsys, monkeypatch, "qa-gate", str(bad))[0] == 2
        bad.write_text("[1, 2]")
        assert run(capsys, monkeypatch, "qa-gate", str(bad))[0] == 2


class TestHardChecks:
    def test_needs_only_the_hard_checks(self, tmp_path, capsys, monkeypatch):
        assert (
            run(
                capsys,
                monkeypatch,
                "hard-checks",
                qa_file(tmp_path, oracle=False, drop=("oracle",)),
            )[0]
            == 0
        )

    def test_failed(self, tmp_path, capsys, monkeypatch):
        assert (
            run(capsys, monkeypatch, "hard-checks", qa_file(tmp_path, hard=False))[0]
            == 1
        )


class TestRowCounts:
    def test_tab_separated_and_sorted(self, tmp_path, capsys, monkeypatch):
        rc, out, _ = run(capsys, monkeypatch, "row-counts", qa_file(tmp_path))
        assert rc == 0
        assert out.splitlines() == [f"{t}\t{n}" for t, n in sorted(COUNTS.items())]

    def test_refuses_a_table_name_that_is_not_an_identifier(
        self, tmp_path, capsys, monkeypatch
    ):
        rc, _, err = run(
            capsys,
            monkeypatch,
            "row-counts",
            qa_file(tmp_path, counts={"papers; DROP": 1}),
        )
        assert rc == 2 and "unexpected table name" in err

    def test_refuses_a_qa_without_counts(self, tmp_path, capsys, monkeypatch):
        assert (
            run(
                capsys,
                monkeypatch,
                "row-counts",
                qa_file(tmp_path, drop=("row_counts",)),
            )[0]
            == 2
        )


class TestShrink:
    def shrink(self, tmp_path, capsys, monkeypatch, new, live):
        return run(
            capsys,
            monkeypatch,
            "shrink",
            qa_file(tmp_path, counts=new),
            "-",
            stdin=live_lines(live),
        )

    def test_equal_and_growing_pass(self, tmp_path, capsys, monkeypatch):
        grown = {t: n + 5 for t, n in COUNTS.items()}
        assert self.shrink(tmp_path, capsys, monkeypatch, COUNTS, COUNTS)[0] == 0
        assert self.shrink(tmp_path, capsys, monkeypatch, grown, COUNTS) == (0, "", "")

    def test_just_below_98_percent_fails(self, tmp_path, capsys, monkeypatch):
        rc, out, _ = self.shrink(
            tmp_path, capsys, monkeypatch, {"papers": 979}, {"papers": 1000}
        )
        assert rc == 1 and out.strip() == "papers 1000 -> 979"

    def test_exactly_98_percent_passes(self, tmp_path, capsys, monkeypatch):
        assert (
            self.shrink(
                tmp_path, capsys, monkeypatch, {"papers": 980}, {"papers": 1000}
            )[0]
            == 0
        )

    def test_the_boundary_is_exact_for_small_counts(
        self, tmp_path, capsys, monkeypatch
    ):
        # 49 of 50 is exactly 98%; 48 of 50 is under it
        assert (
            self.shrink(tmp_path, capsys, monkeypatch, {"papers": 49}, {"papers": 50})[
                0
            ]
            == 0
        )
        rc, out, _ = self.shrink(
            tmp_path, capsys, monkeypatch, {"papers": 48}, {"papers": 50}
        )
        assert rc == 1 and out.strip() == "papers 50 -> 48"

    def test_lists_every_shrunk_table_sorted(self, tmp_path, capsys, monkeypatch):
        new = {"papers": 10, "cites": 5, "external_works": 200}
        live = {"papers": 1000, "cites": 500, "external_works": 200}
        rc, out, _ = self.shrink(tmp_path, capsys, monkeypatch, new, live)
        assert rc == 1 and out.strip() == "cites 500 -> 5, papers 1000 -> 10"

    def test_a_table_new_to_the_build_cannot_shrink(
        self, tmp_path, capsys, monkeypatch
    ):
        new = {"papers": 1000, "brand_new": 1}
        assert self.shrink(tmp_path, capsys, monkeypatch, new, {"papers": 1000})[0] == 0

    def test_a_live_table_the_build_dropped_is_ignored(
        self, tmp_path, capsys, monkeypatch
    ):
        assert (
            self.shrink(
                tmp_path,
                capsys,
                monkeypatch,
                {"papers": 1000},
                {"papers": 1000, "old_table": 50},
            )[0]
            == 0
        )

    def test_an_empty_live_table_does_not_block(self, tmp_path, capsys, monkeypatch):
        assert (
            self.shrink(tmp_path, capsys, monkeypatch, {"papers": 0}, {"papers": 0})[0]
            == 0
        )

    def test_a_build_that_emptied_a_table_fails(self, tmp_path, capsys, monkeypatch):
        assert (
            self.shrink(tmp_path, capsys, monkeypatch, {"papers": 0}, {"papers": 1000})[
                0
            ]
            == 1
        )

    def test_malformed_live_lines_are_bad_input(self, tmp_path, capsys, monkeypatch):
        for text in (
            "papers\n",
            "papers x\n",
            "papers 1 2\n",
            "ERROR: relation does not exist\n",
        ):
            rc, _, err = run(
                capsys, monkeypatch, "shrink", qa_file(tmp_path), "-", stdin=text
            )
            assert rc == 2 and "cannot read a live count" in err, text

    def test_blank_lines_are_skipped_and_a_live_file_works(
        self, tmp_path, capsys, monkeypatch
    ):
        live = tmp_path / "live.txt"
        live.write_text("\npapers 1000\n\n")
        rc, out, _ = run(
            capsys,
            monkeypatch,
            "shrink",
            qa_file(tmp_path, counts={"papers": 900}),
            str(live),
        )
        assert rc == 1 and out.strip() == "papers 1000 -> 900"


class TestSummary:
    def test_the_five_headline_tables(self, tmp_path, capsys, monkeypatch):
        rc, out, _ = run(capsys, monkeypatch, "summary", qa_file(tmp_path))
        assert rc == 0
        assert (
            out.strip()
            == "papers=1000 cites=500 cite_contexts=300 external_works=200 paper_embeddings=900"
        )

    def test_a_missing_table_counts_as_zero(self, tmp_path, capsys, monkeypatch):
        rc, out, _ = run(
            capsys, monkeypatch, "summary", qa_file(tmp_path, counts={"papers": 7})
        )
        assert rc == 0 and out.startswith("papers=7 cites=0 ")


def test_runs_as_a_script(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(CITEGRAPH / "refresh_gates.py"),
            "qa-gate",
            qa_file(tmp_path, oracle=False),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1 and "reference check" in proc.stderr
