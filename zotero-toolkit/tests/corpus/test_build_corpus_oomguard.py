#!/usr/bin/env python3
"""Behavioural tests for build_corpus's failure containment.
Run:  .venv/bin/python test_build_corpus_oomguard.py     (plain asserts; pytest also collects it)
Covers: the local OCR server is stopped on EVERY exit of main() (normal, exception, SIGTERM); a pool worker that dies
(SIGKILL, as the kernel OOM killer does) costs one recorded document, not the build; the per-worker memory ceiling is a
clean per-document failure that children (ocrmypdf, llama-server) do not inherit; a page too dense for pymupdf4llm's
layout model (N x N matrices: 16-25 GiB for 12-20k text lines) is read from the text layer instead.
Synthetic folders and PDFs only; convert() is replaced by fakes where a worker has to die; all text is invented."""

import contextlib
import faulthandler
import io
import json
import os
import signal
import stat
import sys
import tempfile
import time
import types

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import pymupdf
import build_corpus as bc

SENTINEL = "KILL0ME1"  # contains 0 and 1, which real Zotero keys never do


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def fake_convert_killing(sentinels):
    """convert() stand-in: a worker handed a sentinel key kills itself the way the kernel OOM killer would."""

    def convert(key):
        if key in sentinels:
            os.kill(os.getpid(), signal.SIGKILL)
        return (key, "pdf", "", 1)

    return convert


def _native_crash():
    """Die by SIGSEGV, as a crash inside a C extension would. pytest's faulthandler (inherited through fork)
    would print a fatal-error trace for it; the crash is expected here."""
    faulthandler.disable()
    os.kill(os.getpid(), signal.SIGSEGV)


def tiny_zotero(td, keys):
    for k in keys:
        os.makedirs(os.path.join(td, "zotero", "storage", k))
    return os.path.join(td, "zotero")


def run_main(td, keys, *extra):
    """main() on a synthetic storage tree with the real pool; returns (exit code or None, stdout, manifest rows)."""
    zotero = tiny_zotero(td, keys)
    out = os.path.join(td, "out")
    argv = [
        "build_corpus.py",
        "--zotero",
        zotero,
        "--out",
        out,
        "--meta",
        os.path.join(td, "none.json"),
        "--denylist",
        "",
        "--oom-keys",
        os.path.join(td, "oom.tsv"),
        "--ocr-vlm",
        "off",
        "--jobs",
        "2",
        *extra,
    ]
    log, code = io.StringIO(), None
    with patched(sys, "argv", argv), contextlib.redirect_stdout(log):
        try:
            bc.main()
        except SystemExit as e:
            code = e.code
    manifest = os.path.join(out, ".manifest.jsonl")
    rows = (
        [json.loads(row) for row in open(manifest)] if os.path.exists(manifest) else []
    )
    return code, log.getvalue(), rows


@contextlib.contextmanager
def stop_counter():
    calls = []
    with patched(bc, "_vlm_local_stop", lambda: calls.append(1)):
        yield calls


def test_ocr_server_is_stopped_when_the_pool_raises():
    with tempfile.TemporaryDirectory() as td, stop_counter() as stops:

        def boom(*a, **kw):
            raise RuntimeError("pool exploded")

        with patched(bc, "run_pool", boom):
            try:
                run_main(td, ["AAAA1111"])
            except RuntimeError as e:
                assert "exploded" in str(e)
            else:
                raise AssertionError("the exception was swallowed")
    assert stops == [1], stops


def test_ocr_server_is_stopped_on_sigterm_and_handlers_are_restored():
    before = signal.getsignal(signal.SIGTERM)
    with tempfile.TemporaryDirectory() as td, stop_counter() as stops:

        def slow(*a, **kw):
            os.kill(
                os.getpid(), signal.SIGTERM
            )  # cron / `kill` while the pool is working
            time.sleep(10)
            raise AssertionError("SIGTERM did not interrupt the build")

        with patched(bc, "run_pool", slow):
            code, _, _ = run_main(td, ["AAAA1111"])
    assert code == 128 + signal.SIGTERM, code
    assert stops == [1], stops
    assert signal.getsignal(signal.SIGTERM) is before


def test_ocr_server_is_stopped_after_a_normal_build_and_stop_is_the_if_build_form():
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "ocr_server.sh")
        record = os.path.join(td, "calls.txt")
        with open(script, "w") as f:
            f.write(f'#!/bin/sh\necho "$@" >> {record}\n')
        os.chmod(script, os.stat(script).st_mode | stat.S_IXUSR)
        with (
            patched(bc, "OCR_VLM_LOCAL_START", script),
            patched(bc, "convert", fake_convert_killing(set())),
        ):
            code, _, rows = run_main(td, ["AAAA1111", "BBBB2220"])
        assert code is None and len(rows) == 2, (code, rows)
        assert open(record).read().split("\n")[:-1] == ["stop --if-build"], open(
            record
        ).read()


def test_a_stop_that_fails_does_not_change_the_outcome():
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "ocr_server.sh")
        with open(script, "w") as f:
            f.write("#!/bin/sh\nsleep 30\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IXUSR)
        real_run = bc.subprocess.run

        def quick_timeout(args, **kw):
            kw["timeout"] = 0.5
            return real_run(args, **kw)

        with (
            patched(bc, "OCR_VLM_LOCAL_START", script),
            patched(bc.subprocess, "run", quick_timeout),
            patched(bc, "convert", fake_convert_killing(set())),
        ):
            code, out, rows = run_main(td, ["AAAA1111"])
        assert code is None and len(rows) == 1, (code, rows)
        assert "ocr_server.sh stop failed" in out, out


def test_a_killed_worker_costs_one_recorded_document_not_the_build():
    with tempfile.TemporaryDirectory() as td, stop_counter() as stops:
        keys = (
            ["A%07d" % i for i in range(1, 9)]
            + [SENTINEL]
            + ["Z%07d" % i for i in range(1, 9)]
        )
        with patched(bc, "convert", fake_convert_killing({SENTINEL})):
            code, out, rows = run_main(td, keys)
        by = {r["key"]: r["method"] for r in rows}
        assert code is None, (code, out)
        assert sorted(by) == sorted(keys), sorted(
            set(keys) - set(by)
        )  # every key accounted for, once
        assert len(rows) == len(keys)
        assert by[SENTINEL] == "error:worker_killed", by[SENTINEL]
        assert all(m == "pdf" for k, m in by.items() if k != SENTINEL), by
        rec = open(os.path.join(td, "oom.tsv")).read().splitlines()
        assert len(rec) == 1 and rec[0].startswith(SENTINEL + "\t"), rec
        assert "died (exit -9)" in rec[0], rec[0]
        assert "BrokenProcessPool" in out and SENTINEL in out, out
        assert stops == [1], stops
        assert SENTINEL in bc.load_oom_keys(
            os.path.join(td, "oom.tsv")
        )  # so the next run skips it


def test_a_crashing_worker_gets_one_retry_a_killed_one_does_not():
    with tempfile.TemporaryDirectory() as td:

        def convert(key):
            if key == SENTINEL and not os.path.exists(os.path.join(td, "crashed")):
                open(os.path.join(td, "crashed"), "w").close()
                _native_crash()  # a one-off native crash
            return (key, "pdf", "", 1)

        keys = ["A%07d" % i for i in range(1, 5)] + [SENTINEL]
        with patched(bc, "convert", convert):
            code, out, rows = run_main(td, keys)
        by = {r["key"]: r["method"] for r in rows}
        assert (
            code is None
            and len(rows) == len(keys)
            and all(m == "pdf" for m in by.values())
        ), (by, out[-300:])
        assert (
            not os.path.exists(os.path.join(td, "oom.tsv"))
            and f"retried once: {SENTINEL}" in out
        ), out
    # the same document crashing every time is recorded at the second crash
    with tempfile.TemporaryDirectory() as td:

        def always(key):
            if key == SENTINEL:
                _native_crash()
            return (key, "pdf", "", 1)

        with patched(bc, "convert", always):
            code, out, rows = run_main(
                td, ["A%07d" % i for i in range(1, 5)] + [SENTINEL]
            )
        by = {r["key"]: r["method"] for r in rows}
        assert (
            code is None and by[SENTINEL] == "error:worker_killed" and len(rows) == 5
        ), (by, out[-300:])
        rec = open(os.path.join(td, "oom.tsv")).read().splitlines()
        assert len(rec) == 1 and "died (exit -11)" in rec[0], rec


def test_two_killed_workers_in_one_death_are_both_recorded():
    with tempfile.TemporaryDirectory() as td:
        sentinels = {"KILL0ME1", "KILL0ME2", "KILL0ME3"}
        keys = ["A%07d" % i for i in range(1, 7)] + sorted(sentinels)
        with patched(bc, "convert", fake_convert_killing(sentinels)):
            code, out, rows = run_main(td, keys)
        by = {r["key"]: r["method"] for r in rows}
        assert code is None and len(rows) == len(keys), (code, len(rows), out)
        assert {k for k, m in by.items() if m == "error:worker_killed"} == sentinels, by
        assert {
            row.split("\t")[0] for row in open(os.path.join(td, "oom.tsv"))
        } == sentinels


def test_pool_restarts_are_capped():
    with tempfile.TemporaryDirectory() as td:
        keys = ["K%07d" % i for i in range(1, 41)]
        got = []
        with patched(bc, "convert", fake_convert_killing(set(keys))):
            left = bc.run_pool(
                keys,
                2,
                {},
                got.append,
                max_restarts=2,
                oom_file=os.path.join(td, "oom.tsv"),
            )
        assert left and len(left) < len(keys), (
            len(left),
            len(got),
        )  # gave up, with work left
        assert len(got) + len(left) == len(keys), (
            len(got),
            len(left),
        )  # nothing lost, nothing counted twice
        with (
            patched(bc, "convert", fake_convert_killing(set(keys))),
            stop_counter() as stops,
        ):
            code, out, rows = run_main(td, keys, "--max-pool-restarts", "1")
        assert code == 1 and "FATAL" in out, (code, out[-300:])
        assert stops == [1]


def test_an_exception_out_of_convert_fails_that_key_only():
    def convert(key):
        if key == SENTINEL:
            # a lone surrogate written outside convert()'s own try once stopped a whole build
            raise UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed")
        return (key, "pdf", "", 1)

    with tempfile.TemporaryDirectory() as td, patched(bc, "convert", convert):
        code, out, rows = run_main(td, ["AAAA1111", SENTINEL, "ZZZZ9990"])
    by = {r["key"]: r["method"] for r in rows}
    assert (
        code is None
        and by[SENTINEL] == "error:UnicodeEncodeError"
        and by["ZZZZ9990"] == "pdf"
    ), (code, by)


def _vmsize_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmSize"):
            return int(line.split()[1]) / 2**20


def test_worker_memory_ceiling_is_a_memoryerror_and_children_do_not_inherit_it():
    cap = _vmsize_gib() + 0.6  # a forked worker starts at the parent's address space

    def convert(key):
        try:
            blob = bytearray(1 << 30)
            return (key, "ceiling_not_enforced", "", len(blob))
        except MemoryError:
            pass
        r = bc._run_child(
            [sys.executable, "-c", "x = bytearray(1 << 30); print('child ok')"],
            capture_output=True,
            text=True,
        )  # a child allocating the same 1 GiB is fine
        try:
            bytearray(1 << 30)  # and the worker's own ceiling is back afterwards
            return (key, "ceiling_lost_after_child", "", 0)
        except MemoryError:
            return (
                key,
                "error:MemoryError"
                if "child ok" in r.stdout
                else "child_limited:" + r.stderr[-80:],
                "",
                0,
            )

    got = []
    with patched(bc, "convert", convert):
        left = bc.run_pool(["AAAA1111"], 1, {"worker_mem_gib": cap}, got.append)
    assert not left and got == [("AAAA1111", "error:MemoryError", "", 0)], got
    got = []
    with patched(bc, "convert", convert):  # 0 = no ceiling (the old behaviour)
        bc.run_pool(["AAAA1111"], 1, {"worker_mem_gib": 0}, got.append)
    assert got[0][1] == "ceiling_not_enforced", got


def test_workers_raise_their_own_oom_score():
    def convert(key):
        return (key, open("/proc/self/oom_score_adj").read().strip(), "", 0)

    got = []
    with patched(bc, "convert", convert):
        bc.run_pool(["AAAA1111"], 1, {}, got.append)
    assert got[0][1] == "1000", got


def _alive(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            return "State:\tZ" not in f.read()
    except OSError:
        return False


def test_sigterm_mid_pool_terminates_the_workers_and_still_stops_the_ocr_server():
    with tempfile.TemporaryDirectory() as td, stop_counter() as stops:
        pids = os.path.join(td, "pids")
        os.makedirs(pids)

        def convert(key):
            open(os.path.join(pids, str(os.getpid())), "w").close()
            time.sleep(60)  # a long conversion
            return (key, "pdf", "", 1)

        # SIGTERM 1.5 s in, from an interval timer rather than a thread: forking the workers while a
        # thread is alive draws a DeprecationWarning, and forked children do not inherit the timer
        old_alarm = signal.signal(
            signal.SIGALRM, lambda *a: os.kill(os.getpid(), signal.SIGTERM)
        )
        signal.setitimer(signal.ITIMER_REAL, 1.5)
        t0 = time.time()
        try:
            with patched(bc, "convert", convert):
                code, out, rows = run_main(td, ["A%07d" % i for i in range(1, 9)])
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        assert code == 128 + signal.SIGTERM, code
        assert time.time() - t0 < 30, "the build waited for the 60 s conversions"
        assert stops == [1], stops
        workers = [int(n) for n in os.listdir(pids)]
        assert len(workers) == 2, workers
        deadline = time.time() + 10
        while time.time() < deadline and any(_alive(p) for p in workers):
            time.sleep(0.1)
        assert not any(_alive(p) for p in workers), [p for p in workers if _alive(p)]


def test_a_worker_that_dies_holding_no_marker_convicts_nobody():
    """Two pool deaths, each with a healthy slow document in flight on the other worker and a victim that never wrote
    a marker: an earlier strike rule recorded the innocent slow document at the second death."""
    with tempfile.TemporaryDirectory() as td:
        slow = "SLOW0KEY1"
        real_tracked = bc._convert_tracked

        def convert(key):
            if key == slow:
                time.sleep(2)  # in flight, on the surviving worker, at both deaths
            return (key, "pdf", "", 1)

        def tracked(key):
            if (
                key == SENTINEL
                and len([n for n in os.listdir(td) if n.startswith("died_")]) < 2
            ):
                time.sleep(0.5)
                open(os.path.join(td, "died_%d" % os.getpid()), "w").close()
                # before any marker is written: killed "between documents"
                os.kill(os.getpid(), signal.SIGKILL)
            return real_tracked(key)

        # pickled by reference: the forked worker finds this one through the patched module attribute
        tracked.__name__ = tracked.__qualname__ = "_convert_tracked"
        tracked.__module__ = bc.__name__
        keys = [SENTINEL, slow] + ["A%07d" % i for i in range(1, 7)]
        with patched(bc, "convert", convert), patched(bc, "_convert_tracked", tracked):
            code, out, rows = run_main(td, keys)
        by = {r["key"]: r["method"] for r in rows}
        assert code is None and sorted(by) == sorted(keys) and len(rows) == len(keys), (
            code,
            out[-300:],
        )
        assert all(m == "pdf" for m in by.values()), (
            by
        )  # everything completed on the rebuilt pools
        assert not os.path.exists(os.path.join(td, "oom.tsv")), open(
            os.path.join(td, "oom.tsv")
        ).read()
        assert out.count("recorded: none") == 2, out


def test_a_survivor_in_flight_at_a_death_is_retried_not_recorded():
    with tempfile.TemporaryDirectory() as td:
        slow = "SLOW0KEY1"
        flag = os.path.join(td, "slow_started")

        def convert(key):
            if key == SENTINEL:
                # die only once the slow key is in flight on the other worker
                while not os.path.exists(flag):
                    time.sleep(0.05)
                time.sleep(0.2)
                os.kill(os.getpid(), signal.SIGKILL)
            if key == slow and not os.path.exists(flag + ".done"):
                open(flag, "w").close()
                time.sleep(2)
                open(flag + ".done", "w").close()
            return (key, "pdf", "", 1)

        with patched(bc, "convert", convert):
            code, out, rows = run_main(td, [SENTINEL, slow, "ZZZZ9990"])
        by = {r["key"]: r["method"] for r in rows}
        assert code is None and by == {
            SENTINEL: "error:worker_killed",
            slow: "pdf",
            "ZZZZ9990": "pdf",
        }, (by, out[-300:])
        assert open(os.path.join(td, "oom.tsv")).read().count("\n") == 1


def make_pdf(path, pages):
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        for i, line in enumerate(lines):
            page.insert_text((20, 20 + 5 * i), line, fontsize=4)
    doc.save(path)
    doc.close()


def test_memoryerror_is_recorded_and_never_sent_to_ocr():
    with tempfile.TemporaryDirectory() as td:

        def convert(key):
            return (key, "error:MemoryError", "", 0)

        with patched(bc, "convert", convert):
            code, out, rows = run_main(td, [SENTINEL])
        assert code is None and rows[0]["method"] == "error:MemoryError", (code, rows)
        assert open(os.path.join(td, "oom.tsv")).read().startswith(SENTINEL + "\t")
        assert "memory ceiling" in out, out
    # pdf_to_md lets a MemoryError through instead of "falling back to OCR" (which would repeat the blow-up, uncapped)
    with tempfile.TemporaryDirectory() as td:
        pdf = os.path.join(td, "a.pdf")
        make_pdf(pdf, [["line %d of an invented page" % i for i in range(5)]])

        def raising(path, **kw):
            raise MemoryError("Unable to allocate 499. MiB for an array")

        saved = sys.modules.get("pymupdf4llm")
        sys.modules["pymupdf4llm"] = types.SimpleNamespace(to_markdown=raising)
        try:
            try:
                bc.pdf_to_md(pdf)
            except MemoryError:
                pass
            else:
                raise AssertionError("MemoryError was swallowed")
        finally:
            if saved is None:
                del sys.modules["pymupdf4llm"]
            else:
                sys.modules["pymupdf4llm"] = saved


def test_oom_keys_file_expires_and_skips_in_convert():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "oom.tsv")
        bc.record_oom_key(path, "NEWKEY01", "test")
        with open(path, "a") as f:
            f.write("OLDKEY01\t2020-01-01 00:00:00\told\n# comment\n\nbad line\n")
        assert bc.load_oom_keys(path) == {"NEWKEY01"}
        assert bc.load_oom_keys(path, max_age_days=1e6) == {"NEWKEY01", "OLDKEY01"}
        assert bc.load_oom_keys(os.path.join(td, "missing")) == set()
        folder = os.path.join(td, "z", "storage", "NEWKEY01")
        os.makedirs(folder)
        open(os.path.join(folder, "a.pdf"), "wb").write(b"%PDF-1.4 not read")
        bc.CFG.clear()
        bc.CFG.update(
            {
                "meta": {},
                "zotero": os.path.join(td, "z"),
                "out": td,
                "deny": set(),
                "force": True,
                "min_chars": 200,
                "ocr": "off",
                "ocr_vlm": "off",
                "held": None,
                "oom_skip": {"NEWKEY01"},
            }
        )
        assert bc.convert("NEWKEY01")[1] == "oom_skip"


# ---- pages too dense for the layout model ----------------------------------------------------------------------
@contextlib.contextmanager
def fake_llm_recording(calls):
    """pymupdf4llm.to_markdown that honours pages= and writes "[engine p<N>]" for each page it was given."""

    def to_markdown(path, page_chunks=False, use_ocr=True, pages=None, **kw):
        calls.append(None if pages is None else list(pages))
        with pymupdf.open(path) as doc:
            nums = range(doc.page_count) if pages is None else pages
            return [
                {
                    "text": f"[engine p{p + 1}] "
                    + "engine text for the page. " * 12
                    + "\n\n",
                    "metadata": {"page_number": p + 1},
                }
                for p in nums
            ]

    saved = sys.modules.get("pymupdf4llm")
    sys.modules["pymupdf4llm"] = types.SimpleNamespace(to_markdown=to_markdown)
    try:
        yield
    finally:
        if saved is None:
            del sys.modules["pymupdf4llm"]
        else:
            sys.modules["pymupdf4llm"] = saved


def dense_page(tag, n):
    return [
        f"{tag} entry {i} of an invented printed listing with some words in it"
        for i in range(n)
    ]


def test_a_dense_page_skips_the_layout_model_and_is_read_from_the_text_layer():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a.pdf")
        make_pdf(
            path,
            [
                dense_page("alpha", 12),
                dense_page("OMEGA", 150),
                dense_page("gamma", 12),
                dense_page("DELTA", 160),
            ],
        )
        bc.CFG.clear()
        bc.CFG.update({"ocr": "off", "ocr_vlm": "off"})
        calls, log = [], io.StringIO()
        with (
            patched(bc, "LAYOUT_MAX_BOXES", 100),
            fake_llm_recording(calls),
            contextlib.redirect_stdout(log),
        ):
            md = bc.pdf_to_md(path)
        assert calls == [[0, 2]], calls  # the model never saw pages 2 and 4
        assert "LAYOUT-SKIP" in log.getvalue() and "p. 2, 4" in log.getvalue(), (
            log.getvalue()
        )
        order = [
            md.index(t)
            for t in ("[engine p1]", "OMEGA entry 0", "[engine p3]", "DELTA entry 0")
        ]
        assert order == sorted(order), order  # still in page order
        assert (
            "OMEGA entry 149" in md and "DELTA entry 159" in md
        )  # the whole text layer, not a stub
        # not "repaired": the engine never saw those pages
        assert bc._pdf_method == "", bc._pdf_method
        assert "TEXTLAYER" not in log.getvalue(), log.getvalue()


def test_a_page_of_vector_paths_skips_the_layout_model_too():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a.pdf")
        make_pdf(
            path,
            [dense_page("alpha", 12), dense_page("beta", 12), dense_page("gamma", 12)],
        )
        # page 2 becomes a plot: 4,000 short lines, ~100 KB of content
        doc = pymupdf.open(path)
        page = doc[1]
        for i in range(4000):
            page.draw_line((10 + i % 500, 300), (12 + i % 500, 310 + i % 7))
        doc.save(path + "2", garbage=0)
        doc.close()
        bc.CFG.clear()
        bc.CFG.update({"ocr": "off", "ocr_vlm": "off"})
        calls, log = [], io.StringIO()
        with (
            patched(bc, "LAYOUT_MAX_CONTENT_BYTES", 50_000),
            fake_llm_recording(calls),
            contextlib.redirect_stdout(log),
        ):
            md = bc.pdf_to_md(path + "2")
        assert calls == [[0, 2]], calls
        assert "LAYOUT-SKIP" in log.getvalue() and "p. 2" in log.getvalue(), (
            log.getvalue()
        )
        assert (
            md.index("[engine p1]") < md.index("beta entry 0") < md.index("[engine p3]")
        ), md[:200]
        calls = []
        with (
            patched(bc, "LAYOUT_MAX_CONTENT_BYTES", 0),
            fake_llm_recording(calls),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            bc.pdf_to_md(path + "2")
        assert calls == [None], calls


def test_heavy_page_order_holds_against_the_real_pymupdf4llm():
    import pymupdf4llm as real

    assert real.to_markdown.__module__ == "pymupdf4llm", (
        real.to_markdown
    )  # the real engine, not a fake left in sys.modules
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a.pdf")
        doc = pymupdf.open()
        for tag, n in (("FIRST", 8), ("HEAVY", 80), ("THIRD", 8), ("FOURTH", 8)):
            page = doc.new_page(width=595, height=842)
            for i in range(n):
                page.insert_text(
                    (50, 60 + (9 if n > 20 else 40) * i),
                    f"{tag} paragraph line {i} of an invented text",
                    fontsize=7 if n > 20 else 12,
                )
        doc.save(path)
        doc.close()
        bc.CFG.clear()
        bc.CFG.update({"ocr": "off", "ocr_vlm": "off"})
        t0, log = time.time(), io.StringIO()
        with patched(bc, "LAYOUT_MAX_BOXES", 40), contextlib.redirect_stdout(log):
            md = bc.pdf_to_md(path)
        assert "LAYOUT-SKIP" in log.getvalue() and "p. 2" in log.getvalue(), (
            log.getvalue()
        )
        order = [
            md.index(t)
            for t in (
                "FIRST paragraph line 0",
                "HEAVY paragraph line 0",
                "THIRD paragraph line 0",
                "FOURTH paragraph line 0",
            )
        ]
        assert order == sorted(order), (order, md[:300])
        assert "HEAVY paragraph line 79" in md
        assert time.time() - t0 < 60


def test_chunks_without_page_numbers_turn_the_skip_off_for_that_document():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a.pdf")
        make_pdf(
            path,
            [
                dense_page("alpha", 12),
                dense_page("OMEGA", 150),
                dense_page("gamma", 12),
            ],
        )
        calls = []

        # a legacy-style engine: no page_number
        def to_markdown(p, page_chunks=False, use_ocr=True, pages=None, **kw):
            calls.append(None if pages is None else list(pages))
            with pymupdf.open(p) as d:
                nums = list(range(3)) if pages is None else list(pages)
                return [
                    {
                        "text": f"LEGACYCHUNK{n + 1} " + d[n].get_text("text") + "\n\n",
                        "metadata": {"page": n + 1},
                    }
                    for n in nums
                ]

        saved = sys.modules.get("pymupdf4llm")
        sys.modules["pymupdf4llm"] = types.SimpleNamespace(to_markdown=to_markdown)
        bc.CFG.clear()
        bc.CFG.update({"ocr": "off", "ocr_vlm": "off"})
        log = io.StringIO()
        try:
            with patched(bc, "LAYOUT_MAX_BOXES", 100), contextlib.redirect_stdout(log):
                md = bc.pdf_to_md(path)
        finally:
            if saved is None:
                del sys.modules["pymupdf4llm"]
            else:
                sys.modules["pymupdf4llm"] = saved
        # tried with the subset, then the whole document
        assert calls == [[0, 2], None], calls
        assert "carry no page_number -> skip disabled" in log.getvalue(), log.getvalue()
        assert [md.index(f"LEGACYCHUNK{i}") for i in (1, 2, 3)] == sorted(
            md.index(f"LEGACYCHUNK{i}") for i in (1, 2, 3)
        )
        # page 2 came from the engine, once
        assert md.count("OMEGA entry 149") == 1, md.count("OMEGA entry 149")


def test_a_page_that_cannot_be_measured_is_heavy():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a.pdf")
        make_pdf(path, [dense_page("a", 5), dense_page("b", 5), dense_page("c", 5)])
        real = pymupdf.Page.read_contents

        def read_contents(self):
            if self.number == 1:
                raise MemoryError("page 2")
            return real(self)

        with (
            patched(pymupdf.Page, "read_contents", read_contents),
            pymupdf.open(path) as doc,
        ):
            assert sorted(bc._heavy_pages(doc)) == [1]


def test_the_ocr_layer_share_is_a_share_of_the_documents_pages():
    with tempfile.TemporaryDirectory() as td:
        left = [
            "The osculating elements of the satellite are estimated from pseudorange data."
        ] * 8
        path = os.path.join(td, "a.pdf")
        # 2 OCR-layer pages + 3 skipped pages
        make_pdf(path, [left, left] + [dense_page("D", 150)] * 3)
        font = None
        with pymupdf.open(path) as d:
            font = d[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["font"]
        calls = []

        def fake_vlm(p):
            calls.append(p)
            return "Invented transcription of the scan by the vision model. " * 20

        bc.CFG.clear()
        bc.CFG.update({"ocr": "auto", "ocr_vlm": "auto"})
        with (
            patched(bc, "LAYOUT_MAX_BOXES", 100),
            patched(bc, "OCR_LAYER_FONTS", {font}),
            patched(bc, "vlm_ocr_pdf_to_md", fake_vlm),
            fake_llm_recording([]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            bc.pdf_to_md(path)
        # 2 of 5 pages is a minority, as without the skip
        assert calls == [], calls
        path2 = os.path.join(td, "b.pdf")
        make_pdf(path2, [left, left])
        with (
            patched(bc, "OCR_LAYER_FONTS", {font}),
            patched(bc, "vlm_ocr_pdf_to_md", fake_vlm),
            fake_llm_recording([]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            bc.pdf_to_md(path2)
        # 2 of 2 pages: the majority route, unchanged
        assert calls == [path2], calls


def test_no_dense_page_means_the_old_call():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a.pdf")
        make_pdf(path, [dense_page("alpha", 12), dense_page("beta", 12)])
        bc.CFG.clear()
        bc.CFG.update({"ocr": "off", "ocr_vlm": "off"})
        calls = []
        with (
            patched(bc, "LAYOUT_MAX_BOXES", 100),
            fake_llm_recording(calls),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            md = bc.pdf_to_md(path)
        assert calls == [None] and "[engine p1]" in md and "[engine p2]" in md, (
            calls,
            md[:80],
        )
        assert bc._pdf_method == ""


def test_every_page_dense_and_the_limit_switched_off():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a.pdf")
        make_pdf(path, [dense_page("ONE", 150), dense_page("TWO", 150)])
        bc.CFG.clear()
        bc.CFG.update({"ocr": "off", "ocr_vlm": "off"})
        calls = []
        with (
            patched(bc, "LAYOUT_MAX_BOXES", 100),
            fake_llm_recording(calls),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            md = bc.pdf_to_md(path)
        # the model is not called at all
        assert calls == [] and md.index("ONE entry 0") < md.index("TWO entry 0"), calls
        calls = []
        with (
            patched(bc, "LAYOUT_MAX_BOXES", 0),
            fake_llm_recording(calls),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            bc.pdf_to_md(path)
        assert calls == [None], calls


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
