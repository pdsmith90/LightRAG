"""Offline tests for citegraph/refresh.sh and citegraph/load_zstruct.sh.

Each test runs the real scripts from a temp directory with stub `ssh`, `rsync`, `psql` and `pgrep` on PATH and a
CITEGRAPH_CODE directory of stub builder and sweep modules that write canned outputs. Nothing leaves the temp
directory: the stubs only write call logs there.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CITEGRAPH = Path(__file__).resolve().parents[2] / "citegraph"
REFRESH = CITEGRAPH / "refresh.sh"
LOAD = CITEGRAPH / "load_zstruct.sh"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or shutil.which("flock") is None
    or shutil.which("realpath") is None,
    reason="needs util-linux flock and realpath",
)

QA = {
    "hard_checks": {"passed": True},
    "oracle": {"passed": True},
    "row_counts": {
        "papers": 3,
        "cites": 2,
        "cite_contexts": 1,
        "external_works": 1,
        "paper_embeddings": 3,
    },
}
LIVE = "papers 3\ncites 2\ncite_contexts 1\nexternal_works 1\npaper_embeddings 3\n"

STUB_BUILDER = """\
import json, os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
os.makedirs(out, exist_ok=True)
stub = os.environ["STUB_DIR"]
with open(os.path.join(stub, "builder_calls.log"), "a") as f:
    f.write(" ".join(["offline" if "--offline" in a else "online", "schema=" + a[a.index("--schema") + 1],
                      "key=" + ("yes" if os.environ.get("OPENALEX_API_KEY") else "no")]) + "\\n")
tag = os.environ.get("STUB_CSV_TAG", "A")
for name in ("papers.csv", "cites.csv"):
    with open(os.path.join(out, name), "w") as f:
        f.write(name + "," + tag + "\\n")
with open(os.path.join(out, "qa.json"), "w") as f:
    f.write(open(os.path.join(stub, "qa.json")).read())
print("stub builder wrote", out)
sys.exit(int(os.environ.get("STUB_BUILD_RC", "0")))
"""
STUB_SWEEP = """\
import os, sys
with open(os.path.join(os.environ["STUB_DIR"], "sweep_calls.log"), "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
print("sweep: 4 papers visited")
sys.exit(int(os.environ.get("STUB_SWEEP_RC", "0")))
"""
STUB_SSH = """\
#!/usr/bin/env bash
# the remote command is the last argument; stdin is the SQL of a count call
echo "ssh $*" >> "$STUB_DIR/ssh.log"
cmd="${*: -1}"
case "$cmd" in
  *"psql -X -At"*)
    cat > "$STUB_DIR/ssh_psql_stdin.sql"
    [ -z "${STUB_COUNT_FAIL:-}" ] || exit 1
    cat "$STUB_DIR/live_counts.txt" ;;
  *load_zstruct.sh*)
    [ -z "${STUB_LOAD_FAIL:-}" ] || { echo "12:00:00 FATAL: stub load failed"; exit 1; }
    [ -z "${STUB_LOAD_NO_MARKER:-}" ] || { echo "stub load exits 0 without a marker"; exit 0; }
    echo "12:00:00 LOADED: schema zstruct = 7 tables; papers 3, cites 2. zstruct_prev: 1 (stub)" ;;
esac
"""
STUB_RSYNC = """\
#!/usr/bin/env bash
echo "rsync $*" >> "$STUB_DIR/rsync.log"
src="${*: -2:1}"; dest="${*: -1}"
case "$src" in
  *:*) cp "$STUB_DIR/remote_files/${src##*/}" "$dest" ;;
  *) case "$dest" in *:*) echo "PUSH $dest" >> "$STUB_DIR/rsync.log" ;; esac ;;
esac
"""
STUB_PGREP = """\
#!/usr/bin/env bash
echo "pgrep $*" >> "$STUB_DIR/pgrep.log"
[ -n "${STUB_SWEEP_RUNNING:-}" ]
"""
STUB_PSQL = """\
#!/usr/bin/env bash
args="$*"
echo "psql $args" >> "$STUB_DIR/psql.log"
echo "env PGHOST=${PGHOST:-} PGUSER=${PGUSER:-} PGDATABASE=${PGDATABASE:-} PGPASSWORD=${PGPASSWORD:-}" >> "$STUB_DIR/psql.log"
case "$args" in
  *"-f -"*) cat > "$STUB_DIR/psql_stdin.sql"; cat "$STUB_DIR/live_counts.txt" ;;
  *"-f "*) : ;;
  *"DROP SCHEMA"*) : ;;
  *pg_extension*) echo "${STUB_PGVECTOR:-1}" ;;
  *"<=>"*) echo "${STUB_SIMILAR:-10 rows, top 0.9123}" ;;
  *'FROM zstruct_next."'*)
    t=$(sed -E 's/.*zstruct_next\\."([a-z_]+)".*/\\1/' <<< "$args")
    grep "^$t " "$STUB_DIR/db_counts.txt" | cut -d' ' -f2 ;;
  *pg_tables*) echo 7 ;;
  *"FROM zstruct.papers"*) echo 3 ;;
  *"FROM zstruct.cites"*) echo 2 ;;
  *pg_namespace*) echo 1 ;;
esac
"""


def script(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


class Site:
    """A temp site: data dir, stub code dir, stub binaries, config file."""

    def __init__(self, root: Path, remote: str = "dbhost"):
        self.root = root
        self.data = root / "data"
        self.code = root / "code"
        self.stub = root / "stub"
        self.bin = root / "bin"
        for d in (
            self.data,
            self.code,
            self.stub,
            self.bin,
            self.stub / "remote_files",
        ):
            d.mkdir(parents=True)
        (self.code / "build_struct_layer.py").write_text(STUB_BUILDER)
        (self.code / "s2_citation_sweep.py").write_text(STUB_SWEEP)
        script(self.bin / "ssh", STUB_SSH)
        script(self.bin / "rsync", STUB_RSYNC)
        script(self.bin / "pgrep", STUB_PGREP)
        script(self.bin / "psql", STUB_PSQL)
        (self.bin / "python3").symlink_to(sys.executable)
        (self.stub / "qa.json").write_text(json.dumps(QA))
        (self.stub / "live_counts.txt").write_text(LIVE)
        (self.stub / "db_counts.txt").write_text(
            "".join(f"{t} {n}\n" for t, n in QA["row_counts"].items())
        )
        (self.stub / "remote_files" / "zotero_metadata.json").write_text("{}")
        (self.stub / "remote_files" / "zotero_references.json").write_text("{}")
        self.pwfile = root / "pw"
        self.pwfile.write_text("dummy-password\n")
        self.notified = root / "notified.txt"
        self.remote = remote
        self.config = root / "citegraph.env"
        self.write_config()

    def write_config(self, extra: str = "", remote: str | None = None) -> None:
        remote = self.remote if remote is None else remote
        lines = [
            f"CITEGRAPH_DATA={self.data}",
            f"CITEGRAPH_CODE={self.code}",
            "CITEGRAPH_METADATA_SRC=srchost:/src/zotero_metadata.json",
            "CITEGRAPH_REFERENCES_SRC=srchost:/src/zotero_references.json",
            f"CITEGRAPH_REMOTE={remote}",
            f"CITEGRAPH_REMOTE_CODE={CITEGRAPH}",
            f"CITEGRAPH_REMOTE_CONFIG={self.config}",
            f"CITEGRAPH_REMOTE_DATA={self.root / 'remote-data'}",
            "PGHOST=dbserver",
            "PGUSER=cguser",
            "PGDATABASE=cgdb",
            f"CITEGRAPH_PGPASSWORD_FILE={self.pwfile}",
            'citegraph_with_key() { OPENALEX_API_KEY=dummy-key "$@"; }',
            f'citegraph_notify() {{ echo "$1" >> {self.notified}; }}',
            extra,
        ]
        self.config.write_text("\n".join(lines) + "\n")

    def env(self, **extra):
        env = dict(
            os.environ,
            PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            STUB_DIR=str(self.stub),
        )
        env.pop("CITEGRAPH_CONFIG", None)
        env.update(extra)
        return env

    def run(self, script_path: Path, *args: str, **env):
        return subprocess.run(
            ["bash", str(script_path), *args],
            capture_output=True,
            text=True,
            input="",
            env=self.env(**env),
            cwd=self.root,
        )

    def refresh(self, *args: str, **env):
        return self.run(REFRESH, "--config", str(self.config), *args, **env)

    def log_lines(self):
        path = self.data / "logs" / f"refresh_{datetime.date.today():%Y%m}.log"
        return path.read_text().splitlines() if path.exists() else []

    def stub_log(self, name: str) -> str:
        path = self.stub / name
        return path.read_text() if path.exists() else ""

    def make_loaded(self, marker="old"):
        d = self.data / "out.loaded"
        d.mkdir()
        (d / "papers.csv").write_text(marker + "\n")
        (d / "cites.csv").write_text(marker + "\n")
        return d


@pytest.fixture
def site(tmp_path):
    return Site(tmp_path)


def strip_stamp(line: str) -> str:
    """Drop the 'YYYY-MM-DD HH:MM:SS ' prefix of a log line."""
    return line.split(" ", 2)[2]


# --- both scripts refuse to run without a config ----------------------------------------------------------------


@pytest.mark.parametrize("which", [REFRESH, LOAD])
class TestConfigRequired:
    def test_refuses_without_a_config(self, site, which):
        proc = site.run(which)
        assert proc.returncode == 2 and "no config" in proc.stderr
        assert not (site.data / "logs").exists()

    def test_refuses_an_unreadable_config(self, site, which):
        proc = site.run(which, "--config", str(site.root / "nope.env"))
        assert proc.returncode == 2 and "cannot read the config" in proc.stderr

    def test_config_from_the_environment(self, site, which):
        proc = site.run(which, CITEGRAPH_CONFIG=str(site.root / "nope.env"))
        assert proc.returncode == 2 and "cannot read the config" in proc.stderr

    def test_refuses_a_config_without_the_data_dir(self, site, which):
        site.config.write_text("PGHOST=x\n")
        proc = site.run(which, "--config", str(site.config))
        assert proc.returncode == 2 and "CITEGRAPH_DATA is not set" in proc.stderr

    def test_refuses_an_unknown_option(self, site, which):
        proc = site.run(which, "--config", str(site.config), "--bogus")
        assert proc.returncode == 2 and "usage" in proc.stderr


def test_refresh_refuses_a_remote_without_its_paths(site):
    site.config.write_text(
        f"CITEGRAPH_DATA={site.data}\nCITEGRAPH_METADATA_SRC=a:b\nCITEGRAPH_REFERENCES_SRC=a:c\nCITEGRAPH_REMOTE=dbhost\n"
    )
    proc = site.refresh("--dry-run")
    assert proc.returncode == 2 and "CITEGRAPH_REMOTE_CODE is not set" in proc.stderr


# --- load_zstruct.sh --------------------------------------------------------------------------------------------


def window_excluding_now() -> str:
    return "1300-1400" if datetime.datetime.now().hour < 12 else "0100-0200"


def make_build(site, name="out.daily", counts=None):
    d = site.data / name
    d.mkdir(parents=True, exist_ok=True)
    qa = dict(QA, row_counts=counts or QA["row_counts"])
    (d / "qa.json").write_text(json.dumps(qa))
    for f in ("apply_zstruct.sql", "swap_zstruct.sql", "papers.csv", "cites.csv"):
        (d / f).write_text("-- stub\n")
    return d


class TestLoad:
    def load(self, site, *args, **env):
        return site.run(LOAD, "--config", str(site.config), *args, **env)

    @pytest.mark.parametrize(
        "bad", ["../out", "outside/x", "out;ls", "data", "/tmp/out", "out x"]
    )
    def test_refuses_a_bad_dir(self, site, bad):
        proc = self.load(site, bad)
        assert proc.returncode == 1 and "DIR must be a build folder name" in proc.stderr
        assert site.stub_log("psql.log") == ""

    def test_refuses_a_missing_build_file(self, site):
        d = make_build(site)
        (d / "cites.csv").unlink()
        proc = self.load(site, "out.daily")
        assert (
            proc.returncode == 1
            and "FATAL: missing" in (proc.stdout + proc.stderr)
            and "cites.csv" in (proc.stdout + proc.stderr)
        )
        assert site.stub_log("psql.log") == ""

    def test_refuses_an_empty_build_file(self, site):
        d = make_build(site)
        (d / "papers.csv").write_text("")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "FATAL: missing" in proc.stdout + proc.stderr

    def test_refuses_outside_its_window(self, site):
        make_build(site)
        site.write_config(f"CITEGRAPH_LOAD_WINDOW={window_excluding_now()}")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "run between" in (proc.stdout + proc.stderr)
        assert site.stub_log("psql.log") == ""

    def test_refuses_a_malformed_window(self, site):
        make_build(site)
        site.write_config("CITEGRAPH_LOAD_WINDOW=8:45-23")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "CITEGRAPH_LOAD_WINDOW must look like" in (
            proc.stdout + proc.stderr
        )

    def test_no_window_means_any_time(self, site):
        make_build(site)
        assert self.load(site, "out.daily").returncode == 0

    def test_refuses_failed_hard_checks(self, site):
        d = make_build(site)
        (d / "qa.json").write_text(json.dumps(dict(QA, hard_checks={"passed": False})))
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "hard checks did not pass" in (
            proc.stdout + proc.stderr
        )

    def test_refuses_without_pgvector(self, site):
        make_build(site)
        proc = self.load(site, "out.daily", STUB_PGVECTOR="0")
        assert proc.returncode == 1 and "pgvector is not installed" in (
            proc.stdout + proc.stderr
        )
        assert "-f apply_zstruct.sql" not in site.stub_log("psql.log")

    def test_loads_and_swaps(self, site):
        make_build(site)
        site.write_config("CITEGRAPH_LOAD_WINDOW=0000-2400")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (
            "papers: 3 ok" in proc.stdout
            and "similar (pgvector): 10 rows, top 0.9123" in proc.stdout
        )
        assert "LOADED: schema zstruct = 7 tables; papers 3, cites 2." in proc.stdout
        calls = site.stub_log("psql.log")
        assert "-f apply_zstruct.sql" in calls and "swap_zstruct.sql" in calls
        assert "DROP SCHEMA" not in calls
        assert (
            "env PGHOST=dbserver PGUSER=cguser PGDATABASE=cgdb PGPASSWORD=dummy-password"
            in calls
        )
        assert (site.data / "logs" / "load_zstruct.log").read_text().count(
            "LOADED:"
        ) == 1

    def test_the_schema_name_comes_from_the_config(self, site):
        d = make_build(site)
        (d / "apply_cg.sql").write_text("-- stub\n")
        (d / "swap_cg.sql").write_text("-- stub\n")
        site.write_config("CITEGRAPH_SCHEMA=cg")
        script(site.bin / "psql", STUB_PSQL.replace("zstruct", "cg"))
        proc = self.load(site, "out.daily")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (
            "LOADED: schema cg = 7 tables" in proc.stdout and "cg_prev:" in proc.stdout
        )
        assert "apply_cg.sql" in site.stub_log("psql.log")

    def test_refuses_a_bad_schema_name(self, site):
        make_build(site)
        site.write_config("CITEGRAPH_SCHEMA='x; drop'")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 2 and "CITEGRAPH_SCHEMA" in proc.stderr

    def test_a_row_count_mismatch_drops_the_new_schema_and_does_not_swap(self, site):
        make_build(site)
        (site.stub / "db_counts.txt").write_text(
            "papers 2\ncites 2\ncite_contexts 1\nexternal_works 1\npaper_embeddings 3\n"
        )
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "row counts differ from qa.json" in (
            proc.stdout + proc.stderr
        )
        assert "papers: 2 in the DB, 3 in qa.json" in proc.stdout
        calls = site.stub_log("psql.log")
        assert (
            "DROP SCHEMA IF EXISTS zstruct_next CASCADE" in calls
            and "swap_zstruct.sql" not in calls
        )

    def test_a_bad_similar_result_drops_the_new_schema(self, site):
        make_build(site)
        proc = self.load(site, "out.daily", STUB_SIMILAR="3 rows, top none")
        assert proc.returncode == 1 and "expected 10 rows" in (
            proc.stdout + proc.stderr
        )
        calls = site.stub_log("psql.log")
        assert (
            "DROP SCHEMA IF EXISTS zstruct_next CASCADE" in calls
            and "swap_zstruct.sql" not in calls
        )

    def test_refuses_an_unreadable_password_file(self, site):
        make_build(site)
        site.pwfile.unlink()
        proc = self.load(site, "out.daily")
        assert proc.returncode == 2 and "CITEGRAPH_PGPASSWORD_FILE" in proc.stderr

    def test_a_leading_tilde_in_the_password_file_means_this_hosts_home(self, site):
        make_build(site)
        site.write_config("CITEGRAPH_PGPASSWORD_FILE='~/.pw'")
        (site.root / ".pw").write_text("tilde-password\n")
        proc = self.load(site, "out.daily", HOME=str(site.root))
        assert proc.returncode == 0, proc.stderr
        assert "PGPASSWORD=tilde-password" in site.stub_log("psql.log")


# --- refresh.sh -------------------------------------------------------------------------------------------------


class TestRefreshDryRun:
    def test_end_to_end_copies_and_loads_nothing(self, site):
        old = site.make_loaded()
        proc = site.refresh("--dry-run")
        assert proc.returncode == 0, proc.stderr
        lines = [strip_stamp(ln) for ln in site.log_lines()]
        assert lines == [
            "[dry-run] start",
            "[dry-run] online build rc=0: stub builder wrote out.daily.dryrun",
            "[dry-run] build rc=0: stub builder wrote out.daily.dryrun",
            "[dry-run] gates passed; would copy out.daily.dryrun to the database host as out.daily and load it: "
            "papers=3 cites=2 cite_contexts=1 external_works=1 paper_embeddings=3",
        ]
        # no sweep, no copy to the database host, no load; the one ssh call is the read-only count
        assert site.stub_log("sweep_calls.log") == ""
        rsync = site.stub_log("rsync.log")
        assert "PUSH" not in rsync and "--delete" not in rsync
        assert (
            rsync.count("rsync -a -e") == 2
            and "srchost:/src/zotero_metadata.json input/" in rsync
        )
        ssh = site.stub_log("ssh.log").splitlines()
        assert (
            len(ssh) == 1 and "psql -X -At" in ssh[0] and "load_zstruct" not in ssh[0]
        )
        assert (
            "PGHOST=dbserver PGUSER=cguser PGDATABASE=cgdb PGPASSWORD=$(cat " in ssh[0]
        )
        assert site.stub_log("pgrep.log") == ""  # the sweep-running skip does not apply
        builds = site.stub_log("builder_calls.log").splitlines()
        assert builds == [
            "online schema=zstruct key=yes",
            "offline schema=zstruct key=no",
        ]
        sql = site.stub_log("ssh_psql_stdin.sql").splitlines()
        assert (
            sql[0] == "SELECT 'cite_contexts', count(*) FROM zstruct.\"cite_contexts\";"
            and len(sql) == 5
        )
        assert (site.data / "out.daily.dryrun" / "papers.csv").exists()
        assert (old / "papers.csv").read_text() == "old\n" and not (
            site.data / "out.loaded.prev"
        ).exists()
        assert not site.notified.exists()

    def test_dry_run_ignores_a_running_sweep(self, site):
        proc = site.refresh("--dry-run", STUB_SWEEP_RUNNING="1")
        assert proc.returncode == 0
        assert strip_stamp(site.log_lines()[-1]).startswith("[dry-run] gates passed;")

    def test_a_failing_dry_run_does_not_notify(self, site):
        qa = json.loads((site.stub / "qa.json").read_text())
        qa["hard_checks"]["passed"] = False
        (site.stub / "qa.json").write_text(json.dumps(qa))
        proc = site.refresh("--dry-run")
        assert proc.returncode == 1
        assert strip_stamp(site.log_lines()[-1]).startswith(
            "[dry-run] FAILED: the hard checks or the reference check"
        )
        assert not site.notified.exists()

    def test_a_dry_run_with_a_local_database_counts_with_a_local_psql(self, site):
        site.write_config(remote="")
        proc = site.refresh("--dry-run")
        assert proc.returncode == 0, proc.stderr
        assert strip_stamp(site.log_lines()[-1]).startswith("[dry-run] gates passed;")
        assert site.stub_log("ssh.log") == ""
        assert "PGPASSWORD=dummy-password" in site.stub_log("psql.log")
        assert site.stub_log("psql_stdin.sql").count("SELECT") == 5

    def test_the_password_file_is_read_on_the_database_host(self, site):
        site.write_config("CITEGRAPH_PGPASSWORD_FILE='~/.pw'")
        proc = site.refresh("--dry-run")
        assert proc.returncode == 0, proc.stderr
        ssh = site.stub_log("ssh.log")
        assert 'PGPASSWORD=$(cat "$HOME"/.pw) psql -X' in ssh
        assert "dummy-password" not in ssh  # only the file's name travels


class TestRefreshRealRun:
    def test_load_succeeds_and_out_loaded_rotates(self, site):
        old = site.make_loaded()
        (site.data / "out.loaded.prev").mkdir()
        (site.data / "out.loaded.prev" / "papers.csv").write_text("older\n")
        proc = site.refresh(STUB_CSV_TAG="NEW")
        assert proc.returncode == 0, proc.stderr
        lines = [strip_stamp(ln) for ln in site.log_lines()]
        assert lines[0] == "start" and lines[1].startswith("online build rc=0:")
        assert lines[2] == "sweep rc=0: sweep: 4 papers visited"
        assert lines[3].startswith("build rc=0:")
        assert (
            lines[4]
            == "LOADED: schema zstruct = 7 tables; papers 3, cites 2. zstruct_prev: 1 (stub)"
        )
        assert (
            lines[5]
            == "done: papers=3 cites=2 cite_contexts=1 external_works=1 paper_embeddings=3"
        )
        assert len(lines) == 6
        # the build went to the database host's out.daily with --delete, then loaded there under a 30 min limit
        rsync = site.stub_log("rsync.log")
        assert (
            f"PUSH dbhost:{site.root / 'remote-data'}/out.daily/" in rsync
            and "--delete" in rsync
        )
        load = [
            ln
            for ln in site.stub_log("ssh.log").splitlines()
            if "load_zstruct.sh" in ln
        ]
        assert (
            len(load) == 1
            and "timeout 1800 bash" in load[0]
            and load[0].endswith("out.daily")
        )
        sweep = site.stub_log("sweep_calls.log")
        assert "--store s2store/api_citations.sqlite --max-hours 3" in sweep
        # rotation: loaded -> prev, new build -> loaded, the one before prev is gone
        assert (
            site.data / "out.loaded" / "papers.csv"
        ).read_text() == "papers.csv,NEW\n"
        assert (site.data / "out.loaded.prev" / "papers.csv").read_text() == "old\n"
        assert (old / "cites.csv").read_text() == "cites.csv,NEW\n"
        assert not (site.data / "out.daily.new").exists()
        assert not site.notified.exists()

    def test_first_load_with_no_loaded_build(self, site):
        assert site.refresh().returncode == 0
        assert (site.data / "out.loaded" / "papers.csv").exists() and not (
            site.data / "out.loaded.prev"
        ).exists()

    def test_the_sweep_max_hours_comes_from_the_config(self, site):
        site.write_config("CITEGRAPH_SWEEP_MAX_HOURS=1")
        site.refresh()
        assert "--max-hours 1" in site.stub_log("sweep_calls.log")

    def test_an_identical_build_is_not_loaded_again(self, site):
        d = site.make_loaded()
        (d / "papers.csv").write_text(
            "papers.csv,A\n"
        )  # the bytes the stub builder writes by default
        (d / "cites.csv").write_text("cites.csv,A\n")
        proc = site.refresh()
        assert proc.returncode == 0
        assert (
            strip_stamp(site.log_lines()[-1])
            == "no change against the last loaded build (out.loaded): not loaded"
        )
        assert (
            "PUSH" not in site.stub_log("rsync.log")
            and not (site.data / "out.daily.new").exists()
        )
        assert not (site.data / "out.loaded.prev").exists()

    def test_a_failed_load_notifies_and_keeps_out_loaded(self, site):
        old = site.make_loaded()
        proc = site.refresh(STUB_LOAD_FAIL="1")
        assert proc.returncode == 1
        last = strip_stamp(site.log_lines()[-1])
        assert last.startswith(
            "FAILED: the load on the database host failed (rc=1, logs/refresh_load.log)"
        )
        assert (old / "papers.csv").read_text() == "old\n" and not (
            site.data / "out.loaded.prev"
        ).exists()
        note = site.notified.read_text()
        assert (
            "the load on the database host failed" in note
            and note.rstrip().endswith(
                f"-- log {site.data}/logs/refresh_{datetime.date.today():%Y%m}.log"
            )
        )

    def test_exit_zero_without_a_loaded_marker_is_a_failure(self, site):
        proc = site.refresh(STUB_LOAD_NO_MARKER="1")
        assert (
            proc.returncode == 1
            and "FAILED: the load on the database host failed (rc=0"
            in site.log_lines()[-1]
        )
        assert not (site.data / "out.loaded").exists()

    def test_a_shrinking_table_blocks_the_load(self, site):
        (site.stub / "live_counts.txt").write_text(
            "papers 100\ncites 2\ncite_contexts 1\nexternal_works 1\npaper_embeddings 3\n"
        )
        proc = site.refresh()
        assert proc.returncode == 1
        assert (
            "tables would shrink by more than 2% against the live zstruct: papers 100 -> 3 -- not loaded"
            in site.log_lines()[-1]
        )
        assert "PUSH" not in site.stub_log("rsync.log") and site.notified.exists()

    def test_a_failed_live_count_blocks_the_load(self, site):
        proc = site.refresh(STUB_COUNT_FAIL="1")
        assert proc.returncode == 1
        assert (
            "could not count the live zstruct tables on the database host -- not loaded"
            in site.log_lines()[-1]
        )

    def test_a_missing_reference_check_blocks_the_load(self, site):
        qa = dict(QA)
        del qa["oracle"]
        (site.stub / "qa.json").write_text(json.dumps(qa))
        proc = site.refresh()
        assert (
            proc.returncode == 1
            and "the hard checks or the reference check did not pass"
            in site.log_lines()[-1]
        )
        assert site.stub_log("ssh.log") == ""

    def test_an_online_build_without_papers_stops_the_run(self, site):
        (site.code / "build_struct_layer.py").write_text(
            "import sys\nprint('boom')\nsys.exit(1)\n"
        )
        proc = site.refresh()
        assert proc.returncode == 1
        lines = site.log_lines()
        assert (
            "online build rc=1: boom" in lines[1] and "wrote no papers.csv" in lines[2]
        )
        assert site.stub_log("sweep_calls.log") == ""

    def test_a_failed_input_copy_stops_the_run(self, site):
        (site.stub / "remote_files" / "zotero_references.json").unlink()
        proc = site.refresh()
        assert proc.returncode == 1
        assert (
            "FAILED: copying zotero_references.json from its source failed"
            in site.log_lines()[-1]
        )

    def test_a_bad_sweep_exit_notifies_and_the_run_goes_on(self, site):
        proc = site.refresh(STUB_SWEEP_RC="2")
        assert proc.returncode == 0
        assert "the citation sweep exited rc=2" in site.notified.read_text()
        assert strip_stamp(site.log_lines()[-1]).startswith("done:")

    def test_a_throttled_sweep_is_not_a_failure(self, site):
        assert site.refresh(STUB_SWEEP_RC="3").returncode == 0
        assert not site.notified.exists()

    def test_skips_while_a_sweep_runs(self, site):
        proc = site.refresh(STUB_SWEEP_RUNNING="1")
        assert proc.returncode == 0
        assert [strip_stamp(ln) for ln in site.log_lines()] == [
            "skip: a citation sweep is running"
        ]
        assert site.stub_log("rsync.log") == ""
        assert "[s]2_citation_sweep.py" in site.stub_log("pgrep.log")

    def test_the_sweep_process_pattern_comes_from_the_config(self, site):
        site.write_config(
            "CITEGRAPH_SWEEP_PROCESS_PATTERN='[s]weep_loop.sh|[s]2_citation_sweep.py'"
        )
        site.refresh(STUB_SWEEP_RUNNING="1")
        assert "[s]weep_loop.sh|[s]2_citation_sweep.py" in site.stub_log("pgrep.log")

    def test_skips_while_another_refresh_holds_the_lock(self, site):
        import fcntl

        (site.data / "logs").mkdir()
        with open(site.data / "logs" / ".refresh.lock", "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            proc = site.refresh("--dry-run")
        assert proc.returncode == 0
        assert [strip_stamp(ln) for ln in site.log_lines()] == [
            "[dry-run] skip: another refresh is running"
        ]
        assert site.stub_log("rsync.log") == ""
