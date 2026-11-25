"""Offline tests for citegraph/refresh.sh and citegraph/load_zstruct.sh.

Each test runs the real scripts from a temp directory with stub `ssh`, `rsync`, `psql`, `pgrep` and `tee` on PATH
and a CITEGRAPH_CODE directory of stub builder and sweep modules that write canned outputs. Nothing leaves the temp
directory: the stubs only write call logs there. Three kinds of run:

- refresh.sh with a canned `ssh` (it answers the live count and the load without running them), for the steps of
  refresh.sh itself;
- refresh.sh against the REAL load_zstruct.sh, with a stub psql that answers the catalog count query and the apply and
  swap steps: either a local database (CITEGRAPH_REMOTE empty) or a stub `ssh` that runs its command line with
  `bash -c`, so the quoting of the remote command is parsed by a real shell;
- load_zstruct.sh on its own.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

CITEGRAPH = Path(__file__).resolve().parents[2] / "citegraph"
REFRESH = CITEGRAPH / "refresh.sh"
LOAD = CITEGRAPH / "load_zstruct.sh"
REAL_TEE = shutil.which("tee")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or shutil.which("flock") is None
    or shutil.which("realpath") is None
    or REAL_TEE is None,
    reason="needs util-linux flock and realpath, and tee",
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
LIVE = "cite_contexts 1\ncites 2\nexternal_works 1\npaper_embeddings 3\npapers 3\n"

STUB_BUILDER = """\
import json, os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
schema = a[a.index("--schema") + 1]
offline = "--offline" in a
stub = os.environ["STUB_DIR"]
with open(os.path.join(stub, "builder_calls.log"), "a") as f:
    f.write(" ".join(["offline" if offline else "online", "schema=" + schema,
                      "key=" + ("yes" if os.environ.get("OPENALEX_API_KEY") else "no")]) + "\\n")
if offline and os.environ.get("STUB_OFFLINE_CRASH"):
    print("Traceback: the offline build crashed")
    sys.exit(1)
os.makedirs(out, exist_ok=True)
tag = os.environ.get("STUB_CSV_TAG", "A")
for name in ("papers.csv", "cites.csv"):
    with open(os.path.join(out, name), "w") as f:
        f.write(name + "," + tag + "\\n")
for name in ("apply_" + schema + ".sql", "swap_" + schema + ".sql"):
    with open(os.path.join(out, name), "w") as f:
        f.write("-- stub\\n")
with open(os.path.join(out, "qa.json"), "w") as f:
    f.write(open(os.path.join(stub, "qa.json")).read())
print("stub builder wrote", out)
"""
STUB_SWEEP = """\
import os, sys
with open(os.path.join(os.environ["STUB_DIR"], "sweep_calls.log"), "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
print("sweep: 4 papers visited")
sys.exit(int(os.environ.get("STUB_SWEEP_RC", "0")))
"""
# The remote command is the last argument. The canned ssh answers the two calls refresh.sh makes without running
# them; with STUB_SSH_REAL it runs the command line with bash -c, as an ssh server's bash login shell would.
STUB_SSH = """\
#!/usr/bin/env bash
echo "ssh $*" >> "$STUB_DIR/ssh.log"
cmd="${*: -1}"
if [ -n "${STUB_SSH_REAL:-}" ]; then
  exec bash -c "$cmd"
fi
case "$cmd" in
  *--count-live*)
    [ -z "${STUB_COUNT_FAIL:-}" ] || exit 1
    cat "$STUB_DIR/live_counts.txt" ;;
  *load_zstruct.sh*)
    [ -z "${STUB_LOAD_FAIL:-}" ] || { echo "12:00:00 FATAL: stub load failed"; exit 1; }
    [ -z "${STUB_LOAD_NO_MARKER:-}" ] || { echo "stub load exits 0 without a marker"; exit 0; }
    echo "12:00:00 LOADED: schema zstruct = 7 tables; papers 3, cites 2. zstruct_prev: 1 (stub)" ;;
esac
"""
# Pulls (a source with host:) copy a canned file to the destination file. A push is logged; with STUB_RSYNC_COPY it
# also copies the build, the way rsync --delete would, into the (local) destination, and STUB_RSYNC_TAMPER then
# changes the copied qa.json.
STUB_RSYNC = """\
#!/usr/bin/env bash
echo "rsync $*" >> "$STUB_DIR/rsync.log"
src="${*: -2:1}"; dest="${*: -1}"
case "$src" in
  *:*) cp "$STUB_DIR/remote_files/${src##*/}" "$dest" ;;
  *)
    case "$dest" in *:*) echo "PUSH $dest" >> "$STUB_DIR/rsync.log" ;; esac
    [ -z "${STUB_PUSH_FAIL:-}" ] || exit 23
    if [ -n "${STUB_RSYNC_COPY:-}" ]; then
      dest="${dest#*:}"
      rm -rf "$dest"; mkdir -p "$dest"; cp -a "$src." "$dest"
      [ -z "${STUB_RSYNC_TAMPER:-}" ] || echo " " >> "${dest}qa.json"
    fi ;;
esac
"""
STUB_PGREP = """\
#!/usr/bin/env bash
echo "pgrep $*" >> "$STUB_DIR/pgrep.log"
[ -n "${STUB_SWEEP_RUNNING:-}" ]
"""
# tee that holds its output back for STUB_TEE_DELAY seconds: a caller that does not wait for it loses the tail
STUB_TEE = """\
#!/usr/bin/env bash
sleep "${STUB_TEE_DELAY:-0}"
exec %s "$@"
"""
STUB_PSQL = """\
#!/usr/bin/env bash
args="$*"
echo "psql $args" >> "$STUB_DIR/psql.log"
echo "env PGHOST=${PGHOST:-} PGUSER=${PGUSER:-} PGDATABASE=${PGDATABASE:-} PGPASSWORD=${PGPASSWORD:-}" >> "$STUB_DIR/psql.log"
case "$args" in
  *pg_class*)
    [ -z "${STUB_CATALOG_FAIL:-}" ] || exit 1
    n=$(( $(cat "$STUB_DIR/catalog_calls" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$STUB_DIR/catalog_calls"
    f="$STUB_DIR/live_counts.txt"
    if [ "$n" -gt 1 ] && [ -f "$STUB_DIR/live_counts_2.txt" ]; then f="$STUB_DIR/live_counts_2.txt"; fi
    cat "$f" ;;
  *"-f "*) : ;;
  *"DROP SCHEMA"*) : ;;
  *pg_extension*) echo "${STUB_PGVECTOR:-1}" ;;
  *"<=>"*) echo "${STUB_SIMILAR:-10 rows, top 0.9123}" ;;
  *'FROM zstruct_next."'*)
    t=$(sed -E 's/.*zstruct_next\\."([a-z_0-9]+)".*/\\1/' <<< "$args")
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        script(self.bin / "tee", STUB_TEE % REAL_TEE)
        (self.bin / "python3").symlink_to(sys.executable)
        (self.stub / "qa.json").write_text(json.dumps(QA))
        (self.stub / "live_counts.txt").write_text(LIVE)
        (self.stub / "db_counts.txt").write_text(
            "".join(f"{t} {n}\n" for t, n in QA["row_counts"].items())
        )
        (self.stub / "remote_files" / "metadata_export.json").write_text("{}")
        (self.stub / "remote_files" / "references_export.json").write_text("{}")
        self.pwfile = root / "pw"
        self.pwfile.write_text("dummy-password\n")
        self.notified = root / "notified.txt"
        self.config = root / "citegraph.env"
        self.write_config(remote=remote)

    def write_config(
        self,
        extra: str = "",
        remote: str = "dbhost",
        remote_code: Path | str | None = None,
        remote_config: Path | str | None = None,
    ) -> None:
        lines = [
            f"CITEGRAPH_DATA={self.data}",
            f"CITEGRAPH_CODE={self.code}",
            "CITEGRAPH_METADATA_SRC=srchost:/src/metadata_export.json",
            "CITEGRAPH_REFERENCES_SRC=srchost:/src/references_export.json",
            "PGHOST=dbserver",
            "PGUSER=cguser",
            "PGDATABASE=cgdb",
            f"CITEGRAPH_PGPASSWORD_FILE={self.pwfile}",
            "CITEGRAPH_LOAD_WINDOW=none",
            'citegraph_with_key() { OPENALEX_API_KEY=dummy-key "$@"; }',
            f'citegraph_notify() {{ echo "$1" >> {self.notified}; }}',
        ]
        if remote:  # the database host's copy of the config is this same file, with the same data dir
            lines += [
                f"CITEGRAPH_REMOTE={remote}",
                f"CITEGRAPH_REMOTE_CODE={shlex.quote(str(remote_code or CITEGRAPH))}",
                f"CITEGRAPH_REMOTE_CONFIG={shlex.quote(str(remote_config or self.config))}",
                f"CITEGRAPH_REMOTE_DATA={self.data}",
            ]
        lines.append(extra)
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

    def log_path(self) -> Path:
        return self.data / "logs" / f"refresh_{datetime.date.today():%Y%m}.log"

    def log_lines(self):
        return (
            self.log_path().read_text().splitlines() if self.log_path().exists() else []
        )

    def stub_log(self, name: str) -> str:
        path = self.stub / name
        return path.read_text() if path.exists() else ""

    def notes(self) -> list[str]:
        return self.notified.read_text().splitlines() if self.notified.exists() else []

    def make_loaded(self, marker="old", age_days=0):
        d = self.data / "out.loaded"
        d.mkdir()
        (d / "papers.csv").write_text(marker + "\n")
        (d / "cites.csv").write_text(marker + "\n")
        if age_days:
            t = time.time() - age_days * 86400
            os.utime(d, (t, t))
        return d


@pytest.fixture
def site(tmp_path):
    return Site(tmp_path)


@pytest.fixture
def local_site(tmp_path):
    """The database is local: refresh.sh runs the real load_zstruct.sh from its own directory."""
    s = Site(tmp_path, remote="")
    s.write_config(remote="")
    return s


def strip_stamp(line: str) -> str:
    """Drop the 'YYYY-MM-DD HH:MM:SS ' prefix of a log line."""
    return line.split(" ", 2)[2]


# --- both scripts refuse to run without a config -----------------------------------------------------------------


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

    @pytest.mark.parametrize("bad", ["data", "./data", "~/data", "../data"])
    def test_refuses_a_data_dir_that_is_not_absolute(self, site, which, bad):
        site.write_config(f"CITEGRAPH_DATA='{bad}'")
        proc = site.run(which, "--config", str(site.config))
        assert (
            proc.returncode == 2
            and "CITEGRAPH_DATA must be an absolute path" in proc.stderr
        )
        assert not (site.data / "logs").exists()

    def test_refuses_an_unknown_option(self, site, which):
        proc = site.run(which, "--config", str(site.config), "--bogus")
        assert proc.returncode == 2 and "usage" in proc.stderr


def test_refresh_refuses_a_remote_without_its_paths(site):
    site.config.write_text(
        f"CITEGRAPH_DATA={site.data}\nCITEGRAPH_METADATA_SRC=a:b\nCITEGRAPH_REFERENCES_SRC=a:c\nCITEGRAPH_REMOTE=dbhost\n"
    )
    proc = site.refresh("--dry-run")
    assert proc.returncode == 2 and "CITEGRAPH_REMOTE_CODE is not set" in proc.stderr


@pytest.mark.parametrize(
    "name",
    [
        "CITEGRAPH_REMOTE_CODE",
        "CITEGRAPH_REMOTE_CONFIG",
        "CITEGRAPH_REMOTE_DATA",
        "CITEGRAPH_CODE",
    ],
)
@pytest.mark.parametrize("bad", ["rel/dir", "~/citegraph"])
def test_refresh_refuses_a_relative_or_tilde_path(site, name, bad):
    site.write_config(f"{name}='{bad}'")
    proc = site.refresh("--dry-run")
    assert proc.returncode == 2 and "must be an absolute path" in proc.stderr
    assert site.stub_log("ssh.log") == ""


def test_refresh_refuses_a_bad_stale_days(site):
    site.write_config("CITEGRAPH_STALE_DAYS=soon")
    proc = site.refresh("--dry-run")
    assert proc.returncode == 2 and "CITEGRAPH_STALE_DAYS" in proc.stderr


# --- load_zstruct.sh ---------------------------------------------------------------------------------------------


def window_excluding_now() -> str:
    return "1300-1400" if datetime.datetime.now().hour < 12 else "0100-0200"


def make_build(site, name="out.daily", counts=None, schema="zstruct"):
    d = site.data / name
    d.mkdir(parents=True, exist_ok=True)
    qa = dict(QA, row_counts=counts or QA["row_counts"])
    (d / "qa.json").write_text(json.dumps(qa))
    for f in (f"apply_{schema}.sql", f"swap_{schema}.sql", "papers.csv", "cites.csv"):
        (d / f).write_text("-- stub\n")
    return d


def combined(proc) -> str:
    """The load script sends stderr through its tee, once its log redirect is in place."""
    return proc.stdout + proc.stderr


def applied(site) -> bool:
    return "-f apply_" in site.stub_log("psql.log")


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
            and "FATAL: missing" in combined(proc)
            and "cites.csv" in combined(proc)
        )
        assert site.stub_log("psql.log") == ""

    def test_refuses_an_empty_build_file(self, site):
        d = make_build(site)
        (d / "papers.csv").write_text("")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "FATAL: missing" in combined(proc)

    def test_refuses_outside_its_window(self, site):
        make_build(site)
        site.write_config(f"CITEGRAPH_LOAD_WINDOW={window_excluding_now()}", remote="")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "run between" in combined(proc)
        assert site.stub_log("psql.log") == ""

    def test_loads_inside_its_window(self, site):
        make_build(site)
        site.write_config("CITEGRAPH_LOAD_WINDOW=0000-2400", remote="")
        assert self.load(site, "out.daily").returncode == 0

    def test_a_load_window_is_required(self, site):
        make_build(site)
        site.write_config("unset CITEGRAPH_LOAD_WINDOW", remote="")
        proc = self.load(site, "out.daily")
        assert (
            proc.returncode == 2 and "CITEGRAPH_LOAD_WINDOW is not set" in proc.stderr
        )
        assert "none" in proc.stderr and site.stub_log("psql.log") == ""

    @pytest.mark.parametrize("bad", ["", "8:45-23", "0845-23:00", "always"])
    def test_refuses_a_malformed_window(self, site, bad):
        make_build(site)
        site.write_config(f"CITEGRAPH_LOAD_WINDOW='{bad}'", remote="")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 2 and "CITEGRAPH_LOAD_WINDOW" in proc.stderr

    def test_none_means_any_time(self, site):
        make_build(site)
        assert self.load(site, "out.daily").returncode == 0

    def test_refuses_failed_hard_checks(self, site):
        d = make_build(site)
        (d / "qa.json").write_text(json.dumps(dict(QA, hard_checks={"passed": False})))
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "hard checks did not pass" in combined(proc)

    @pytest.mark.parametrize(
        "qa",
        [
            {"hard_checks": {"passed": True}},
            {"hard_checks": {"passed": True}, "row_counts": {}},
        ],
    )
    def test_refuses_a_qa_whose_row_counts_cannot_be_read(self, site, qa):
        d = make_build(site)
        (d / "qa.json").write_text(json.dumps(qa))
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "row counts cannot be read" in combined(proc)
        assert "nothing changed" in combined(proc) and not applied(site)

    def test_refuses_without_pgvector(self, site):
        make_build(site)
        proc = self.load(site, "out.daily", STUB_PGVECTOR="0")
        assert proc.returncode == 1 and "pgvector is not installed" in combined(proc)
        assert not applied(site)

    def test_loads_and_swaps(self, site):
        make_build(site)
        proc = self.load(site, "out.daily")
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
        assert (
            "papers: 3 ok" in out and "similar (pgvector): 10 rows, top 0.9123" in out
        )
        assert "LOADED: schema zstruct = 7 tables; papers 3, cites 2." in out
        calls = site.stub_log("psql.log")
        assert "-f apply_zstruct.sql" in calls and "swap_zstruct.sql" in calls
        assert calls.index("pg_class") < calls.index(
            "-f apply_zstruct.sql"
        )  # the shrink check comes first
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
        site.write_config("CITEGRAPH_SCHEMA=cg", remote="")
        script(site.bin / "psql", STUB_PSQL.replace("zstruct", "cg"))
        proc = self.load(site, "out.daily")
        assert proc.returncode == 0, proc.stderr
        assert (
            "LOADED: schema cg = 7 tables" in proc.stdout and "cg_prev:" in proc.stdout
        )
        calls = site.stub_log("psql.log")
        assert "apply_cg.sql" in calls and "n.nspname = 'cg'" in calls

    def test_refuses_a_bad_schema_name(self, site):
        make_build(site)
        site.write_config("CITEGRAPH_SCHEMA='x; drop'", remote="")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 2 and "CITEGRAPH_SCHEMA" in proc.stderr

    def test_a_row_count_mismatch_drops_the_new_schema_and_does_not_swap(self, site):
        make_build(site)
        (site.stub / "db_counts.txt").write_text(
            "papers 2\ncites 2\ncite_contexts 1\nexternal_works 1\npaper_embeddings 3\n"
        )
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "row counts differ from qa.json" in combined(
            proc
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
        assert proc.returncode == 1 and "expected 10 rows" in combined(proc)
        calls = site.stub_log("psql.log")
        assert (
            "DROP SCHEMA IF EXISTS zstruct_next CASCADE" in calls
            and "swap_zstruct.sql" not in calls
        )

    def test_refuses_an_unreadable_password_file(self, site):
        make_build(site)
        site.pwfile.unlink()
        proc = self.load(site, "out.daily")
        assert proc.returncode == 2 and "cannot read the password file" in proc.stderr
        assert site.stub_log("psql.log") == ""

    def test_a_leading_tilde_in_the_password_file_means_this_hosts_home(self, site):
        make_build(site)
        site.write_config("CITEGRAPH_PGPASSWORD_FILE='~/.pw'", remote="")
        (site.root / ".pw").write_text("tilde-password\n")
        proc = self.load(site, "out.daily", HOME=str(site.root))
        assert proc.returncode == 0, proc.stderr
        assert "PGPASSWORD=tilde-password" in site.stub_log("psql.log")

    # --expect-qa
    def test_expect_qa_accepts_the_matching_hash(self, site):
        d = make_build(site)
        proc = self.load(site, "--expect-qa", sha256(d / "qa.json"), "out.daily")
        assert proc.returncode == 0, proc.stderr

    def test_expect_qa_refuses_another_qa_json_before_anything_changes(self, site):
        d = make_build(site)
        proc = self.load(site, "--expect-qa", "0" * 64, "out.daily")
        assert proc.returncode == 1 and "is not the one that was checked" in combined(
            proc
        )
        assert "nothing changed" in combined(proc) and site.stub_log("psql.log") == ""
        assert sha256(d / "qa.json") in combined(proc)

    def test_expect_qa_wants_a_sha256(self, site):
        make_build(site)
        proc = self.load(site, "--expect-qa", "abc", "out.daily")
        assert proc.returncode == 2 and "--expect-qa wants a sha256" in proc.stderr

    # the shrink check inside the load
    def test_a_table_below_98_percent_of_the_live_rows_is_refused_before_apply(
        self, site
    ):
        make_build(site)
        (site.stub / "live_counts.txt").write_text("cites 2\npapers 100\n")
        proc = self.load(site, "out.daily")
        assert (
            proc.returncode == 1
            and "tables would shrink by more than 2%" in combined(proc)
        )
        assert "papers 100 -> 3" in combined(proc) and "nothing changed" in combined(
            proc
        )
        assert not applied(site) and "swap_zstruct.sql" not in site.stub_log("psql.log")

    def test_allow_shrink_loads_a_deliberately_smaller_build(self, site):
        make_build(site)
        (site.stub / "live_counts.txt").write_text("cites 2\npapers 100\n")
        proc = self.load(site, "--allow-shrink", "out.daily")
        assert proc.returncode == 0, proc.stderr
        assert (
            "--allow-shrink given, loading anyway" in proc.stdout
            and "LOADED:" in proc.stdout
        )

    def test_exactly_98_percent_loads(self, site):
        make_build(site, counts=dict(QA["row_counts"], papers=980))
        (site.stub / "db_counts.txt").write_text(
            "papers 980\ncites 2\ncite_contexts 1\nexternal_works 1\npaper_embeddings 3\n"
        )
        (site.stub / "live_counts.txt").write_text("papers 1000\n")
        assert self.load(site, "out.daily").returncode == 0

    def test_a_table_new_to_the_build_cannot_shrink(self, site):
        make_build(site)
        (site.stub / "live_counts.txt").write_text(
            "cites 2\npapers 3\n"
        )  # the live schema lacks the other three
        proc = self.load(site, "out.daily")
        assert proc.returncode == 0, combined(proc)

    def test_a_first_load_has_nothing_to_compare(self, site):
        make_build(site)
        (site.stub / "live_counts.txt").write_text(
            ""
        )  # no live schema: the catalog query lists nothing
        proc = self.load(site, "out.daily")
        assert proc.returncode == 0, combined(proc)

    def test_malformed_live_counts_stop_the_load_before_apply(self, site):
        make_build(site)
        (site.stub / "live_counts.txt").write_text("papers many\n")
        proc = self.load(site, "out.daily")
        assert proc.returncode == 1 and "the shrink check did not run" in combined(proc)
        assert not applied(site)

    def test_a_failed_live_count_stops_the_load_before_apply(self, site):
        make_build(site)
        proc = self.load(site, "out.daily", STUB_CATALOG_FAIL="1")
        assert (
            proc.returncode == 1
            and "could not count the live zstruct tables" in combined(proc)
        )
        assert not applied(site)

    # --count-live
    def test_count_live_prints_the_catalog_counts_and_changes_nothing(self, site):
        site.write_config(f"CITEGRAPH_LOAD_WINDOW={window_excluding_now()}", remote="")
        proc = self.load(site, "--count-live")
        assert proc.returncode == 0 and proc.stdout == LIVE
        calls = site.stub_log("psql.log")
        assert (
            calls.count("psql ") == 1
            and "pg_class" in calls
            and "n.nspname = 'zstruct'" in calls
        )
        assert "relkind = 'r'" in calls and "query_to_xml" in calls
        assert "-f " not in calls and "DROP" not in calls
        assert not (
            site.data / "logs" / "load_zstruct.log"
        ).exists()  # no log, no window, no DIR

    def test_count_live_of_a_missing_schema_prints_nothing(self, site):
        (site.stub / "live_counts.txt").write_text("")
        proc = self.load(site, "--count-live")
        assert proc.returncode == 0 and proc.stdout == ""

    def test_count_live_fails_when_the_query_fails(self, site):
        assert self.load(site, "--count-live", STUB_CATALOG_FAIL="1").returncode != 0

    def test_count_live_needs_the_password_file_to_be_readable(self, site):
        site.pwfile.unlink()
        proc = self.load(site, "--count-live")
        assert proc.returncode == 2 and "cannot read the password file" in proc.stderr


# --- refresh.sh with a canned ssh ---------------------------------------------------------------------------------


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
        assert rsync.count("rsync -a -e") == 2
        # the inputs land under fixed names, whatever the sources are called
        assert "srchost:/src/metadata_export.json input/zotero_metadata.json" in rsync
        assert (
            "srchost:/src/references_export.json input/zotero_references.json" in rsync
        )
        assert (site.data / "input" / "zotero_metadata.json").exists()
        ssh = site.stub_log("ssh.log").splitlines()
        assert (
            len(ssh) == 1
            and "load_zstruct.sh" in ssh[0]
            and ssh[0].endswith("--count-live")
        )
        # refresh.sh holds no database settings: only the load script's name and its config travel
        assert (
            "psql" not in ssh[0]
            and "PG" not in ssh[0]
            and "dummy-password" not in ssh[0]
        )
        assert site.stub_log("pgrep.log") == ""  # the sweep-running skip does not apply
        builds = site.stub_log("builder_calls.log").splitlines()
        assert builds == [
            "online schema=zstruct key=yes",
            "offline schema=zstruct key=no",
        ]
        assert (site.data / "out.daily.dryrun" / "papers.csv").exists()
        assert (old / "papers.csv").read_text() == "old\n"
        assert (
            not (site.data / "out.loaded.prev").exists() and not site.notified.exists()
        )

    def test_dry_run_ignores_a_running_sweep(self, site):
        proc = site.refresh("--dry-run", STUB_SWEEP_RUNNING="1")
        assert proc.returncode == 0
        assert strip_stamp(site.log_lines()[-1]).startswith("[dry-run] gates passed;")

    def test_a_failing_dry_run_does_not_notify(self, site):
        site.make_loaded(age_days=30)
        qa = json.loads((site.stub / "qa.json").read_text())
        qa["hard_checks"]["passed"] = False
        (site.stub / "qa.json").write_text(json.dumps(qa))
        proc = site.refresh("--dry-run")
        assert proc.returncode == 1
        assert strip_stamp(site.log_lines()[-1]).startswith(
            "[dry-run] FAILED: the hard checks or the reference check"
        )
        assert not site.notified.exists()  # not even the stale-build notice

    def test_a_table_new_to_the_build_is_not_a_shrink(self, site):
        (site.stub / "live_counts.txt").write_text("cites 2\npapers 3\n")
        proc = site.refresh("--dry-run")
        assert proc.returncode == 0
        assert strip_stamp(site.log_lines()[-1]).startswith("[dry-run] gates passed;")

    def test_a_first_load_has_no_live_schema(self, site):
        (site.stub / "live_counts.txt").write_text("")
        assert site.refresh("--dry-run").returncode == 0

    def test_a_dry_run_with_a_local_database_counts_through_the_load_script(
        self, local_site
    ):
        proc = local_site.refresh("--dry-run")
        assert proc.returncode == 0, proc.stderr
        assert strip_stamp(local_site.log_lines()[-1]).startswith(
            "[dry-run] gates passed;"
        )
        assert local_site.stub_log("ssh.log") == ""
        psql = local_site.stub_log("psql.log")
        assert psql.count("psql ") == 1 and "pg_class" in psql
        assert (
            "PGPASSWORD=dummy-password" in psql
        )  # read by the load script from its config

    def test_the_shrink_check_that_cannot_run_is_reported(self, site):
        (site.stub / "live_counts.txt").write_text("papers lots\n")
        proc = site.refresh("--dry-run")
        assert proc.returncode == 1
        assert (
            "FAILED: the shrink check did not run -- not loaded" in site.log_lines()[-1]
        )

    def test_a_malformed_row_count_in_the_qa_is_a_shrink_check_that_cannot_run(
        self, site
    ):
        qa = dict(QA, row_counts=dict(QA["row_counts"], papers="many"))
        (site.stub / "qa.json").write_text(json.dumps(qa))
        proc = site.refresh("--dry-run")
        assert proc.returncode == 1
        assert (
            "FAILED: the shrink check did not run -- not loaded" in site.log_lines()[-1]
        )


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
        # the build went to the database host's out.daily with --delete, then loaded under a 30 min limit, with
        # the sha256 of the qa.json the gates checked
        rsync = site.stub_log("rsync.log")
        assert f"PUSH dbhost:{site.data}/out.daily/" in rsync and "--delete" in rsync
        ssh = site.stub_log("ssh.log").splitlines()
        assert len(ssh) == 2 and ssh[0].endswith("--count-live")
        sha = sha256(site.data / "out.loaded" / "qa.json")
        assert (
            "timeout 1800 bash" in ssh[1] and f"--expect-qa {sha} out.daily" in ssh[1]
        )
        assert "--allow-shrink" not in ssh[1]
        sweep = site.stub_log("sweep_calls.log")
        assert "--store s2store/api_citations.sqlite --max-hours 3" in sweep
        # rotation: loaded -> prev, new build -> loaded, the one before prev is gone
        assert (
            site.data / "out.loaded" / "papers.csv"
        ).read_text() == "papers.csv,NEW\n"
        assert (site.data / "out.loaded.prev" / "papers.csv").read_text() == "old\n"
        assert (old / "cites.csv").read_text() == "cites.csv,NEW\n"
        assert not (site.data / "out.daily.new").exists() and not site.notified.exists()

    def test_first_load_with_no_loaded_build(self, site):
        assert site.refresh().returncode == 0
        assert (site.data / "out.loaded" / "papers.csv").exists()
        assert not (site.data / "out.loaded.prev").exists()

    def test_the_sweep_max_hours_comes_from_the_config(self, site):
        site.write_config("CITEGRAPH_SWEEP_MAX_HOURS=1")
        site.refresh()
        assert "--max-hours 1" in site.stub_log("sweep_calls.log")

    def test_an_identical_build_is_not_loaded_again_and_counts_as_current(self, site):
        d = site.make_loaded(age_days=10)
        (d / "papers.csv").write_text(
            "papers.csv,A\n"
        )  # the bytes the stub builder writes by default
        (d / "cites.csv").write_text("cites.csv,A\n")
        t = time.time() - 10 * 86400
        os.utime(d, (t, t))
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
        assert time.time() - d.stat().st_mtime < 3600  # checked today: not stale
        assert site.refresh().returncode == 0 and not site.notified.exists()

    def test_a_dry_run_with_an_identical_build_leaves_the_age_alone(self, site):
        d = site.make_loaded(age_days=10)
        (d / "papers.csv").write_text(
            "papers.csv,A\n"
        )  # the stub builder's default bytes
        (d / "cites.csv").write_text("cites.csv,A\n")
        t = time.time() - 10 * 86400
        os.utime(d, (t, t))
        proc = site.refresh("--dry-run")
        assert proc.returncode == 0
        assert strip_stamp(site.log_lines()[-1]).endswith(
            "no change against the last loaded build (out.loaded): not loaded"
        )
        assert (
            abs(d.stat().st_mtime - t) < 5
        )  # a dry run must not hide a stale live build

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
        (note,) = site.notes()
        assert "the load on the database host failed" in note
        assert note.endswith(f"-- log {site.log_path()}")

    def test_exit_zero_without_a_loaded_marker_is_a_failure(self, site):
        proc = site.refresh(STUB_LOAD_NO_MARKER="1")
        assert (
            proc.returncode == 1
            and "FAILED: the load on the database host failed (rc=0"
            in site.log_lines()[-1]
        )
        assert not (site.data / "out.loaded").exists()

    def test_a_failed_push_stops_before_the_load(self, site):
        old = site.make_loaded()
        proc = site.refresh(STUB_PUSH_FAIL="1")
        assert proc.returncode == 1
        assert (
            "FAILED: copying the build to the database host failed -- not loaded"
            in site.log_lines()[-1]
        )
        assert (
            len(site.stub_log("ssh.log").splitlines()) == 1
        )  # only the count: no load
        assert len(site.notes()) == 1 and (old / "papers.csv").read_text() == "old\n"

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

    def test_a_crashing_offline_build_is_not_loaded_and_is_named(self, site):
        old = site.make_loaded()
        proc = site.refresh(STUB_OFFLINE_CRASH="1")
        assert proc.returncode == 1
        lines = site.log_lines()
        assert "build rc=1: Traceback: the offline build crashed" in lines[-2]
        # the online build's qa.json was removed first, so the message names the offline build, not the hard checks
        assert lines[-1].endswith(
            "FAILED: the offline build wrote no qa.json (logs/refresh_build.log) -- not loaded"
        )
        assert site.stub_log("ssh.log") == "" and "PUSH" not in site.stub_log(
            "rsync.log"
        )
        (note,) = site.notes()
        assert "the offline build wrote no qa.json" in note
        assert (old / "papers.csv").read_text() == "old\n"

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
        (site.stub / "remote_files" / "references_export.json").unlink()
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
        assert "[s]2_citation_sweep.py --out" in site.stub_log("pgrep.log")

    def test_the_sweep_process_pattern_comes_from_the_config(self, site):
        site.write_config(
            "CITEGRAPH_SWEEP_PROCESS_PATTERN='[s]weep_loop.sh|[s]2_citation_sweep.py'"
        )
        site.refresh(STUB_SWEEP_RUNNING="1")
        assert "[s]weep_loop.sh|[s]2_citation_sweep.py" in site.stub_log("pgrep.log")

    def test_the_default_sweep_pattern_does_not_match_a_pytest_or_a_pager(self, site):
        import re

        pattern = "[s]2_citation_sweep.py --out"
        assert re.search(
            pattern, "python3 s2_citation_sweep.py --out out --store s2store/x.sqlite"
        )
        assert not re.search(
            pattern, "python -m pytest tests/citegraph/test_s2_citation_sweep.py"
        )
        assert not re.search(pattern, "less s2_citation_sweep.py")
        assert not re.search(pattern, "pgrep -f [s]2_citation_sweep.py --out")

    def test_skips_while_another_refresh_holds_the_lock(self, site):
        (site.data / "logs").mkdir()
        with open(site.data / "logs" / ".refresh.lock", "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            proc = site.refresh("--dry-run")
        assert proc.returncode == 0
        assert [strip_stamp(ln) for ln in site.log_lines()] == [
            "[dry-run] skip: another refresh is running"
        ]
        assert site.stub_log("rsync.log") == ""


class TestStaleBuild:
    def test_a_skip_names_the_age_of_an_old_build(self, site):
        site.make_loaded(age_days=5)
        proc = site.refresh(STUB_SWEEP_RUNNING="1")
        assert proc.returncode == 0
        lines = [strip_stamp(ln) for ln in site.log_lines()]
        assert lines == [
            "skip: a citation sweep is running",
            "the live build is 5 days old (out.loaded)",
        ]
        (note,) = site.notes()
        assert "5 days ago" in note and note.endswith(f"-- log {site.log_path()}")

    def test_a_fresh_build_is_not_reported_at_a_skip(self, site):
        site.make_loaded(age_days=1)
        site.refresh(STUB_SWEEP_RUNNING="1")
        assert not site.notified.exists()

    def test_the_limit_is_the_configured_number_of_days(self, site):
        site.make_loaded(age_days=2)
        site.write_config("CITEGRAPH_STALE_DAYS=1")
        site.refresh(STUB_SWEEP_RUNNING="1")
        assert "2 days ago" in site.notified.read_text()

    def test_no_loaded_build_is_not_a_stale_build(self, site):
        site.refresh(STUB_SWEEP_RUNNING="1")
        assert not site.notified.exists()

    def test_a_failure_adds_one_stale_notice_even_after_a_sweep_notice(self, site):
        site.make_loaded(age_days=6)
        proc = site.refresh(STUB_SWEEP_RC="2", STUB_LOAD_FAIL="1")
        assert proc.returncode == 1
        notes = site.notes()
        assert len(notes) == 3  # the sweep, the failed load, and the age: once
        assert sum("6 days ago" in n for n in notes) == 1

    def test_the_lock_skip_also_reports_it(self, site):
        site.make_loaded(age_days=4)
        (site.data / "logs").mkdir()
        with open(site.data / "logs" / ".refresh.lock", "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            site.refresh()
        assert "4 days ago" in site.notified.read_text()

    def test_a_dry_run_never_reports_it(self, site):
        site.make_loaded(age_days=9)
        site.refresh("--dry-run", STUB_OFFLINE_CRASH="1")
        assert not site.notified.exists()


# --- refresh.sh against the real load_zstruct.sh ---------------------------------------------------------------------


class TestRefreshWithTheRealLoadScript:
    """CITEGRAPH_REMOTE empty: refresh.sh runs load_zstruct.sh from its own directory, with a stub psql."""

    def test_load_succeeds_and_rotates_even_when_the_tee_is_slow(self, local_site):
        site = local_site
        old = site.make_loaded()
        proc = site.refresh(STUB_CSV_TAG="NEW", STUB_RSYNC_COPY="1", STUB_TEE_DELAY="1")
        assert proc.returncode == 0, proc.stderr
        lines = [strip_stamp(ln) for ln in site.log_lines()]
        assert lines[-2].startswith(
            "LOADED: schema zstruct = 7 tables; papers 3, cites 2."
        )
        assert lines[-1].startswith("done: papers=3")
        load_log = (site.data / "logs" / "refresh_load.log").read_text()
        assert "LOADED:" in load_log and "swap: zstruct_next -> zstruct" in load_log
        calls = site.stub_log("psql.log")
        assert "-f apply_zstruct.sql" in calls and "swap_zstruct.sql" in calls
        assert calls.count("pg_class") == 2  # the gate's count and the load's own
        assert (
            site.data / "out.loaded" / "papers.csv"
        ).read_text() == "papers.csv,NEW\n"
        assert (site.data / "out.loaded.prev" / "papers.csv").read_text() == "old\n"
        assert (old / "cites.csv").read_text() == "cites.csv,NEW\n"
        assert (
            site.data / "out.daily" / "qa.json"
        ).exists()  # the local copy that the load read
        assert not site.notified.exists()

    def test_a_qa_json_that_is_not_the_one_checked_is_not_applied(self, local_site):
        site = local_site
        old = site.make_loaded()
        proc = site.refresh(STUB_RSYNC_COPY="1", STUB_RSYNC_TAMPER="1")
        assert proc.returncode == 1
        assert (
            "is not the one that was checked"
            in (site.data / "logs" / "refresh_load.log").read_text()
        )
        assert (
            "FAILED: the load on the database host failed (rc=1" in site.log_lines()[-1]
        )
        assert not applied(site)
        assert (old / "papers.csv").read_text() == "old\n" and not (
            site.data / "out.loaded.prev"
        ).exists()
        assert len(site.notes()) == 1

    def test_a_shrink_seen_only_by_the_load_is_refused_there(self, local_site):
        site = local_site
        old = site.make_loaded()
        # the gate's count (the first catalog query) is fine; the load's own count, a moment later, shows more rows
        (site.stub / "live_counts_2.txt").write_text("cites 2\npapers 100\n")
        proc = site.refresh(STUB_RSYNC_COPY="1")
        assert proc.returncode == 1
        load_log = (site.data / "logs" / "refresh_load.log").read_text()
        assert "tables would shrink by more than 2%" in load_log
        assert (
            "loading anyway" not in load_log
        )  # refresh.sh never passes --allow-shrink
        assert not applied(site)
        assert (old / "papers.csv").read_text() == "old\n" and len(site.notes()) == 1

    def test_the_load_refuses_outside_its_window(self, local_site):
        site = local_site
        site.write_config(f"CITEGRAPH_LOAD_WINDOW={window_excluding_now()}", remote="")
        proc = site.refresh(STUB_RSYNC_COPY="1")
        assert proc.returncode == 1
        assert "run between" in (site.data / "logs" / "refresh_load.log").read_text()
        assert not applied(site)


class TestRemoteCommandParsedByARealShell:
    """The stub ssh runs its command line with bash -c, as an sshd would with a bash login shell. The install
    directory and its config have a space in their names, so the printf %q quoting is what makes them work."""

    def install(self, site):
        inst = site.root / "in stall dir"
        inst.mkdir()
        for name in ("load_zstruct.sh", "refresh_gates.py"):
            shutil.copy(CITEGRAPH / name, inst / name)
        conf = inst / "citegraph db.env"
        site.write_config(remote_code=inst, remote_config=conf)
        conf.write_text(site.config.read_text())
        return inst, conf

    def test_load_through_a_bash_c_ssh(self, site):
        site.make_loaded()
        self.install(site)
        proc = site.refresh(STUB_SSH_REAL="1", STUB_RSYNC_COPY="1", STUB_CSV_TAG="NEW")
        assert proc.returncode == 0, proc.stderr
        ssh = site.stub_log("ssh.log")
        assert "in\\ stall\\ dir/load_zstruct.sh" in ssh and "citegraph\\ db.env" in ssh
        assert ssh.count("--count-live") == 1 and "--expect-qa" in ssh
        assert "-f apply_zstruct.sql" in site.stub_log("psql.log")
        assert (
            site.data / "out.loaded" / "papers.csv"
        ).read_text() == "papers.csv,NEW\n"
        assert (site.data / "out.loaded.prev" / "papers.csv").read_text() == "old\n"
        assert strip_stamp(site.log_lines()[-1]).startswith("done:")

    def test_the_live_count_and_a_dry_run_through_a_bash_c_ssh(self, site):
        self.install(site)
        proc = site.refresh("--dry-run", STUB_SSH_REAL="1")
        assert proc.returncode == 0, proc.stderr
        assert strip_stamp(site.log_lines()[-1]).startswith("[dry-run] gates passed;")
        assert "pg_class" in site.stub_log("psql.log") and not applied(site)

    def test_a_missing_install_on_the_database_host_fails_the_count(self, site):
        site.write_config(remote_code=site.root / "nowhere")
        proc = site.refresh("--dry-run", STUB_SSH_REAL="1")
        assert proc.returncode == 1
        assert (
            "could not count the live zstruct tables on the database host"
            in site.log_lines()[-1]
        )
