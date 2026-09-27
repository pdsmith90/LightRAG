#!/usr/bin/env bash
# Run the offline test suite from the toolkit directory (zotero-toolkit/, the parent of
# scripts/), wherever it is called from. Extra arguments go to pytest.
#
#   python3 -m pip install -r requirements-dev.txt    # once, in a virtual environment
#   scripts/test.sh                                   # or: scripts/test.sh tests/mcp -x
#
# PYTHON selects the interpreter (default python3). The PostgreSQL + Apache AGE
# integration test is skipped unless CLEAN_DANGLING_TEST_DSN is set; see
# maintenance/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then
  set -- tests
fi
exec "${PYTHON:-python3}" -m pytest "$@"
