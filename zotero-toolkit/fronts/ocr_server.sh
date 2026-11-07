#!/usr/bin/env bash
# ocr_server.sh — PepperOCR-VL on the VM's own GPU, ON DEMAND (2026-10-07; operator decision: "the ocr model can supplant the
# rag model if ocr is needed"). The 5700 XT (8 GB) cannot hold it beside the ollama 7b (test 2026-10-07: spilled to
# host RAM, 4.5 tok/s, the VM at 0.9 GB available), so `start` SUPPLANTS the RAG model:
#   1. write .local_llm_paused (epoch "until") -- llm_failover.py stops sending the extraction lane to the local
#      ollama (the primary when allowed, else 503 so LightRAG retries later); in-flight local calls finish (<= 590 s);
#   2. ask ollama to drop the 7b (keep_alive 0) until /api/ps no longer lists it;
#   3. launch llama-server (Vulkan build) with the model + mmproj as alias pepperocr-vl on :19531 (OCR_PORT
#      overrides; :19530 has been embed_failover.py's embedding front since 2026-11-03).
# `stop` kills it (TERM, then KILL: it ignores TERM mid-generation), removes the pause -- the 7b reloads on the
# next local call. build_corpus.py calls `start --for-build` from a worker that needs whole-document OCR and
# `stop --if-build` when the build ends; the marker file keeps a hand-started server out of the build's teardown.
#   ocr_server.sh start [--for-build]   idempotent; flock so three build workers do not race
#   ocr_server.sh stop  [--if-build]    --if-build: only if `start --for-build` started it
#   ocr_server.sh status
# `--for-build` (2026-10-07, operator decision: "ocr should have dedicated run time or otherwise avoid interfering with the other
# model"; that morning the build's OCR took the 7b from three books still extracting -- 75 x 503, all three FAILED):
# the build may take the GPU only while LightRAG has no work for the 7b, and keeps it that way while serving --
# .feeder_pause carries this script's marker (the feeder holds releases, the watchdog defers scans and
# reprocess_failed), and llama-server inherits the trickle's flock (no deletion can start; the lock goes when the
# server dies). Exit 3 = not now: build_corpus.py defers the document. It asks only inside its local window
# (01:00-08:00); a hand-started server (no flag) skips all of this.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${OCR_LLAMA_BIN:-$HOME/llama-vulkan/llama-b11371}
GGUF=${OCR_GGUF:-$HOME/models/PepperOCR-VL-GGUF/PepperOCR-VL.Q8_0.gguf}
MMPROJ=${OCR_MMPROJ:-$HOME/models/PepperOCR-VL-GGUF/PepperOCR-VL.mmproj-Q8_0.gguf}
PORT=${OCR_PORT:-19531}   # 19530 is embed_failover.py's embedding front since 2026-11-03; build_corpus passes OCR_PORT
OLLAMA=${OCR_OLLAMA:-http://127.0.0.1:11436}
RAG_MODEL=${OCR_RAG_MODEL:-qwen2.5:7b-instruct}
PAUSE=$HERE/.local_llm_paused; PID=$HERE/.ocr_server.pid; MARK=$HERE/.ocr_server.started_by_build
LOCK=$HERE/.ocr_server.lock; LOG=$HERE/ocr_server.log
FEEDER_PAUSE=$HERE/.feeder_pause; FP_MARK=ocr_server; TRICKLE_LOCK=$HERE/.reingest_trickle.lock
STOP_LOCK_WAIT_S=${OCR_STOP_LOCK_WAIT_S:-100}   # `stop` waits this long for a `start` in progress (build_corpus allows it 180 s)
UNLOAD_WAIT_S=${OCR_UNLOAD_WAIT_S:-720}   # > llm_failover's FALLBACK_TIMEOUT (590 s): the last in-flight local call
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
not_now() { log "start --for-build: not now -- $*"; exit 3; }
release_feeder() { [ "$(head -c ${#FP_MARK} "$FEEDER_PAUSE" 2>/dev/null)" = "$FP_MARK" ] && rm -f "$FEEDER_PAUSE"; return 0; }
rag_busy() {   # why LightRAG may still need the 7b; prints nothing when it is idle (reingest_trickle.gate_open's test)
  ( set -a; . "$HERE/.env" >/dev/null 2>&1; set +a
    python3 - "${PORT:-9621}" "${LIGHTRAG_API_KEY:-}" <<'PY'
import json, sys, urllib.request
port, key = sys.argv[1], sys.argv[2]
def get(path):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"X-API-Key": key})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)
try:
    counts = get("/documents/status_counts").get("status_counts", {})
    st = get("/documents/pipeline_status")
except Exception as e:
    print(f"LightRAG unreachable ({type(e).__name__})")
    sys.exit()
work = {k: v for k, v in counts.items() if k not in ("processed", "failed", "all") and v}
if work:
    print("LightRAG has " + ", ".join(f"{v} {k}" for k, v in sorted(work.items())) + " document(s)")
elif st.get("busy") or st.get("scanning") or st.get("pending_enqueues"):
    print("the LightRAG pipeline is busy")
PY
  )
}
is_ocr_server() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -qai 'pepperocr-vl'; }   # llama-server's -a alias / the GGUF name
is_llama_server() {   # the executable or argv[0] really is llama-server (never SIGKILL something else that merely greps)
  local a0
  a0=$(tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null | head -n 1)
  case "${a0%% *}" in *llama-server) return 0 ;; esac
  case "$(readlink "/proc/$1/exe" 2>/dev/null)" in */llama-server) return 0 ;; esac
  return 1
}
script_ancestor_pids() {   # this script and its ancestors: a SIGKILL one level up would take the build with it
  local p=$$
  while [ -n "$p" ] && [ "$p" -gt 1 ]; do echo "$p"; p=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null); done
}
ocr_server_pids() {   # live pepperocr-vl llama-servers of this user, minus this script and its ancestors
  local p a
  a=" $(script_ancestor_pids | tr '\n' ' ') "
  for p in $(pgrep -u "$(id -u)" -f 'llama-server.*pepperocr-vl' 2>/dev/null); do
    case "$a" in *" $p "*) continue ;; esac
    is_ocr_server "$p" && is_llama_server "$p" && echo "$p"
  done
}
ocr_model_listed() { curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"pepperocr-vl"'; }   # the -a alias at launch
healthy() { curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && ocr_model_listed; }
port_answers() {   # something is listening and speaking HTTP here (a refused connection prints 000)
  local c
  c=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  case "$c" in ''|000) ;; *) return 0 ;; esac
  c=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  case "$c" in ''|000) return 1 ;; *) return 0 ;; esac
}
rag_loaded() { curl -s -m 5 "$OLLAMA/api/ps" 2>/dev/null | grep -q "\"$RAG_MODEL\""; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/ {printf "%d", $NF/1048576}'; }

cmd=${1:-status}; shift || true
case "$cmd" in
  start)
    exec 9>"$LOCK"; flock 9
    if healthy; then log "start: already serving on :$PORT"; exit 0; fi
    # 2026-11-05: a port that answers without our model is NOT our server (the embedding front answered /health with
    # {"status":"ok"} on :19530 since 11-03, so documents were deferred as "start failed (exit 0)"). Same reading as
    # build_corpus._vlm_probe's "other": never launch onto it.
    if port_answers; then log "start: port $PORT is held by another service (it answers, but /v1/models does not list pepperocr-vl); not launching"; exit 1; fi
    [ -x "$BIN/llama-server" ] && [ -f "$GGUF" ] && [ -f "$MMPROJ" ] || { log "start: binary or GGUFs missing"; exit 2; }
    if [ "${1:-}" = "--for-build" ]; then
      # 2026-11-07: a pause whose "pid N" is one of this script's ancestors belongs to the job that
      # asked for this OCR. reingest_cycle.py writes "reingest_cycle <date> <time> pid N" into .feeder_pause and holds
      # the trickle lock while its conversion (vm_weekly_update.sh -> build_corpus.py -> here) runs, so both checks
      # below refused the cycle's own conversions: one night every start was "held by another job", most of a cycle's
      # deleted documents were deferred, new cycles held, and extraction stood idle until morning. That pause and lock
      # already hold the feeder and deletions off for as long as the build runs, so take neither; release_feeder
      # removes only our own marker and leaves the caller's pause alone.
      hold_pid=$(head -c 200 "$FEEDER_PAUSE" 2>/dev/null | tr '\n' ' ' | sed -n 's/.* pid \([0-9][0-9]*\).*/\1/p')
      if [ -n "$hold_pid" ] && script_ancestor_pids | grep -qx "$hold_pid"; then
        log "start --for-build: the feeder pause is our caller's ($(head -c 60 "$FEEDER_PAUSE" | tr '\n' ' ')); using its pause and trickle lock"
      else
        if [ -e "$FEEDER_PAUSE" ] && [ "$(head -c ${#FP_MARK} "$FEEDER_PAUSE")" != "$FP_MARK" ]; then
          not_now "the feeder is held by another job ($(head -c 60 "$FEEDER_PAUSE" | tr '\n' ' '))"
        fi
        exec 8>>"$TRICKLE_LOCK"
        flock -n 8 || not_now "a reingest/deletion run holds the trickle lock"
        echo "$FP_MARK $(date '+%F %T')" > "$FEEDER_PAUSE"   # first, so nothing starts between the check and the GPU
      fi
      why=$(rag_busy)
      [ -z "$why" ] || { release_feeder; not_now "$why"; }
    fi
    log "start: pausing the local LLM lane, unloading $RAG_MODEL (vram $(vram) MiB)"
    date -d '+6 hours' +%s > "$PAUSE"
    waited=0
    while rag_loaded; do
      curl -s -m 20 "$OLLAMA/api/generate" -d "{\"model\":\"$RAG_MODEL\",\"keep_alive\":0}" >/dev/null 2>&1
      sleep 5; waited=$((waited+5))
      if [ "$waited" -ge "$UNLOAD_WAIT_S" ]; then log "start: $RAG_MODEL still loaded after $waited s -- giving up"; rm -f "$PAUSE"; release_feeder; exit 1; fi
    done
    log "start: $RAG_MODEL unloaded after $waited s (vram $(vram) MiB); launching"
    # fd 9 (this script's start lock) is closed for the server so a later `start` never waits on it; fd 8, the trickle
    # lock (--for-build only), is inherited on purpose.
    LD_LIBRARY_PATH="$BIN" nohup "$BIN/llama-server" --host 127.0.0.1 --port "$PORT" -m "$GGUF" --mmproj "$MMPROJ" \
      -c 8192 --parallel 1 -ngl 99 --jinja --no-warmup -t 6 -a pepperocr-vl >> "$HERE/ocr_server.server.log" 2>&1 9>&- &
    srv=$!
    echo $srv > "$PID"
    # 2026-10-15: build_corpus workers run at oom_score_adj 1000 (so the kernel kills a worker before lightrag-server, 500)
    # and the server inherits that through the worker that started it. An unprivileged task may lower its own score back
    # to 0, so reset it: the OCR server must rank below the workers, not with them.
    if echo 0 > "/proc/$srv/oom_score_adj" 2>/dev/null; then
      log "start: llama-server pid $srv oom_score_adj reset to $(cat "/proc/$srv/oom_score_adj" 2>/dev/null)"
    else
      log "start: WARNING could not reset oom_score_adj of pid $srv (it is $(cat "/proc/$srv/oom_score_adj" 2>/dev/null || echo unknown))"
    fi
    for i in $(seq 1 90); do healthy && break; sleep 2; done
    if ! healthy; then log "start: no health after 180 s -- stopping"; kill -KILL "$(cat "$PID")" 2>/dev/null; rm -f "$PID" "$PAUSE"; release_feeder; exit 1; fi
    if grep -q 'fit params to free device memory' "$HERE/ocr_server.server.log" 2>/dev/null && [ "$(vram)" -gt 7900 ]; then
      log "start: WARNING the model did not fit the card (spilled to host RAM) -- vram $(vram) MiB"
    fi
    [ "${1:-}" = "--for-build" ] && touch "$MARK"
    log "start: serving pepperocr-vl on :$PORT (vram $(vram) MiB)$( [ "${1:-}" = "--for-build" ] && echo ' [for-build]')"
    ;;
  stop)
    # Same lock as `start`: a start that is still unloading the 7b or launching the server (a worker's child that outlived
    # its build) finishes first, so its server and marker are there to stop. Bounded, so a stuck start cannot hang the
    # build's teardown (build_corpus gives stop 180 s): after the wait it goes on without the lock.
    exec 9>"$LOCK"
    flock -w "$STOP_LOCK_WAIT_S" 9 || log "stop: the start lock is still held after $STOP_LOCK_WAIT_S s; going on without it"
    if [ "${1:-}" = "--if-build" ] && [ ! -e "$MARK" ]; then exit 0; fi
    # 2026-10-15: the pidfile pid is authoritative only while it IS the server. A recycled pid, a dead pid, or a
    # transient /proc read failure must not strand a live pepperocr-vl llama-server as a permanent orphan (nothing
    # later would find it, and the old one holds the GPU while the 7b lane resumes). So before the cleanup that
    # deletes $PID and $MARK, look for the real server and adopt it.
    p=$(cat "$PID" 2>/dev/null || true)
    q=
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null && is_ocr_server "$p"; then
      q=$p
    else
      if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
        log "stop: pid $p in $PID is not the pepperocr-vl llama-server (stale pidfile); not killing it"
      fi
      cand=$(ocr_server_pids | tr '\n' ' '); cand=${cand% }
      case "$cand" in
        '') : ;;   # none: the server really is gone (or never there) -- clean up as before
        *' '*)
          log "stop: WARNING $(set -- $cand; echo $#) processes look like a pepperocr-vl llama-server ($cand) while $PID holds '$p'; killing none, keeping $PID, $MARK and $PAUSE for a hand check"
          exit 0 ;;
        *) q=$cand ;;
      esac
    fi
    if [ -n "$q" ]; then
      kill -TERM "$q"; for i in $(seq 1 10); do kill -0 "$q" 2>/dev/null || break; sleep 1; done
      kill -0 "$q" 2>/dev/null && kill -KILL "$q"
      if [ "$q" = "$p" ]; then log "stop: pid $q stopped"
      else log "stop: pidfile pid $p was stale; stopped the pepperocr-vl server found as pid $q"; fi
    fi
    rm -f "$PID" "$MARK" "$PAUSE"
    release_feeder
    log "stop: local LLM lane resumed (vram $(vram) MiB)"
    ;;
  status)
    echo "serving: $(healthy && echo yes || echo no)  pid: $(cat "$PID" 2>/dev/null || echo -)  for-build: $([ -e "$MARK" ] && echo yes || echo no)"
    echo "feeder pause: $([ -e "$FEEDER_PAUSE" ] && head -c 60 "$FEEDER_PAUSE" | tr '\n' ' ' || echo none)"
    echo "local lane paused: $([ -e "$PAUSE" ] && echo "until $(date -d @"$(cat "$PAUSE")" +%H:%M)" || echo no)  rag model loaded: $(rag_loaded && echo yes || echo no)  vram: $(vram) MiB"
    ;;
  *) echo "usage: $0 start [--for-build] | stop [--if-build] | status"; exit 2 ;;
esac
