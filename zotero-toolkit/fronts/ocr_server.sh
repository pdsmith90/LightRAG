#!/usr/bin/env bash
# ocr_server.sh — PepperOCR-VL on the VM's own GPU, ON DEMAND (2026-10-07; operator decision: "the ocr model can supplant the
# rag model if ocr is needed"). The 5700 XT (8 GB) cannot hold it beside the ollama 7b (test 2026-10-07: spilled to
# host RAM, 4.5 tok/s, the VM at 0.9 GB available), so `start` SUPPLANTS the RAG model:
#   1. write .local_llm_paused (epoch "until") -- llm_failover.py stops sending the extraction lane to the local
#      ollama (the primary when allowed, else 503 so LightRAG retries later); in-flight local calls finish (<= 590 s);
#   2. ask ollama to drop the 7b (keep_alive 0) until /api/ps no longer lists it;
#   3. launch llama-server (Vulkan build) with the model + mmproj as alias pepperocr-vl on :19530.
# `stop` kills it (TERM, then KILL: it ignores TERM mid-generation), removes the pause -- the 7b reloads on the
# next local call. build_corpus.py calls `start --for-build` from a worker that needs whole-document OCR and
# `stop --if-build` when the build ends; the marker file keeps a hand-started server out of the build's teardown.
#   ocr_server.sh start [--for-build]   idempotent; flock so three build workers do not race
#   ocr_server.sh stop  [--if-build]    --if-build: only if `start --for-build` started it
#   ocr_server.sh status
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${OCR_LLAMA_BIN:-$HOME/llama-vulkan/llama-b11371}
GGUF=${OCR_GGUF:-$HOME/models/PepperOCR-VL-GGUF/PepperOCR-VL.Q8_0.gguf}
MMPROJ=${OCR_MMPROJ:-$HOME/models/PepperOCR-VL-GGUF/PepperOCR-VL.mmproj-Q8_0.gguf}
PORT=${OCR_PORT:-19530}
OLLAMA=${OCR_OLLAMA:-http://127.0.0.1:11436}
RAG_MODEL=${OCR_RAG_MODEL:-qwen2.5:7b-instruct}
PAUSE=$HERE/.local_llm_paused; PID=$HERE/.ocr_server.pid; MARK=$HERE/.ocr_server.started_by_build
LOCK=$HERE/.ocr_server.lock; LOG=$HERE/ocr_server.log
STOP_LOCK_WAIT_S=${OCR_STOP_LOCK_WAIT_S:-100}   # `stop` waits this long for a `start` in progress (build_corpus allows it 180 s)
UNLOAD_WAIT_S=${OCR_UNLOAD_WAIT_S:-720}   # > llm_failover's FALLBACK_TIMEOUT (590 s): the last in-flight local call
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
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
healthy() { curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; }
rag_loaded() { curl -s -m 5 "$OLLAMA/api/ps" 2>/dev/null | grep -q "\"$RAG_MODEL\""; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/ {printf "%d", $NF/1048576}'; }

cmd=${1:-status}; shift || true
case "$cmd" in
  start)
    exec 9>"$LOCK"; flock 9
    if healthy; then log "start: already serving on :$PORT"; exit 0; fi
    [ -x "$BIN/llama-server" ] && [ -f "$GGUF" ] && [ -f "$MMPROJ" ] || { log "start: binary or GGUFs missing"; exit 2; }
    log "start: pausing the local LLM lane, unloading $RAG_MODEL (vram $(vram) MiB)"
    date -d '+6 hours' +%s > "$PAUSE"
    waited=0
    while rag_loaded; do
      curl -s -m 20 "$OLLAMA/api/generate" -d "{\"model\":\"$RAG_MODEL\",\"keep_alive\":0}" >/dev/null 2>&1
      sleep 5; waited=$((waited+5))
      if [ "$waited" -ge "$UNLOAD_WAIT_S" ]; then log "start: $RAG_MODEL still loaded after $waited s -- giving up"; rm -f "$PAUSE"; exit 1; fi
    done
    log "start: $RAG_MODEL unloaded after $waited s (vram $(vram) MiB); launching"
    # fd 9 (this script's start lock) is closed for the server, so a later `start` or `stop` never waits on it.
    LD_LIBRARY_PATH="$BIN" nohup "$BIN/llama-server" --host 127.0.0.1 --port "$PORT" -m "$GGUF" --mmproj "$MMPROJ" \
      -c 8192 --parallel 1 -ngl 99 --jinja --no-warmup -t 6 -a pepperocr-vl >> "$HERE/ocr_server.server.log" 2>&1 9>&- &
    srv=$!
    echo $srv > "$PID"
    # build_corpus workers run at oom_score_adj 1000 (so the kernel kills a worker first), and the server
    # inherits that through the worker that started it. An unprivileged task may lower its own score back
    # to 0, so reset it: the OCR server must rank below the workers, not with them.
    if echo 0 > "/proc/$srv/oom_score_adj" 2>/dev/null; then
      log "start: llama-server pid $srv oom_score_adj reset to $(cat "/proc/$srv/oom_score_adj" 2>/dev/null)"
    else
      log "start: WARNING could not reset oom_score_adj of pid $srv (it is $(cat "/proc/$srv/oom_score_adj" 2>/dev/null || echo unknown))"
    fi
    for i in $(seq 1 90); do healthy && break; sleep 2; done
    if ! healthy; then log "start: no health after 180 s -- stopping"; kill -KILL "$(cat "$PID")" 2>/dev/null; rm -f "$PID" "$PAUSE"; exit 1; fi
    if grep -q 'fit params to free device memory' "$HERE/ocr_server.server.log" 2>/dev/null && [ "$(vram)" -gt 7900 ]; then
      log "start: WARNING the model did not fit the card (spilled to host RAM) -- vram $(vram) MiB"
    fi
    [ "${1:-}" = "--for-build" ] && touch "$MARK"
    log "start: serving pepperocr-vl on :$PORT (vram $(vram) MiB)$( [ "${1:-}" = "--for-build" ] && echo ' [for-build]')"
    ;;
  stop)
    # Same lock as `start`: a start that is still unloading the RAG model or launching the server (a
    # worker's child that outlived its build) finishes first, so its server and marker are there to stop.
    # Bounded, so a stuck start cannot hang the build's teardown (build_corpus gives stop 180 s): after the
    # wait it goes on without the lock.
    exec 9>"$LOCK"
    flock -w "$STOP_LOCK_WAIT_S" 9 || log "stop: the start lock is still held after $STOP_LOCK_WAIT_S s; going on without it"
    if [ "${1:-}" = "--if-build" ] && [ ! -e "$MARK" ]; then exit 0; fi
    # The pidfile pid is authoritative only while it IS the server. A recycled pid, a dead pid, or a
    # transient /proc read failure must not strand a live pepperocr-vl llama-server as a permanent orphan
    # (nothing later would find it, and it holds the GPU while the RAG model's lane resumes). So before the
    # cleanup that deletes $PID and $MARK, look for the real server and adopt it.
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
    log "stop: local LLM lane resumed (vram $(vram) MiB)"
    ;;
  status)
    echo "serving: $(healthy && echo yes || echo no)  pid: $(cat "$PID" 2>/dev/null || echo -)  for-build: $([ -e "$MARK" ] && echo yes || echo no)"
    echo "local lane paused: $([ -e "$PAUSE" ] && echo "until $(date -d @"$(cat "$PAUSE")" +%H:%M)" || echo no)  rag model loaded: $(rag_loaded && echo yes || echo no)  vram: $(vram) MiB"
    ;;
  *) echo "usage: $0 start [--for-build] | stop [--if-build] | status"; exit 2 ;;
esac
