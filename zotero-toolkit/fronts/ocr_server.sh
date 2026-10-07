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
UNLOAD_WAIT_S=${OCR_UNLOAD_WAIT_S:-720}   # > llm_failover's FALLBACK_TIMEOUT (590 s): the last in-flight local call
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
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
    LD_LIBRARY_PATH="$BIN" nohup "$BIN/llama-server" --host 127.0.0.1 --port "$PORT" -m "$GGUF" --mmproj "$MMPROJ" \
      -c 8192 --parallel 1 -ngl 99 --jinja --no-warmup -t 6 -a pepperocr-vl >> "$HERE/ocr_server.server.log" 2>&1 &
    echo $! > "$PID"
    for i in $(seq 1 90); do healthy && break; sleep 2; done
    if ! healthy; then log "start: no health after 180 s -- stopping"; kill -KILL "$(cat "$PID")" 2>/dev/null; rm -f "$PID" "$PAUSE"; exit 1; fi
    if grep -q 'fit params to free device memory' "$HERE/ocr_server.server.log" 2>/dev/null && [ "$(vram)" -gt 7900 ]; then
      log "start: WARNING the model did not fit the card (spilled to host RAM) -- vram $(vram) MiB"
    fi
    [ "${1:-}" = "--for-build" ] && touch "$MARK"
    log "start: serving pepperocr-vl on :$PORT (vram $(vram) MiB)$( [ "${1:-}" = "--for-build" ] && echo ' [for-build]')"
    ;;
  stop)
    if [ "${1:-}" = "--if-build" ] && [ ! -e "$MARK" ]; then exit 0; fi
    p=$(cat "$PID" 2>/dev/null || true)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      kill -TERM "$p"; for i in $(seq 1 10); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
      kill -0 "$p" 2>/dev/null && kill -KILL "$p"
      log "stop: pid $p stopped"
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
