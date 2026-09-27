#!/usr/bin/env bash
# Example: a local llama.cpp reranker to serve as rerank_failover.py's fallback (or primary).
# Restarts llama-server 10 s after any exit. Settings (only the model is required):
#   RERANK_MODEL_PATH   reranker GGUF, e.g. a Qwen3-Reranker-0.6B or bge-reranker-v2-m3 quant
#   LLAMA_SERVER        llama-server binary (default: llama-server on PATH)
#   LLAMA_LIB_DIR       prepended to LD_LIBRARY_PATH, for a self-built llama.cpp (optional)
#   RERANK_SERVER_HOST  default 127.0.0.1
#   RERANK_SERVER_PORT  default 8081 (rerank_failover.py's default fallback)
#   RERANK_NGL          layers offloaded to the GPU: 0 = CPU (default), 99 = all
#   RERANK_UBATCH       -c, -b and -ub (default 2048)
#   RERANK_ALIAS        model name reported by /v1/models (optional)
#   RERANK_SERVER_LOG   default rerank_server.log beside this script
#
# Sizing notes for LightRAG's rerank traffic:
#   * -ub is a HARD per-sequence cap: rank pooling cannot split a sequence across ubatches, so
#     an overflow is a FAILED rerank, not a slow one. Each sequence is the query plus one chunk,
#     and LightRAG's RERANK_MAX_TOKENS_PER_DOC counts tiktoken tokens while the reranker's own
#     tokenizer may emit more. With RERANK_MAX_TOKENS_PER_DOC=1024, real sequences can run to
#     about 1.5 times that in the reranker's tokens: measure your own traffic and size -ub from
#     its maximum plus headroom, never from one observed request.
#   * --parallel 1: a 0.6B reranker already saturates a GPU with one ~1000-token sequence; two
#     slots were slower per document, and -c must grow with the slot count.
#   * On a small GPU shared with the LLM, offloading the reranker can make the LLM runner evict
#     models. The symptom is silent: reranks time out and LightRAG answers from unranked chunks.
#     CPU (RERANK_NGL=0) is slower but cannot cause that; if you offload, watch for evictions.
: "${RERANK_MODEL_PATH:?set RERANK_MODEL_PATH to a reranker GGUF}"
here="$(cd "$(dirname "$0")" && pwd)" || exit 1
bin="${LLAMA_SERVER:-llama-server}"
log="${RERANK_SERVER_LOG:-$here/rerank_server.log}"
ub="${RERANK_UBATCH:-2048}"
args=(--host "${RERANK_SERVER_HOST:-127.0.0.1}" --port "${RERANK_SERVER_PORT:-8081}"
      -m "$RERANK_MODEL_PATH" --reranking -c "$ub" --parallel 1 -b "$ub" -ub "$ub"
      -ngl "${RERANK_NGL:-0}" --no-warmup)
if [ -n "${RERANK_ALIAS:-}" ]; then args+=(-a "$RERANK_ALIAS"); fi
if [ -n "${LLAMA_LIB_DIR:-}" ]; then export LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
while :; do
  "$bin" "${args[@]}" >> "$log" 2>&1
  rc=$?
  echo "$(date -Is) rerank server exited rc=$rc - restarting in 10s" >> "$log"
  sleep 10
done
