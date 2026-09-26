# fronts: failover in front of LightRAG's model servers

LightRAG points each LLM role (`EXTRACT`, `KEYWORD`, `QUERY`, ...) and the reranker at exactly
one server, and it has no fallback of its own:

- when an LLM role's server refuses, stalls or errors, every call of that role fails;
- when the reranker fails, LightRAG logs an error and answers from the **unranked** chunks,
  which looks like a normal answer.

The two small HTTP fronts here sit between LightRAG and its model servers. Each one tries a
**primary** backend and replays the request against a **fallback** when the primary cannot
answer. Both use only the Python standard library (Python >= 3.10).

```text
LightRAG ──▶ llm_failover.py    :19520 ──▶ primary  (OpenAI-compatible: llama-server, llama-swap, vLLM, ...)
                                       └─▶ fallback (ollama native /api/chat, or OpenAI-compatible)
LightRAG ──▶ rerank_failover.py :19510 ──▶ primary  /v1/rerank
                                       └─▶ fallback /v1/rerank (e.g. rerank_server.sh)
```

A typical setup uses a faster or larger GPU server as the primary and a small local model
as the fallback, so queries keep working while the primary is busy, restarting or switched off.

| File | What it is |
| --- | --- |
| `llm_failover.py` | `/v1/chat/completions` front with circuit breaker, `/health`, `/v1/models`, optional policies |
| `rerank_failover.py` | `/v1/rerank` front with circuit breaker and `/health` |
| `llm_failover.sh`, `rerank_failover.sh` | supervisors: restart the front 5 s after any exit and append to a log |
| `rerank_server.sh` | example llama.cpp `llama-server --reranking` launcher, usable as a fallback reranker |
| `requirements.txt` | nothing at run time; `pytest` for the tests |

## Quick start

1. Start the fronts. Every setting is an environment variable (tables below):

   ```bash
   LLM_FO_PRIMARY=http://127.0.0.1:8080 \
   LLM_FO_FALLBACK=http://127.0.0.1:11434 LLM_FO_FALLBACK_MODEL=qwen2.5:7b-instruct \
     ./fronts/llm_failover.sh &

   RERANK_FO_PRIMARY=http://127.0.0.1:8080 RERANK_FO_FALLBACK=http://127.0.0.1:8081 \
     ./fronts/rerank_failover.sh &
   ```

   Backend URLs are base URLs **without** `/v1`. The fronts add `/v1/chat/completions`,
   `/v1/rerank`, `/api/chat` and so on.

2. Point LightRAG at them in its `.env`. Use the roles you want protected:

   ```ini
   QUERY_LLM_BINDING=openai
   QUERY_LLM_BINDING_HOST=http://127.0.0.1:19520/v1
   QUERY_LLM_MODEL=<model name the primary serves>
   KEYWORD_LLM_BINDING=openai
   KEYWORD_LLM_BINDING_HOST=http://127.0.0.1:19520/v1
   KEYWORD_LLM_MODEL=<model name the primary serves>

   RERANK_BINDING=cohere
   RERANK_BINDING_HOST=http://127.0.0.1:19510/v1/rerank
   ```

   The fronts neither check nor forward LightRAG's own `*_API_KEY` values. If your LightRAG
   version insists on one, any placeholder works. To give a backend a key, set
   `LLM_FO_PRIMARY_API_KEY` and the related variables (sent as `Authorization: Bearer`).

3. Check: `curl -s http://127.0.0.1:19520/health` and `curl -s http://127.0.0.1:19510/health`.

## How `llm_failover.py` behaves

- **Primary first.** The request body goes to the primary **byte for byte**. The one exception
  is a model with schema shaping enabled (see below).
- **When it falls back:** on a connection error, any non-200 status, or no response within
  `LLM_FO_PRIMARY_TIMEOUT` seconds. A streamed response that has already begun cannot fall
  back, because the client has received bytes.
- **Circuit breaker.** A primary failure sends every request straight to the fallback for
  `LLM_FO_BREAKER_S` seconds. A **4xx** from the primary still falls back, but does **not**
  open the breaker. A 4xx is about that one request (a prompt over the context size, say), not
  the primary's health. Without this rule, a single oversized prompt would move all traffic
  to the fallback.
- **Fallback, `LLM_FO_FALLBACK_API=ollama` (default).** The front calls ollama's native
  `/api/chat` with an explicit `options.num_ctx` (`LLM_FO_NUM_CTX`, else `OLLAMA_LLM_NUM_CTX`,
  else 8192). That is how LightRAG's own ollama binding calls it. The reply is translated back
  to the OpenAI schema, plain or as SSE `chat.completion.chunk` events ending in
  `data: [DONE]`.
  - Request mapping: `temperature` → `options.temperature`, `max_tokens` → `options.num_predict`,
    `seed` → `options.seed`; `response_format` `json_schema` → `format: <schema>` and
    `json_object` → `format: "json"`.
  - Reply mapping: `done_reason: "length"` → `finish_reason: "length"`, and
    `prompt_eval_count`/`eval_count` → `usage`.
  - Other OpenAI fields (`tools`, `stop`, `top_p`, ...) are not forwarded to ollama.
- **Fallback, `LLM_FO_FALLBACK_API=openai`.** The request goes to another OpenAI-compatible
  server and its reply is relayed unchanged. Only `model` is rewritten, and only when
  `LLM_FO_FALLBACK_MODEL` is set.
- **Fallback model.** `LLM_FO_FALLBACK_MODEL` names the fallback's model. Empty (default) means
  the fallback receives the model named in the request.
- **Both down.** The response is `502` with a JSON `error`.
- **`X-LLM-Failover-Backend: primary|fallback`** is set on every completion.
- **`GET /health`** returns `200` when either backend answers and `503` when neither does. The
  JSON body includes `primary`, `fallback` and `breaker_open`, plus the state of every optional
  policy (`guard`, `quiet`, `yield`, `reserved`, `shared`, `schema_models`). The primary is
  probed at `/v1/models`. The fallback is probed at `/api/tags` (ollama) or `/v1/models`
  (openai).
- **`GET /v1/models`** lists `LLM_FO_PRIMARY_MODELS` plus `LLM_FO_FALLBACK_MODEL`. It is a
  static list, not a proxy.
- **Log.** One line per request, backend first, with the reason for any fallback (`breaker`,
  `quiet`, `reserved`, `guarded`, `yield`, `shared`).

## How `rerank_failover.py` behaves

- `POST /v1/rerank` (the Cohere/Jina-style body that llama.cpp, llama-swap and vLLM accept)
  is forwarded unchanged to the primary. On a connection error, any non-200 status or a
  timeout, it is replayed against the fallback's `/v1/rerank`, and the fallback's status and
  body are returned as they are.
- A primary failure opens the breaker for `RERANK_FO_BREAKER_S` seconds. Unlike the LLM front,
  this includes a 4xx.
- `GET /health` probes the primary's `/v1/models` and the fallback's `/health` (llama-server
  has both). It returns `200` when either answers, `503` when neither, with
  `{"status", "primary", "fallback", "breaker_open"}`.

## Settings: `llm_failover.py`

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_FO_HOST` | `127.0.0.1` | listen address. The front has no authentication; keep it on loopback |
| `LLM_FO_PORT` | `19520` | listen port |
| `LLM_FO_PRIMARY` | `http://127.0.0.1:8080` | primary OpenAI-compatible base URL (no `/v1`) |
| `LLM_FO_PRIMARY_API_KEY` | *(empty)* | bearer key sent to the primary |
| `LLM_FO_PRIMARY_MODELS` | *(empty)* | comma list reported by `/v1/models` |
| `LLM_FO_FALLBACK` | `http://127.0.0.1:11434` | fallback base URL (ollama's default port) |
| `LLM_FO_FALLBACK_API` | `ollama` | `ollama` (native `/api/chat`, translated) or `openai` |
| `LLM_FO_FALLBACK_API_KEY` | *(empty)* | bearer key sent to the fallback |
| `LLM_FO_FALLBACK_MODEL` | *(empty)* | fallback model name. Empty means the requested name |
| `LLM_FO_NUM_CTX` | `$OLLAMA_LLM_NUM_CTX`, else `8192` | ollama `num_ctx` for fallback calls |
| `LLM_FO_PRIMARY_TIMEOUT` | `180` | seconds to wait for the primary's first byte |
| `LLM_FO_FALLBACK_TIMEOUT` | `590` | keep it just under LightRAG's `LLM_TIMEOUT` / `*_LLM_TIMEOUT` |
| `LLM_FO_BREAKER_S` | `60` | breaker hold time after a primary failure |
| `LLM_FO_LOG` | *(empty = stderr)* | log file (the launcher sets `fronts/llm_failover.log`) |

LightRAG's timeout for the role must cover `LLM_FO_PRIMARY_TIMEOUT` plus a fallback answer.
Otherwise LightRAG gives up before the front has had a chance to fall back.

### Optional policies (all off by default)

These policies exist for a primary whose GPU also serves other work. The first four apply
only to requests whose `model` is in `LLM_FO_GUARDED_MODELS` (default `*`, every model), and
each stays off until its own variable is set.

| Variable | Default | Policy |
| --- | --- | --- |
| `LLM_FO_GUARDED_MODELS` | `*` | which requested models the quiet window, reservation, guard and yield apply to |
| `LLM_FO_QUIET_WINDOW` | *(off)* | `HH:MM-HH:MM` local time, may cross midnight. Guarded requests use the fallback throughout |
| `LLM_FO_RESERVED_FILE` | *(off)* | path to a file holding an epoch second. Until then, guarded requests use the fallback |
| `LLM_FO_ALLOWED_RUNNING` | *(off)* | llama-swap guard: use the primary only while every model in its `GET /running` is on this comma list |
| `LLM_FO_YIELD_EVERY_S` | `0` (off) | shared + guarded models: periodically drain the primary so a queued model load can swap in |
| `LLM_FO_YIELD_HOLD_S` / `LLM_FO_YIELD_MAX_S` | `3` / `120` | idle hold after the drain / give up waiting for a drain |
| `LLM_FO_SHARED_MODELS` | *(off)* | comma list: use the fallback as a second worker for these models |
| `LLM_FO_PRIMARY_SLOTS` / `LLM_FO_FALLBACK_SLOTS` | `4` / `2` | concurrent calls per side (match `--parallel` / `OLLAMA_NUM_PARALLEL`) |
| `LLM_FO_FALLBACK_MAX_CHARS` | `20000` | longer prompts stay on the primary (about 6k tokens, which fits `num_ctx` 8192) |
| `LLM_FO_SLOT_WAIT_S` | `90` | how long a shared request waits for a free slot before queueing on the primary |
| `LLM_FO_RETRY_502_S` | `4` | shared models: one retry this many seconds after a primary 502/503 |
| `LLM_FO_RETRY_READY_S` | `0` | before that retry, poll llama-swap `/running` up to this long for the model to be ready (40 works) |
| `LLM_FO_SCHEMA_MODELS` | `$LLM_FO_SHARED_MODELS` | attach LightRAG's extraction JSON schema per request (see below) |

- **Quiet window.** llama-swap does not swap a model out while requests are in flight. A steady
  stream of ingest calls therefore starves any other model requested on the same server, and
  that request waits for a pause that never comes. If the primary's GPU runs scheduled jobs,
  name their window here. Start it a few minutes early so in-flight calls can drain. A
  malformed value stops the front at start-up.
- **Reservation file.** This covers unscheduled work, such as a benchmark or a manual job:
  `date -d '+2 hours' +%s > "$LLM_FO_RESERVED_FILE"` reserves the primary, and `rm` releases it.
  A missing file, an unreadable file or a time already past means no reservation. Use an
  absolute path; the launcher runs the front from `fronts/`.
- **Running-model guard.** A small model in a llama-swap group that does not swap loads *beside*
  a resident large model and pushes that model's buffers out of VRAM. With the guard on, guarded
  requests use the fallback unless every running model is on the allow list. A failed probe
  also means fallback. The verdict is cached for 1 s.
- **Shared dispatch.** This is meant for the ingest (extraction) role when both sides serve
  the same weights. Each call goes to the side with a free slot, primary first, so both GPUs
  work. Prompts too long for the fallback's context wait for the primary.
- **Schema shaping.** A role's `EXTRA_BODY` applies to every call of that role, including the
  entity/relation merge summaries, which must stay prose. So for `LLM_FO_SCHEMA_MODELS` the
  front decides per request:
  - a system prompt carrying LightRAG's JSON extraction template (`"entities"` and
    `"relationships"`, used with `ENTITY_EXTRACTION_USE_JSON=true`) gets a `json_schema`
    `response_format` (and ollama `format`);
  - any other prompt has `response_format` removed.

  Use it when a llama.cpp build ignores `response_format: json_object` but enforces
  `json_schema`. Set it to an empty value to turn it off while shared dispatch stays on.

## Settings: `rerank_failover.py`

| Variable | Default | Meaning |
| --- | --- | --- |
| `RERANK_FO_HOST` | `127.0.0.1` | listen address. No authentication; keep it on loopback |
| `RERANK_FO_PORT` | `19510` | listen port |
| `RERANK_FO_PRIMARY` | `http://127.0.0.1:8080` | primary base URL (no `/v1`) |
| `RERANK_FO_FALLBACK` | `http://127.0.0.1:8081` | fallback base URL (`rerank_server.sh`'s default port) |
| `RERANK_FO_PRIMARY_API_KEY` / `RERANK_FO_FALLBACK_API_KEY` | *(empty)* | bearer keys |
| `RERANK_FO_PRIMARY_TIMEOUT` | `90` | seconds before the primary counts as failed |
| `RERANK_FO_FALLBACK_TIMEOUT` | `290` | keep it just under LightRAG's `RERANK_TIMEOUT` |
| `RERANK_FO_BREAKER_S` | `60` | breaker hold time |
| `RERANK_FO_LOG` | *(empty = stderr)* | log file (the launcher sets `fronts/rerank_failover.log`) |

LightRAG's `RERANK_TIMEOUT` defaults to 30 s in 1.5.7. That suits hosted APIs, but not a local
reranker scoring ~50 chunks, and it is shorter than `RERANK_FO_PRIMARY_TIMEOUT`. With these
fronts, raise it (for example `RERANK_TIMEOUT=300`) so the fallback has time to answer.

## Running them

`llm_failover.sh` and `rerank_failover.sh` restart their front 5 s after any exit and append
its output to `LLM_FO_LOG` / `RERANK_FO_LOG`, by default a `.log` file beside the script.
Before starting, each sources an env file if one exists: `LLM_FO_ENV_FILE` /
`RERANK_FO_ENV_FILE`, by default `fronts/llm_failover.env` / `fronts/rerank_failover.env`.
`PYTHON` selects the interpreter (default `python3`).

A systemd user unit is one way to keep a front up:

```ini
# ~/.config/systemd/user/llm-failover.service
[Unit]
Description=LLM failover front for LightRAG

[Service]
# The leading "-" makes the env file optional, as it is for the launcher.
EnvironmentFile=-%h/LightRAG/zotero-toolkit/fronts/llm_failover.env
# Standard library only: any Python >= 3.10 works; name it explicitly.
ExecStart=/usr/bin/python3 %h/LightRAG/zotero-toolkit/fronts/llm_failover.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

`tmux`, `nohup` or an `@reboot` crontab line with the `.sh` launcher work too.

The fronts use `urllib`, which honours `http_proxy` / `no_proxy`. If you set a proxy, add the
backend hosts to `no_proxy`.

### Example fallback reranker: `rerank_server.sh`

This script runs llama.cpp's `llama-server --reranking` on `127.0.0.1:8081` in a restart loop.
Set `RERANK_MODEL_PATH` to a reranker GGUF, for example a Qwen3-Reranker-0.6B or
bge-reranker-v2-m3 quantization. Optional settings: `LLAMA_SERVER` (binary, default
`llama-server` on `PATH`), `LLAMA_LIB_DIR`, `RERANK_SERVER_HOST`, `RERANK_SERVER_PORT`,
`RERANK_NGL` (default `0`, CPU), `RERANK_UBATCH` (default `2048`, used for `-c`/`-b`/`-ub`),
`RERANK_ALIAS` and `RERANK_SERVER_LOG`. The script's header explains the sizing: `-ub` is a
hard per-sequence cap for rank pooling, and `--parallel 1` is the right setting for a small
reranker.

## Tests

The tests run offline, against stdlib fake servers on ephemeral loopback ports. From the
`zotero-toolkit/` directory:

```bash
python -m pip install -r fronts/requirements.txt    # pytest only
python -m pytest tests/fronts
```

They cover:

- success via the primary;
- fallback on a refused connection, HTTP 500, a 4xx (breaker stays closed) and a slow first byte;
- the breaker opening and closing;
- ollama → OpenAI translation, plain and SSE;
- the OpenAI-compatible fallback and bearer keys;
- `/health` and `/v1/models`;
- every optional policy;
- the rerank front's fallback, breaker and health.
