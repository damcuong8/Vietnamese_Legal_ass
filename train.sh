#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SERVER_URL="http://127.0.0.1:8002"
SERVER_LOG="${SERVER_LOG:-llama-server-8002.log}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

wait_for_server() {
  local deadline=$((SECONDS + 600))

  while true; do
    if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1 || \
       curl -fsS "$SERVER_URL/v1/models" >/dev/null 2>&1; then
      return 0
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "llama-server exited before becoming ready. Last log lines:" >&2
      tail -n 80 "$SERVER_LOG" >&2 || true
      exit 1
    fi

    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for llama-server at $SERVER_URL. Last log lines:" >&2
      tail -n 80 "$SERVER_LOG" >&2 || true
      exit 1
    fi

    sleep 2
  done
}

trap cleanup EXIT INT TERM

: > "$SERVER_LOG"

CUDA_VISIBLE_DEVICES=0 ../llama.cpp/build/bin/llama-server \
  -m model_cache/Qwen3.5-9B-UD-Q6_K_XL.gguf \
  --alias qwen35-9b,qwen3.5-9b \
  --host 127.0.0.1 \
  --port 8002 \
  -c 132000 \
  -ngl 999 \
  --parallel 8 \
  --kv-unified \
  --flash-attn on \
  --metrics \
  --jinja \
  --reasoning auto \
  --reasoning-format deepseek \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "llama-server started with PID $SERVER_PID. Log: $SERVER_LOG"
wait_for_server
echo "llama-server is ready at $SERVER_URL"

env \
  CUDA_VISIBLE_DEVICES=0 \
  LLM_BASE_URL=http://127.0.0.1:8002/v1 \
  LANGSMITH_TRACING=false \
  CHECKPOINTER_BACKEND=none \
  COMPRESS_LLM_ENABLE_THINKING=false \
  python plz_5h_left.py \
    --llm-backend default \
    --workers 8 \
    --start-id 650 \
    --end-id 1000 \
    --graph-mode stateless \
    --out-dir submission_eval_llama_smoke_2 \
    --results-name results.json \
    --no-zip
