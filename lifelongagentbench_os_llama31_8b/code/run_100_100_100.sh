#!/usr/bin/env bash
# One approved 100/100/100 OS evaluation block using local Ollama llama3.1:8b.
# Ollama must already be running on 127.0.0.1:11434 with the model loaded.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="llama3.1:8b"
OUT="${OUT:-results/llama31_8b_os_100_100_100}"
GUARD_DIR="${GUARD_LOG_DIR:-${OUT}_guard}"
GUARD_PORT="${GUARD_PORT:-11436}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
BASE_URL="http://127.0.0.1:${GUARD_PORT}/v1"
UPSTREAM="http://127.0.0.1:${OLLAMA_PORT}/v1"
mkdir -p "$OUT" "$GUARD_DIR"

for command in python3 curl; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 2; }
done
python3 - <<'PY'
import platform, sys
if sys.platform != 'linux':
    raise SystemExit('Run this inside a Linux WSL2 distribution, not PowerShell.')
if platform.machine() not in ('x86_64', 'amd64'):
    raise SystemExit('This kit currently supports Linux x86_64 only.')
PY
HOST_USR="${OS_SANDBOX_HOST_USR:-$HOME/.cache/lab_8b_kit/usr}"
if [[ ! -d "$HOST_USR/bin" || ! -d "$HOST_USR/lib" ]]; then
  echo "Missing the user-owned /usr snapshot at $HOST_USR." >&2
  echo "Run: bash prepare_wsl_usr.sh" >&2
  exit 2
fi
export OS_SANDBOX_HOST_USR="$HOST_USR"

# Verify the local Ollama endpoint before starting the guard.
if ! curl -fsS --max-time 3 "${UPSTREAM}/models" -o "$OUT/ollama_models.json"; then
  echo "Ollama is not reachable at ${UPSTREAM}. Start it inside WSL with OLLAMA_CONTEXT_LENGTH=8192 and pull ${MODEL}." >&2
  exit 2
fi
python3 - "$OUT/ollama_models.json" "$MODEL" <<'PY'
import json, sys
models = json.load(open(sys.argv[1], encoding='utf-8')).get('data', [])
names = {m.get('id') for m in models}
if sys.argv[2] not in names:
    raise SystemExit(f"Expected {sys.argv[2]} in Ollama /v1/models; found {sorted(names)}. Run `ollama pull {sys.argv[2]}`.")
print(f"Ollama model available: {sys.argv[2]}")
PY

# Ollama's local OpenAI-compatible endpoint ignores bearer auth. The guard still
# requires a non-empty upstream token; this placeholder is not a credential.
export OLLAMA_LOCAL_PLACEHOLDER="local-only-not-a-secret"
export GUARD_KEY_ENV=OLLAMA_LOCAL_PLACEHOLDER
export GUARD_CLIENT_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
export GUARD_UPSTREAM_BASE="$UPSTREAM"
export GUARD_ALLOWED_HOSTS="127.0.0.1:${OLLAMA_PORT}"
export GUARD_ALLOWED_PATH="/v1"
export GUARD_ALLOWED_MODEL="$MODEL"
export GUARD_PORT GUARD_LOG_DIR="$GUARD_DIR"
export GUARD_MAX_REQUESTS="1200"
# Local Ollama has no hosted Neuron quota; disable hosted-only accounting caps.
export GUARD_MAX_NEURONS="0"
export GUARD_INPUT_NEURONS_PER_MTOK="0"
export GUARD_OUTPUT_NEURONS_PER_MTOK="0"
export GUARD_MAX_OUTPUT_TOKENS="512"
export GUARD_MAX_TOTAL_OUTPUT_TOKENS="600000"
export GUARD_MAX_TOTAL_TOKENS="2500000"
export GUARD_MAX_USD="0"
export GUARD_INPUT_USD_PER_MTOK="0"
export GUARD_OUTPUT_USD_PER_MTOK="0"
export GUARD_MAX_PROMPT_CHARS="250000"
export GUARD_MAX_MESSAGES="24"
export GUARD_TIMEOUT="600"
export GUARD_STRIP_SEED="0"
export GUARD_ALLOW_GRAMMAR="0"

python3 tools/api_guard.py --host 127.0.0.1 --port "$GUARD_PORT" >"$GUARD_DIR/guard.log" 2>&1 &
GUARD_PID=$!
echo "$GUARD_PID" > "$GUARD_DIR/guard.pid"
cleanup() {
  kill "$GUARD_PID" 2>/dev/null || true
  wait "$GUARD_PID" 2>/dev/null || true
  rm -f "$GUARD_DIR/guard.pid"
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 50); do
  if curl -fsS --max-time 1 "http://127.0.0.1:${GUARD_PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 0.2
done
curl -fsS --max-time 2 "http://127.0.0.1:${GUARD_PORT}/health" >/dev/null || {
  echo "Guard did not start; inspect $GUARD_DIR/guard.log" >&2; exit 1;
}

# Tiny local-only request confirms the guard can reach Ollama. It is not scored.
curl -fsS --max-time 120 \
  -H "Authorization: Bearer $GUARD_CLIENT_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly OK\"}],\"temperature\":0,\"seed\":0,\"max_tokens\":8}" \
  "$BASE_URL/chat/completions" > "$OUT/preflight.json"
python3 - "$OUT/preflight.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], encoding='utf-8'))
print('Ollama preflight:', repr(r['choices'][0]['message']['content'][:80]))
PY

COMMON=(
  --mode api --base-url "$BASE_URL" --api-key-env GUARD_CLIENT_TOKEN
  --model-name "$MODEL" --data data/os_tasks.json
  --max-rounds 5 --max-tokens 512 --model-timeout 600 --model-retries 0
  --command-timeout 20 --background-grace 0.5 --resume
)
export GUARD_CLIENT_TOKEN
unset OLLAMA_LOCAL_PLACEHOLDER

echo '=== Run 1: warm-up tasks 0-99 ==='
env -u OLLAMA_LOCAL_PLACEHOLDER python3 tools/run_os_cpu.py "${COMMON[@]}" \
  --tasks 0:100 --replay-count 0 --output "$OUT/run1_warmup.jsonl" > "$OUT/run1.log"

echo '=== Run 2: baseline tasks 100-199 ==='
env -u OLLAMA_LOCAL_PLACEHOLDER python3 tools/run_os_cpu.py "${COMMON[@]}" \
  --tasks 100:200 --replay-count 0 --output "$OUT/run2_baseline.jsonl" > "$OUT/run2.log"

echo '=== Run 3: replay=4, same tasks 100-199 ==='
env -u OLLAMA_LOCAL_PLACEHOLDER python3 tools/run_os_cpu.py "${COMMON[@]}" \
  --tasks 100:200 --replay-count 4 --seed-sessions "$OUT/run1_warmup.jsonl" \
  --output "$OUT/run3_replay.jsonl" > "$OUT/run3.log"

python3 tools/check_replay_validity.py \
  --warmup "$OUT/run1_warmup.jsonl" --replay "$OUT/run3_replay.jsonl" \
  --replay-count 4 --output "$OUT/replay_validity.json"
python3 tools/analyze_os_cpu.py "$OUT/run2_baseline.jsonl" "$OUT/run3_replay.jsonl" \
  --output "$OUT/paired_comparison.json"
echo "Finished. Results: $OUT"
