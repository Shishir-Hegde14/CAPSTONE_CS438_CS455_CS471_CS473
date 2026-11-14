#!/usr/bin/env bash
# exp=8 pilot on tasks 100-199 using the existing warm-up, at the current
# Ollama context. Runs one task at a time and stops at the first model/API or
# environment infrastructure error, preserving that record for diagnosis.
set -Eeuo pipefail
cd "$(dirname "$0")"

MODEL="llama3.1:8b"
PRIOR="${PRIOR:-results/llama31_8b_os_100_100_100}"
OUT="${OUT:-results/llama31_8b_os_exp8_context_pilot_100}"
GUARD_DIR="${GUARD_LOG_DIR:-${OUT}/guard}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
GUARD_PORT="${GUARD_PORT:-11437}"
BASE_URL="http://127.0.0.1:${GUARD_PORT}/v1"
UPSTREAM="http://127.0.0.1:${OLLAMA_PORT}/v1"

for cmd in python3 curl flock; do
  command -v "$cmd" >/dev/null || { echo "Missing required command: $cmd" >&2; exit 2; }
done
[[ -f data/os_tasks.json ]] || { echo "Missing data/os_tasks.json" >&2; exit 2; }
[[ -s "$PRIOR/run1_warmup.jsonl" && -s "$PRIOR/run2_baseline.jsonl" && -s "$PRIOR/run3_replay.jsonl" ]] || {
  echo "Missing completed original run files under $PRIOR" >&2; exit 2;
}
mkdir -p "$OUT" "$GUARD_DIR"
exec 9>"$OUT/.lock"
flock -n 9 || { echo "Another exp=8 pilot is already running for $OUT." >&2; exit 2; }

# Verify exact seed/control ranges, enough seed examples, and safe resume state.
python3 - "$PRIOR" "$OUT/run_exp8_100_200.jsonl" data/os_tasks.json <<'PY'
import json, sys
from pathlib import Path
prior=Path(sys.argv[1])
def load(name):
    rows=[json.loads(x) for x in (prior/name).read_text().splitlines() if x.strip()]
    ids=[int(r['task_index']) for r in rows]
    if len(ids)!=len(set(ids)): raise SystemExit(f'{name}: duplicate task IDs')
    return rows,set(ids)
w,wi=load('run1_warmup.jsonl')
b,bi=load('run2_baseline.jsonl')
r,ri=load('run3_replay.jsonl')
if wi!=set(range(0,100)) or bi!=set(range(100,200)) or ri!=set(range(100,200)):
    raise SystemExit('Original results are not the expected 0:100 warm-up and 100:200 paired runs.')
if sum(bool(x.get('correct')) and x.get('status')=='completed' for x in w)<8:
    raise SystemExit('Warm-up has fewer than 8 completed correct experiences.')
if len(json.loads(Path(sys.argv[3]).read_text())['tasks'])!=500:
    raise SystemExit('Expected the 500-task OS dataset.')
out=Path(sys.argv[2])
if out.exists():
    rows=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    ids=[int(r['task_index']) for r in rows]
    if len(ids)!=len(set(ids)) or any(i<100 or i>=200 for i in ids):
        raise SystemExit('Existing exp=8 output has duplicate/out-of-range IDs; stop and inspect it.')
    bad=[r for r in rows if r.get('status') in ('infrastructure_error','environment_error')]
    if bad:
        print(json.dumps(bad[0],indent=2))
        raise SystemExit('Existing exp=8 output contains an infrastructure/environment error. Preserve it; do not resume with unchanged settings.')
print('Prior results, 8 replay seeds, and resume file verified.')
PY

curl -fsS --max-time 5 "$UPSTREAM/models" -o "$OUT/ollama_models.json" || {
  echo "Local Ollama is not responding at $UPSTREAM." >&2; exit 2;
}
python3 - "$OUT/ollama_models.json" "$MODEL" <<'PY'
import json,sys
names={m.get('id') for m in json.load(open(sys.argv[1],encoding='utf-8')).get('data',[])}
if sys.argv[2] not in names: raise SystemExit(f'Missing {sys.argv[2]} from local Ollama: {sorted(names)}')
print(f'Local model verified: {sys.argv[2]}')
PY

# Fresh, loopback-only guard budget sized for a 100-task exp=8 pilot.
export OLLAMA_LOCAL_PLACEHOLDER="local-only-not-a-secret"
export GUARD_KEY_ENV=OLLAMA_LOCAL_PLACEHOLDER
export GUARD_CLIENT_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
export GUARD_UPSTREAM_BASE="$UPSTREAM"
export GUARD_ALLOWED_HOSTS="127.0.0.1:${OLLAMA_PORT}"
export GUARD_ALLOWED_PATH=/v1
export GUARD_ALLOWED_MODEL="$MODEL"
export GUARD_PORT GUARD_LOG_DIR="$GUARD_DIR"
export GUARD_MAX_REQUESTS=2500
export GUARD_MAX_NEURONS=0
export GUARD_INPUT_NEURONS_PER_MTOK=0
export GUARD_OUTPUT_NEURONS_PER_MTOK=0
export GUARD_MAX_OUTPUT_TOKENS=512
export GUARD_MAX_TOTAL_OUTPUT_TOKENS=1000000
export GUARD_MAX_TOTAL_TOKENS=12000000
export GUARD_MAX_USD=0
export GUARD_INPUT_USD_PER_MTOK=0
export GUARD_OUTPUT_USD_PER_MTOK=0
export GUARD_MAX_PROMPT_CHARS=250000
export GUARD_MAX_MESSAGES=24
export GUARD_TIMEOUT=600
export GUARD_STRIP_SEED=0
export GUARD_ALLOW_GRAMMAR=0

# Refuse to reuse a stale/unrelated local guard process.
python3 - "$GUARD_PORT" <<'PY'
import socket,sys
s=socket.socket()
try: s.bind(('127.0.0.1',int(sys.argv[1])))
except OSError as e: raise SystemExit(f'Guard port {sys.argv[1]} is occupied; inspect it, do not kill it: {e}')
finally: s.close()
PY
python3 tools/api_guard.py --host 127.0.0.1 --port "$GUARD_PORT" >"$GUARD_DIR/guard.log" 2>&1 &
GUARD_PID=$!
cleanup(){ kill "$GUARD_PID" 2>/dev/null || true; wait "$GUARD_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 50); do
  curl -fsS --max-time 1 "http://127.0.0.1:${GUARD_PORT}/health" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS --max-time 2 "http://127.0.0.1:${GUARD_PORT}/health" >/dev/null || { echo "Guard failed; inspect $GUARD_DIR/guard.log" >&2; exit 2; }

curl -fsS --max-time 120 -H "Authorization: Bearer $GUARD_CLIENT_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly OK\"}],\"temperature\":0,\"seed\":0,\"max_tokens\":8}" \
  "$BASE_URL/chat/completions" > "$OUT/preflight.json"
python3 - "$OUT/preflight.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print('Local Ollama preflight:',repr(r['choices'][0]['message']['content'][:80]))
PY
unset OLLAMA_LOCAL_PLACEHOLDER
export GUARD_CLIENT_TOKEN

COMMON=(--mode api --base-url "$BASE_URL" --api-key-env GUARD_CLIENT_TOKEN
  --model-name "$MODEL" --data data/os_tasks.json --max-rounds 5 --max-tokens 512
  --model-timeout 600 --model-retries 0 --command-timeout 20
  --background-grace 0.5 --resume)
OUTFILE="$OUT/run_exp8_100_200.jsonl"

# One-task canary, then proceed in order and stop on the first infrastructure
# or environment error. This makes any context-window boundary visible.
for i in $(seq 100 199); do
  j=$((i+1))
  if [[ -s "$OUTFILE" ]] && grep -q '"task_index"' "$OUTFILE"; then
    : # --resume below skips task IDs already recorded.
  fi
  echo "=== exp=8 task $i (of 100: $((i-99))/100) ==="
  env -u OLLAMA_LOCAL_PLACEHOLDER python3 tools/run_os_cpu.py "${COMMON[@]}" \
    --tasks "$i:$j" --replay-count 8 --seed-sessions "$PRIOR/run1_warmup.jsonl" \
    --output "$OUTFILE" >> "$OUT/run_exp8.log" 2>&1
  python3 - "$OUTFILE" "$i" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); idx=int(sys.argv[2])
rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
r=next((x for x in rows if int(x['task_index'])==idx),None)
if r is None: raise SystemExit(f'No output record for task {idx}')
print(f"recorded task={idx}, status={r.get('status')}, correct={r.get('correct')}")
if r.get('status') in ('infrastructure_error','environment_error'):
    print('Stopping at first infrastructure/environment error:')
    print('failure_type=',r.get('failure_type'))
    print('model_error=',r.get('model_error'))
    raise SystemExit(10)
PY
done

python3 tools/analyze_os_cpu.py "$PRIOR/run2_baseline.jsonl" "$OUTFILE" \
  --output "$OUT/paired_comparison_vs_baseline.json"
echo "exp=8 pilot finished. Outputs: $OUT"
echo "Inspect run_exp8.log for per-task summaries and guard logs in $GUARD_DIR."
