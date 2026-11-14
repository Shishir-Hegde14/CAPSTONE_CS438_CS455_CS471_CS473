#!/usr/bin/env bash
# Continue the approved local Ollama Llama 3.1 8B OS experiment on tasks 200-499.
# Requires the original WSL kit/results under ~/lab_8b_kit.
# Local loopback only: no hosted inference, provider key, or paid service.
set -Eeuo pipefail
cd "$(dirname "$0")"

MODEL="llama3.1:8b"
PRIOR="${PRIOR:-results/llama31_8b_os_100_100_100}"
OUT="${OUT:-results/llama31_8b_os_200_500_extension}"
GUARD_DIR="${GUARD_LOG_DIR:-${OUT}/guard}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
GUARD_PORT="${GUARD_PORT:-11437}"
BASE_URL="http://127.0.0.1:${GUARD_PORT}/v1"
UPSTREAM="http://127.0.0.1:${OLLAMA_PORT}/v1"

for command in python3 curl flock; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 2; }
done
[[ -f data/os_tasks.json ]] || { echo "Missing data/os_tasks.json" >&2; exit 2; }
[[ -f "$PRIOR/run1_warmup.jsonl" && -f "$PRIOR/run2_baseline.jsonl" && -f "$PRIOR/run3_replay.jsonl" ]] || {
  echo "Missing original warm-up/baseline/replay JSONL files under $PRIOR" >&2; exit 2;
}
mkdir -p "$OUT" "$GUARD_DIR"
exec 9>"$OUT/.lock"
flock -n 9 || { echo "Another extension launcher holds $OUT/.lock; do not start a second copy." >&2; exit 2; }

# Validate the prior block and dataset before spending model calls.
python3 - "$PRIOR" data/os_tasks.json <<'PY'
import json, sys
from pathlib import Path
prior = Path(sys.argv[1])
def read(name):
    rows = [json.loads(x) for x in (prior / name).read_text().splitlines() if x.strip()]
    ids = [int(x['task_index']) for x in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f'{name}: duplicate task indices')
    return rows, set(ids)
w, wi = read('run1_warmup.jsonl')
b, bi = read('run2_baseline.jsonl')
r, ri = read('run3_replay.jsonl')
if wi != set(range(0,100)) or bi != set(range(100,200)) or ri != set(range(100,200)):
    raise SystemExit('Prior JSONL files do not contain exactly the expected task ranges 0:100 and 100:200.')
if len(json.loads(Path(sys.argv[2]).read_text())['tasks']) != 500:
    raise SystemExit('Expected the published 500-task dataset.')
if sum(x.get('correct') and x.get('status') == 'completed' for x in w) < 4:
    raise SystemExit('Warm-up has fewer than four completed correct experiences.')
if len(r[0].get('replay_task_indices', [])) != 4:
    raise SystemExit('Existing replay file does not start with four examples; stop and inspect it.')
print('Prior results and 500-task dataset verified.')
PY

# On an intentional resume, remove only infrastructure-error rows so those task IDs
# can be retried. Task-limit/format/ordinary incorrect outcomes are kept unchanged.
retry_infra_rows() {
  python3 - "$1" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(0)
rows = [x for x in p.read_text().splitlines() if x.strip()]
keep, removed = [], 0
for line in rows:
    row = json.loads(line)
    if row.get('status') == 'infrastructure_error':
        removed += 1
    else:
        keep.append(line)
if removed:
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(('\n'.join(keep) + '\n') if keep else '', encoding='utf-8')
    os.replace(tmp, p)
print(f'{p.name}: cleared {removed} retryable infrastructure-error record(s)' if removed else f'{p.name}: no infrastructure errors to retry')
PY
}
check_infra() {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists(): raise SystemExit(f'Missing output: {p}')
rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
bad=[r for r in rows if r.get('status') == 'infrastructure_error']
if bad:
    print(f'{p}: {len(bad)} infrastructure-error row(s); first task={bad[0].get("task_index")}, failure={bad[0].get("failure_type")}', file=sys.stderr)
    raise SystemExit(3)
print(f'{p.name}: {len(rows)} records; no model/API infrastructure errors')
PY
}

for f in run2_baseline_200_500.jsonl run3_replay_200_500.jsonl; do retry_infra_rows "$OUT/$f"; done

# Confirm only local Ollama is reachable and has the selected model.
curl -fsS --max-time 5 "$UPSTREAM/models" -o "$OUT/ollama_models.json" || {
  echo "Local Ollama not reachable at $UPSTREAM; start it in WSL and retry." >&2; exit 2;
}
python3 - "$OUT/ollama_models.json" "$MODEL" <<'PY'
import json, sys
names={m.get('id') for m in json.load(open(sys.argv[1], encoding='utf-8')).get('data', [])}
if sys.argv[2] not in names: raise SystemExit(f'{sys.argv[2]} is not available in local Ollama; found {sorted(names)}')
print(f'Local model verified: {sys.argv[2]}')
PY

# Guard limits are sized for this 600-session extension with generous headroom.
# Local Ollama has no hosted-provider rate limit; the guard remains local-only.
export OLLAMA_LOCAL_PLACEHOLDER="local-only-not-a-secret"
export GUARD_KEY_ENV=OLLAMA_LOCAL_PLACEHOLDER
export GUARD_CLIENT_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
export GUARD_UPSTREAM_BASE="$UPSTREAM"
export GUARD_ALLOWED_HOSTS="127.0.0.1:${OLLAMA_PORT}"
export GUARD_ALLOWED_PATH="/v1"
export GUARD_ALLOWED_MODEL="$MODEL"
export GUARD_PORT GUARD_LOG_DIR="$GUARD_DIR"
export GUARD_MAX_REQUESTS=5000
export GUARD_MAX_NEURONS=0
export GUARD_INPUT_NEURONS_PER_MTOK=0
export GUARD_OUTPUT_NEURONS_PER_MTOK=0
export GUARD_MAX_OUTPUT_TOKENS=512
export GUARD_MAX_TOTAL_OUTPUT_TOKENS=1500000
export GUARD_MAX_TOTAL_TOKENS=12000000
export GUARD_MAX_USD=0
export GUARD_INPUT_USD_PER_MTOK=0
export GUARD_OUTPUT_USD_PER_MTOK=0
export GUARD_MAX_PROMPT_CHARS=250000
export GUARD_MAX_MESSAGES=24
export GUARD_TIMEOUT=600
export GUARD_STRIP_SEED=0
export GUARD_ALLOW_GRAMMAR=0

# Refuse to accidentally reuse an unrelated/stale process on the guard port.
python3 - "$GUARD_PORT" <<'PY'
import socket, sys
s=socket.socket()
try:
    s.bind(('127.0.0.1', int(sys.argv[1])))
except OSError as exc:
    raise SystemExit(f'Guard port {sys.argv[1]} is already occupied; inspect the process instead of killing it: {exc}')
finally:
    s.close()
PY
python3 tools/api_guard.py --host 127.0.0.1 --port "$GUARD_PORT" >"$GUARD_DIR/guard.log" 2>&1 &
GUARD_PID=$!
cleanup() { kill "$GUARD_PID" 2>/dev/null || true; wait "$GUARD_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 50); do
  curl -fsS --max-time 1 "http://127.0.0.1:${GUARD_PORT}/health" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS --max-time 2 "http://127.0.0.1:${GUARD_PORT}/health" >/dev/null || { echo "Guard failed; inspect $GUARD_DIR/guard.log" >&2; exit 2; }

# Local-only, unscored preflight through the guard.
curl -fsS --max-time 120 -H "Authorization: Bearer $GUARD_CLIENT_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly OK\"}],\"temperature\":0,\"seed\":0,\"max_tokens\":8}" \
  "$BASE_URL/chat/completions" > "$OUT/preflight.json"
python3 - "$OUT/preflight.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], encoding='utf-8'))
print('Local Ollama preflight:', repr(r['choices'][0]['message']['content'][:80]))
PY
unset OLLAMA_LOCAL_PLACEHOLDER

COMMON=(--mode api --base-url "$BASE_URL" --api-key-env GUARD_CLIENT_TOKEN
  --model-name "$MODEL" --data data/os_tasks.json
  --max-rounds 5 --max-tokens 512 --model-timeout 600 --model-retries 0
  --command-timeout 20 --background-grace 0.5 --resume)
export GUARD_CLIENT_TOKEN

echo '=== Baseline extension: tasks 200-499 ==='
env -u OLLAMA_LOCAL_PLACEHOLDER python3 tools/run_os_cpu.py "${COMMON[@]}" \
  --tasks 200:500 --replay-count 0 --output "$OUT/run2_baseline_200_500.jsonl" \
  > "$OUT/run2_baseline_200_500.log"
check_infra "$OUT/run2_baseline_200_500.jsonl"

echo '=== Replay extension: tasks 200-499; continue from prior replay experience ==='
env -u OLLAMA_LOCAL_PLACEHOLDER python3 tools/run_os_cpu.py "${COMMON[@]}" \
  --tasks 200:500 --replay-count 4 --seed-sessions "$PRIOR/run3_replay.jsonl" \
  --output "$OUT/run3_replay_200_500.jsonl" > "$OUT/run3_replay_200_500.log"
check_infra "$OUT/run3_replay_200_500.jsonl"

# Join prior paired tasks 100-199 with extension 200-499; verify a complete
# 400-task paired set before writing the final paired-test report.
cat "$PRIOR/run2_baseline.jsonl" "$OUT/run2_baseline_200_500.jsonl" > "$OUT/.baseline_100_500.tmp"
mv "$OUT/.baseline_100_500.tmp" "$OUT/baseline_100_500.jsonl"
cat "$PRIOR/run3_replay.jsonl" "$OUT/run3_replay_200_500.jsonl" > "$OUT/.replay_100_500.tmp"
mv "$OUT/.replay_100_500.tmp" "$OUT/replay_100_500.jsonl"
python3 - "$OUT/baseline_100_500.jsonl" "$OUT/replay_100_500.jsonl" <<'PY'
import json, sys
from pathlib import Path
expected=set(range(100,500))
for filename in sys.argv[1:]:
    rows=[json.loads(x) for x in Path(filename).read_text().splitlines() if x.strip()]
    ids=[int(r['task_index']) for r in rows]
    if len(rows)!=400 or set(ids)!=expected or len(set(ids))!=400:
        raise SystemExit(f'{filename}: expected each task 100-499 once; got {len(rows)} rows, {len(set(ids))} distinct IDs')
print('Both combined arms contain the same 400 task IDs, 100-499, exactly once.')
PY
python3 tools/analyze_os_cpu.py "$OUT/baseline_100_500.jsonl" "$OUT/replay_100_500.jsonl" \
  --output "$OUT/paired_comparison_100_500.json"
python3 tools/check_replay_validity.py --warmup "$PRIOR/run1_warmup.jsonl" \
  --replay "$OUT/replay_100_500.jsonl" --replay-count 4 \
  --output "$OUT/replay_validity_100_500.json"
echo "Extension finished. New logs/results: $OUT"
echo "Check guard totals: $GUARD_DIR/budget.json; request log: $GUARD_DIR/requests.jsonl"
