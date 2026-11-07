#!/bin/bash
# Runs the three Option B experiments in order and stops at the first failure.
# Rerunning resumes: finished tasks in outputs/optb_runN/runs.json are skipped.
#
# Usage: bash run_option_b.sh <path to LifelongAgentBench repo>
# Uses `python` from PATH (activate the virtual environment first), or set PYTHON.
# Logs go to <repo>/optb_runN.log.
cd "${1:?usage: bash run_option_b.sh <path to LifelongAgentBench repo>}" || exit 1
PYTHON="${PYTHON:-python}"
CONFIG=configs/assignments/experiments/llama_31_8b_instruct_ollama/instance/db_bench/instance
for run in 1 2 3; do
  echo "=== run $run start $(date '+%H:%M:%S')"
  PYTHONPATH=. "$PYTHON" ./src/run_experiment.py \
    --config_path "$CONFIG/optb_run$run.yaml" >> "optb_run$run.log" 2>&1
  code=$?
  echo "=== run $run end $(date '+%H:%M:%S') exit $code"
  [ $code -eq 0 ] || exit $code
done
