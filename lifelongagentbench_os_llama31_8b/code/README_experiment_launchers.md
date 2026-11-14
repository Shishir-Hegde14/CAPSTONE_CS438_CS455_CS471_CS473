# Experiment launcher index

These are source/reference scripts only. The completed experiment outputs are documented under the package's `results/` directory. The exp=8 and exp=12 pilot scripts write to separate output folders and stop on the first recorded infrastructure/environment error. Do not rerun them in place if you want to preserve the completed results; set a new `OUT` if intentionally repeating a condition.

- `run_100_100_100.sh`: original local 100 warm-up / 100 baseline / 100 replay=4 block.
- `extend_llama31_8b_os_200_500.sh`: baseline and replay=4 extension on task IDs 200–499; combines the existing task IDs 100–199 for the 400-pair report.
- `run_llama31_8b_exp8_context_pilot.sh`: replay=8 on task IDs 100–199, same 0–99 warm-up, 8K context.
- `run_llama31_8b_exp12_context_pilot.sh`: replay=12 on task IDs 100–199, same 0–99 warm-up, 8K context.

The exp=12 pilot completed with 100 format failures, not an explicit API context error. Its long prompt-character measurements suggest prompt/context overload or truncation may be involved; this is not confirmed. Raw JSONL outputs remain on the WSL system and are not included in this aggregate package.
