# Llama 3.1 8B OS evaluation kit

This kit is for **one separate 100/100/100 OS experiment** using local
`llama3.1:8b`. It uses the
published OS task set and evaluator, Ollama's OpenAI-compatible local API, the
benchmark's original OS prompt, temperature 0, seed 0, max 512 output tokens,
5 rounds, 8K model context, and replay of four successful warm-up experiences.
No grammar or action-first prompt is used. This is the OS task protocol on a
Linux namespace evaluator, not the paper's
full A800/Docker research setup.

## Before downloading the model
This runner requires Linux x86-64 under WSL2, Ubuntu 24.04, Python 3, `curl`,
`unshare`, and `mount`. It runs directly in WSL, not in a regular Docker
container. The evaluator mounts system files into task-local namespaces. On
some WSL setups, an unprivileged namespace can bind-mount directories owned by
the WSL user but not the system-owned `/usr`. The helper below creates a
user-owned snapshot for those mounts. It does **not** use `sudo` or change
system files; it uses several GB of WSL disk. It skips only the inaccessible
`/usr/lib/modules/*/lost+found` recovery directory. The evaluator exposes the
copy read-only inside each task.

Confirm the GPU and base namespace support in Ubuntu/WSL:

```bash
nvidia-smi
unshare --user --map-root-user true
```

Do not install a Linux NVIDIA driver in WSL. If the GPU is not visible, or
`unshare` fails, stop and ask IT rather than changing security settings.

## One-time evaluator setup

Extract this kit under `~/lab_8b_kit`. Then, in Ubuntu/WSL:

```bash
cd ~/lab_8b_kit
bash prepare_wsl_usr.sh
python3 tools/api_guard.py --self-test
python3 tools/run_os_cpu.py --mode oracle --tasks 0:2 \
  --output /tmp/lab_os_oracle_smoke.jsonl --command-timeout 20
```

The oracle check uses published ground-truth commands only to validate the
local evaluator. It makes no model calls. Do not pull the model unless both
tasks complete without `infrastructure_error` or `sandbox_exception`.

## One-time local model setup

1. Install Ollama inside WSL only if IT permits. Use Ollama's official Linux
   instructions; installation may require administrator/root permission. Do
   not use `sudo` if your lab account isn't authorized.
2. In a WSL terminal, set the intended context and start Ollama:

   ```bash
   OLLAMA_CONTEXT_LENGTH=8192 ollama serve
   ```

   Leave that terminal running. In another WSL terminal, pull the model:

   ```bash
   ollama pull llama3.1:8b
   ollama ps
   ```

   Confirm an **8192 context** before evaluating. The model download is about
   5 GB and stays on the lab computer.

## Run the approved block

From the kit directory, in WSL:

```bash
bash run_100_100_100.sh
```

The script verifies Ollama, uses a local-only guard, then checkpoints 300
sessions: warm-up on tasks 0-99, baseline on 100-199, and replay=4 on those
same 100-199 tasks using completed warm-up successes. Rerunning with the same
output directory resumes completed tasks. Plan for several hours; keep the
machine awake and don't launch multiple copies. `Ctrl+C` leaves completed task
records for resuming, and the local guard stops automatically.


## Files included

- `data/os_tasks.json`: published 500-task OS dataset; this block uses indices
  0-199.
- `tools/run_os_cpu.py`, `tools/os_sandbox.py`: evaluator and task environment.
- `tools/api_guard.py`: fixes inference to the local Ollama API and records a
  zero-dollar local budget with hosted-only Neuron accounting disabled.
- `tools/vendor/`: small Linux evaluator helpers and their license/readme.
- `tools/analyze_os_cpu.py`, `tools/check_replay_validity.py`: analysis checks.
- `prepare_wsl_usr.sh`: WSL-only, user-owned copy helper (no sudo).
- `run_100_100_100.sh`: guarded launcher.

No model weights are included.
