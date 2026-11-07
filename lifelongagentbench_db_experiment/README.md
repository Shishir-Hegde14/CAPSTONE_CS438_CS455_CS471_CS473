# LifelongAgentBench: database tasks with a local Llama 3.1 8B

This folder holds everything needed to rerun our experiment on the database (DB) tasks of
[LifelongAgentBench](https://arxiv.org/abs/2505.11942) (arXiv 2505.11942v3), using the
repository's own agent and a free local model.

**Question:** does showing the agent its own past successful attempts ("experience replay")
help it solve new database tasks?

**Answer from our run:** on the same 100 tasks, the agent solved **19 without examples and 71
with 4 examples**. Most of the gain came from following the required response format. See
[`LifelongAgentBench_Experiment_Report.pdf`](LifelongAgentBench_Experiment_Report.pdf) for the
full write-up.

## What is in this folder

| Path | What it is |
|---|---|
| `patch/lifelongagentbench_changes.patch` | Every change we made to the original repository (26 files). Applies to upstream commit `d6f19b4`. |
| `tools/convert_db_parquet.py` | Converts the published dataset into the JSON files the code reads. |
| `tools/make_option_b_configs.py` | Shows how the three run configs were generated. The patch already contains them, so you don't need to run it. |
| `tools/run_option_b.sh` | Runs runs 1, 2 and 3 in order and stops at the first failure. Rerunning resumes where it stopped. |
| `tools/analyze_option_b.py` | Prints scores, failure types, tokens, latency and the run 2 vs run 3 comparison. |
| `requirements-lock.txt` | Exact Python package versions used. |
| `results/` | Our outputs: `metric.json`, `runs.json` (full transcripts), `usage.jsonl` (per call) and `config.yaml` for each run, plus the perfect-answer check and the logs. |
| `LifelongAgentBench_Experiment_Report.pdf` | Report on the setup, steps, results and limitations. |

## The experiment

| Run | Tasks | Examples in the prompt | Purpose |
|---|---|---|---|
| 1 | 0-99 | None | Collect successful attempts |
| 2 | 100-199 | None | Baseline |
| 3 | 100-199 | 4 most recent successes, seeded from run 1 | Measure the effect of examples |

Runs 2 and 3 use identical tasks in the same order. The only difference is the examples. In
run 3, each new success replaces the oldest example.

## What we used

| Part | Version or setting |
|---|---|
| OS | Windows 11. Linux and macOS should work, but are untested. |
| Python | 3.11.9 |
| Docker | Docker Desktop, engine 29.9.1. The task database is the `mysql:9.2` image. |
| Model server | Ollama 0.40.13 |
| Model | `llama3.1:8b` (4-bit, 4.9 GB, Ollama ID `46e0c10c039e`) |
| Context window | 8,192 tokens (`OLLAMA_CONTEXT_LENGTH=8192`) |
| Decoding | temperature 0, seed 0, max 512 output tokens per reply |
| Hardware | 8-core laptop CPU, 24 GB RAM, 6 GB GPU. The model ran about one-third on CPU and two-thirds on GPU. |

The three runs took about 2 hours in total, and the perfect-answer check about 25 minutes.

## How to reproduce

Commands are for Git Bash on Windows. On Linux or macOS, use `source .venv/bin/activate` instead.

### 1. Install the tools

- [Docker Desktop](https://www.docker.com/products/docker-desktop/), then start it.
- [Ollama](https://ollama.com/download), then download the model:

```bash
docker pull mysql:9.2
ollama pull llama3.1:8b
```

### 2. Get the original code and apply our patch

```bash
git clone https://github.com/caixd-220529/LifelongAgentBench
cd LifelongAgentBench
git config core.longpaths true
git checkout d6f19b4
git apply ../lifelongagentbench_db_experiment/patch/lifelongagentbench_changes.patch
```

Adjust the patch path to wherever this folder is on your machine.

On Windows, the original repo has some very long file names. If `git clone` or `git checkout`
fails with "Filename too long", clone into a short folder (for example `C:\lab`) and use:

```bash
git -c core.longpaths=true clone https://github.com/caixd-220529/LifelongAgentBench
```

### 3. Create the Python environment

Run this inside `LifelongAgentBench`:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
pip install -r ../lifelongagentbench_db_experiment/requirements-lock.txt
```

The CPU build of PyTorch is enough, because the model runs in Ollama, not in Python.

### 4. Download and convert the dataset

```bash
mkdir -p ../hf_dataset/db_bench
curl -L -o ../hf_dataset/db_bench/train-00000-of-00001.parquet \
  https://huggingface.co/datasets/csyq/LifelongAgentBench/resolve/main/db_bench/train-00000-of-00001.parquet
python ../lifelongagentbench_db_experiment/tools/convert_db_parquet.py ../hf_dataset/db_bench/train-00000-of-00001.parquet .
```

Expected output: `entries: 500 | direct: 350 | md5: 150`.

The file we used was 466,381 bytes, with MD5 `e02fcbef180a44c4ce4b73565c18f677`.

### 5. Build the prompt files

```bash
PYTHONPATH=. python src/factories/chat_history_item/offline/construct.py
```

This creates the `chat_history_items/` folder.

### 6. Start Ollama with an 8K context

Quit any Ollama already running, including the tray app, then start it with the context length set:

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

Leave that terminal open. To check it, run `ollama ps` in another terminal after the first
request: it should show `llama3.1:8b` with `CONTEXT 8192`.

### 7. Check the system with perfect answers (optional, about 25 minutes)

This feeds the dataset's correct answers through the full pipeline without any model. All
500 should come out correct.

```bash
PYTHONPATH=. python src/run_experiment.py \
  --config_path configs/assignments/experiments/oracle_check/instance/db_bench/instance/standard.yaml
```

For a quick 5-task test with Llama, use
`configs/assignments/experiments/llama_31_8b_instruct_ollama/instance/db_bench/instance/smoke_5.yaml`.

### 8. Run the experiment (about 2 hours)

```bash
bash ../lifelongagentbench_db_experiment/tools/run_option_b.sh .
```

Results go to `outputs/optb_run1`, `outputs/optb_run2` and `outputs/optb_run3`. Logs go to
`optb_run1.log`, `optb_run2.log` and `optb_run3.log`.

Keep the computer awake and plugged in while it runs. If a run is interrupted, run the same
command again: finished tasks are skipped.

Run 3 starts with examples taken from `outputs/optb_run1/runs.json`, so run 1 must finish first.

### 9. Analyse

```bash
python ../lifelongagentbench_db_experiment/tools/analyze_option_b.py outputs optb_run1 optb_run2 optb_run3
```

To analyse our results without rerunning anything, run this from inside this folder:

```bash
python tools/analyze_option_b.py results optb_run1 optb_run2 optb_run3
```

## What the patch changes

The agent's logic is unchanged. The changes are:

- **Measurement:** `UsageLoggingCallback` (new) and changes to `OpenAILanguageModel` record
  latency, prompt tokens and output tokens for every model call. Each record is linked to the
  task number and written to `callback_state/callback_usage/usage.jsonl`.
- **Starting examples:** `PreviousSampleUtilizationCallback` gets an optional
  `seed_session_list_path`, so a run can start with successes from an earlier run. It uses the
  same selection rule as the original: the most recent sessions that are both correct and
  completed.
- **MySQL:** the image is pinned to `mysql:9.2`, and the database port is bound to
  `127.0.0.1` only.
- **Model:** a `Llama-3.1-8B-Instruct-Ollama` entry points the repo's OpenAI-compatible client
  at Ollama. An OpenRouter entry is also included but was not used. It reads its key from the
  `OPENAI_API_KEY` environment variable, so no key is stored in the code.
- **Configs:** Llama configs (`standard`, `smoke_5`, `optb_run1-3`, and replay with 1, 4, 16,
  32 or 64 examples) and the perfect-answer check (`oracle_check`).

## Our results

| | Run 1: no examples | Run 2: no examples | Run 3: with examples |
|---|---|---|---|
| Tasks | 0-99 | 100-199 | 100-199 |
| **Correct** | **29** | **19** | **71** |
| Finished with an answer | 39 | 31 | 88 |
| Ran out of turns | 36 | 46 | 10 |
| Rejected for bad response format | 25 | 23 | 2 |
| Model calls per task | 2.71 | 2.88 | 2.25 |
| Average prompt tokens per call | 851 | 822 | 2,067 |
| Average output tokens per call | 128 | 114 | 97 |
| Average seconds per call | 8.0 | 6.7 | 7.2 |

On the same tasks, only run 3 was correct on 56 tasks and only run 2 on 4 (sign test
p < 0.001). The perfect-answer check scored 500/500.

**Expect small differences.** Even at temperature 0, Ollama's output can change slightly with
a different GPU, driver or CPU/GPU split. A rerun should land near these numbers, not exactly
on them.

## Known limitations

- Each condition was run once, with one model and one setting of 4 examples. Only tasks 0-199 were used.
- The model is a 4-bit version, and the paper's MySQL version is not confirmed, so scores are
  not directly comparable with the paper.
- Every DB task in this data has difficulty level 0, so the paper's difficulty results can't be reproduced.
- The repo has no ready configs for 2 or 8 examples. Settings of 16 or more need a context
  window larger than 8K.
