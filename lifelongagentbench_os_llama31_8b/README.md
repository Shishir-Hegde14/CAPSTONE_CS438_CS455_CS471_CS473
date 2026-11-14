# Llama 3.1 8B OS experiments (local Ollama / WSL2)

This folder documents one warm-up, the 100/100/100 initial block, the expanded 400-task paired comparison, and replay-size pilots at exp=8 and exp=12. It includes the evaluator/launcher source and aggregate reports. No model weights, API keys, or paid-provider calls are included.

## Results at a glance

| Experiment | Baseline | Replay | Paired exact two-sided sign test | Notes |
|---|---:|---:|---:|---|
| Initial tasks 100–199, exp=4 | 34/100 | 45/100 | p=0.05224 | Not significant at 0.05 |
| Pooled tasks 100–499, exp=4 | 141/400 | 173/400 | p=0.002685 | Exploratory: extension followed an interim look at the first 100 tasks |
| Added tasks 200–499, exp=4 | 107/300 | 128/300 | p=0.02565 | Same direction; also exploratory |
| Tasks 100–199, exp=8 vs baseline | 34/100 | 46/100 | p=0.04277 | Nominal comparison to baseline; exploratory |
| Tasks 100–199, exp=8 vs exp=4 | 45/100 | 46/100 | p=1.0 | No evidence exp=8 improves over exp=4 |
| Tasks 100–199, exp=12 vs baseline | 34/100 | 0/100 | p=1.164e-10 | All exp=12 sessions were format failures; not a valid performance comparison |
| Tasks 100–199, exp=12 vs exp=4 | 45/100 | 0/100 | p=5.684e-14 | All exp=12 sessions were format failures; not a valid performance comparison |

Correctness is the final task-evaluator outcome and is reported separately from normal completion. Exp=8 had 46 raw final-evaluator passes but only 24 completed-correct successes; exp=12 had zero passes and 100/100 format failures.

## Status and reliability

### Primary exp=4 comparison, tasks 100–499

| Condition | Completed | Task limit | Format/validation failure | Final-evaluator correct | Model timeouts | Shell timeouts |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 89/400 | 237/400 | 74/400 | 141/400 | 0 | 9 |
| Replay, exp=4 | 276/400 | 66/400 | 58/400 | 173/400 | 0 | 0 |

### Replay-size pilots, tasks 100–199

| Condition | Completed | Task limit | Format/validation failure | Final-evaluator correct | Model timeouts |
|---|---:|---:|---:|---:|---:|
| Replay, exp=4 | 76/100 | 16/100 | 8/100 | 45/100 | 0 |
| Replay, exp=8 | 47/100 | 20/100 | 33/100 | 46/100 | 0 |
| Replay, exp=12 | 0/100 | 0/100 | 100/100 | 0/100 | 0 |

The exp=12 pilot made one model call per task and recorded no infrastructure errors or explicit API context errors. Its prompts were much longer: guard prompt-character median/p95/max was 42,479/42,601/42,737 for exp=12 versus 30,367/35,551/37,767 for exp=8. Ollama reported an 8,192 context after the pilot. The OpenAI-compatible usage counter reported only 8,006 prompt tokens total for exp=12, inconsistent with the prompt-character size; treat that token-usage number as unreliable. Prompt/context overload or truncation is plausible, but an explicit context overflow was not recorded. Do not interpret exp=12’s zero score as a clean replay-performance estimate.

## Replay validity and scope

The warm-up had 21 completed correct successes. Exp=4 replay began with four examples; replay validity is true. Exp=8 and exp=12 used the same warm-up tasks and the existing baseline tasks 100–199; each had its own output directory. The exp=12 pilot used 12 warm-up examples.

This is a local OS-task evaluation using Ollama `llama3.1:8b` Q4_K_M on an RTX 4500 Ada under WSL2 Ubuntu 24.04. It is **not** a DB/MySQL run, is **not** paper-equivalent, and is not directly comparable to the paper’s A800 setup. The 400-task p-value and replay-size pilots are exploratory, not preregistered confirmatory results.

## Files

- `code/`: evaluator, task set, original launcher, and extension/exp-size pilot launchers.
- `results/`: aggregate run summaries, paired comparisons, status counts, replay-validity checks, and prompt-size diagnostics.
- `config.yaml`: protocol summary; `config_exp8.yaml` and `config_exp12.yaml`: pilot settings.

Raw per-task JSONL transcripts remain on the WSL experiment machine and are not included in this aggregate package. Preserve those separately if you need the full per-task record.
