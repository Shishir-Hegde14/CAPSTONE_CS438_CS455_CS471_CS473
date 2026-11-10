# LifelongAgentBench: OS tasks with a local Qwen2.5 0.5B model

We ran the OS tasks from [LifelongAgentBench](https://arxiv.org/abs/2505.11942)
on a small local model to see whether experience replay changes accuracy on a
CPU-only machine.

Three 100-session runs:

| Run | Tasks | Examples in the prompt | Purpose |
| --- | --- | --- | --- |
| 1 | 0-99 | none | collect successful attempts |
| 2 | 100-199 | none | baseline |
| 3 | 100-199 | 4 most recent successes from run 1 | measure the effect of examples |

Runs 2 and 3 use the same tasks in the same order. In run 3, each new success
replaces the oldest example.

## Setup

| | |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, Q4_K_M GGUF (0.49B, Apache-2.0) |
| Serving | llama.cpp b12110, CPU only, 2 threads, 8,192-token context |
| Decoding | temperature 0, seed 0, 512 output tokens, 5 rounds per task |
| Hardware | 2 vCPU, 1.9 GiB RAM, no GPU |
| Evaluator | OS tasks in a Linux namespace instead of Docker |
| Dataset | 500 OS tasks, patched copy of task 359 (see below) |
| Output format | action-first prompt and a GBNF grammar |

The three runs took about 31 minutes in total (597 model calls). Exact settings
are in `config.yaml`; the per-run aggregates are in `results/`.

## Our results

| | Run 1: no examples | Run 2: no examples | Run 3: with examples |
| --- | --- | --- | --- |
| Tasks | 0-99 | 100-199 | 100-199 |
| Correct | 12 | 13 | 15 |
| Finished with an answer | 98 | 95 | 98 |
| Ran out of turns | 0 | 0 | 0 |
| Rejected for bad response format | 2 | 5 | 2 |
| Model calls per task | 2.00 | 1.98 | 1.99 |
| Average prompt tokens per call | 352 | 390 | 889 |
| Average output tokens per call | 57 | 60 | 45 |
| Average seconds per call | 2.9 | 3.3 | 2.9 |

"Finished with an answer" is runner status `completed`; "Ran out of turns" is
`task_limit_reached`; "Rejected for bad response format" is
`agent_validation_failed`.

On the same 100 tasks, only run 3 was correct on 5 and only run 2 on 3 (both
10, neither 82). The exact two-sided sign test gives p = 0.7265625, so the +2
difference is not significant. Replay was valid: run 1 gave 12 completed
successes and run 3 started with the expected 4 examples.

## Limitations

This is a 0.49B model on 2 vCPU / 1.9 GiB with no GPU, using an action-first
prompt and a GBNF grammar to keep replies parseable. Those changes are not part
of the original benchmark prompt. The OS evaluator runs in a Linux namespace
instead of Docker. Task 359 of the published dataset has a ground-truth shell
bug, so we used a patched copy (perfect-answer check 500/500 patched, 499/500
original).

Because of the model size, hardware and protocol changes, these numbers are not
comparable to the paper's Llama 3.1 8B / A800 results. The replay difference
here is within noise and should not be read as a replay benefit.
