#!/usr/bin/env python3
"""CPU-safe runner for the OS part of LifelongAgentBench.

Modes:
  oracle : feed the published ground-truth command through the full evaluator.
           No LLM, no GPU, no API calls.  This validates the environment and
           checks for false shell timeouts.
  api    : sequential LLM-agent run against any OpenAI-compatible endpoint
           (Ollama, llama.cpp server, OpenRouter, ...).  Supports experience
           replay seeded from recent successful sessions.

The runner intentionally separates `command_timeout` (Bash inside the OS) from
`model_timeout` (LLM inference).  A slow CPU model is recorded as
`model_timeout`/`infrastructure_error`, never silently counted as an incorrect
answer.  `--resume` skips completed sessions and rebuilds replay state.

This is a protocol/measurement reproduction, not a reproduction of the paper's
absolute accuracy unless the same 8B model and all 500 tasks are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from os_sandbox import InteractiveSandbox  # noqa: E402

OS_PROMPT = """I will provide you with a task to perform on a Linux (Ubuntu) system. Your objective is to complete the task by executing the appropriate Bash commands.

### Interaction Rules:
1. **Thorough Analysis and Reasoning**:
    - Before performing any action, carefully analyze the task and explain your thought process.
    - Include a detailed explanation of the logic behind your choice of commands and approach.

2. **Action Choices**:
   - At the end of your reasoning, select **one and only one action** for each turn.
     - **"bash"**: When you need to execute a command or perform an operation, provide the corresponding Bash code. Structure your response as:
        Act: bash
        ```bash
        # Your Bash command(s) here
        ```
     - **"finish"**: When the task is complete and no further action is required, conclude with:
        Act: finish

3. **Other Guidelines**:
    - I will use "Act: bash" and "Act: finish" literally to determine whether your action is to execute commands or conclude the task.
    - Use the provided format accurately and consistently.
    - Ensure all Bash commands are compatible with Linux (Ubuntu) systems.
    - Avoid interactive operations (e.g., read, readline) in your Bash commands.

4. **Task Completion**:
    - The task will conclude either when you select the "finish" action or when the number of rounds reaches 5.
    - The system will evaluate your performance to determine if the task was successfully completed."""

REPLAY_SUFFIX = """

{previous_sample_utilization_target_position}

Now, I will give you the question that you need to solve."""
STANDARD_SUFFIX = """

Now, I will give you the question that you need to solve."""

# CPU-time optimisation: a much shorter but still format-compliant prompt.
SHORT_OS_PROMPT = """Solve the Ubuntu shell task in ONE action. Output executable Bash commands only: no prose, no comments, no restating the task. Join all commands with semicolons or && on one line. Reply in exactly this format:
Act: bash
```bash
command1 && command2
```
The evaluator runs immediately after this single action."""

SHORT_REPLAY_SUFFIX = """

Recent solved examples (question + script only):
{previous_sample_utilization_target_position}
Now solve the new task in one action."""

SHORT_STANDARD_SUFFIX = """

Now solve the new task in one action."""


# Disclosed format-first protocol aid: the benchmark defaults above ask for
# "Thorough Analysis and Reasoning" before the action, which a 0.5B model often
# follows so literally that it never emits "Act: bash". This variant forbids
# prose and demands exactly one action. It is not the original upstream prompt.
ACTION_FIRST_OS_PROMPT = """You are an Ubuntu shell task agent. Respond with exactly one action and no other text. No analysis, reasoning, numbered steps, or explanations.

Use exactly one of these two formats:

Act: bash
```bash
your multi-line shell script here
```

or, only when the task is already complete:

Act: finish

Rules:
- On the first turn, always use "Act: bash".
- Include exactly one bash code block per bash action; multi-line commands, comments, semicolons, pipes, and heredocs are allowed.
- Do not put three consecutive backticks inside the script.
- After seeing command output, either issue one more "Act: bash" action or use "Act: finish" when done.
- Never include more than one action in a response."""

ACTION_FIRST_REPLAY_SUFFIX = """

Recent solved examples follow. They show task/action transcripts only; do not copy any prose or formatting from them. Continue to use exactly one action in the format above.
{previous_sample_utilization_target_position}

Now solve the new task. The first response must be "Act: bash".
"""

ACTION_FIRST_STANDARD_SUFFIX = """

Now solve the new task. The first response must be "Act: bash".
"""

# Disclosed environment-aware prompt aid. The local namespace/fakeroot sandbox
# already has root-like privileges but deliberately does not provide sudo,
# systemd, package management, or interactive editors. Small code-tuned models
# otherwise emit `sudo ...`, receive an error, and repeat the same failed
# command for every remaining round. This rule is a system-design fix, not
# ground-truth supervision and not part of the original benchmark prompt.
NO_SUDO_PROMPT_RULE = """Additional environment rules:
- Commands already run with root-like privileges. Never use sudo, su, or doas.
- If a command fails because sudo is missing, retry the plain command without it.
- Prefer coreutils, shell built-ins, and simple text tools. Do not use apt-get, systemctl, service, nano, vi, or vim unless the task explicitly requires them.
- Use the first action to complete the whole task when possible; do not repeat an identical failed command."""


def load_tasks(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = payload["tasks"] if isinstance(payload, dict) else payload
    tasks.sort(key=lambda task: int(task["sample_index"]))
    return tasks


def parse_indices(spec: str, task_count: int) -> list[int]:
    if spec in {"all", "*", ""}:
        return list(range(task_count))
    if ":" in spec:
        start_text, stop_text = spec.split(":", 1)
        start = int(start_text) if start_text else 0
        stop = int(stop_text) if stop_text else task_count
        return list(range(start, min(stop, task_count)))
    if "-" in spec and "," not in spec:
        start_text, stop_text = spec.split("-", 1)
        return list(range(int(start_text), min(int(stop_text) + 1, task_count)))
    return [int(part) for part in spec.split(",") if part.strip()]


def load_seed_sessions(path: str | None, count: int) -> list[dict[str, Any]]:
    if not path or count <= 0:
        return []
    sessions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("correct") and record.get("status") == "completed":
            sessions.append(record)
    return sessions[-count:]


def example_text(
    sessions: list[dict[str, Any]], compact: bool = False, char_limit: int = 600
) -> str:
    if not sessions:
        return "\n"
    parts = ["\n"]
    for session in sessions:
        if compact:
            actions = session.get("actions", []) or []
            script = actions[0].get("script", "") if actions else ""
            if not script:
                script = "\n".join(session.get("transcript", []))
            instruction = str(session.get("instruction", ""))
            text = f"Q: {instruction}\nA: Act: bash\n```bash\n{script}\n```\n"
            parts.append(text[:char_limit] + "\n")
        else:
            transcript = "\n".join(session.get("transcript", []))
            parts.append(f"Question {session.get('instruction', '')}:\n{transcript}\n")
    return "".join(parts)


def initial_prompt(sessions: list[dict[str, Any]], args: argparse.Namespace) -> str:
    replay_count = args.replay_count
    if args.action_first_prompt:
        prefix = ACTION_FIRST_OS_PROMPT
        suffix = ACTION_FIRST_REPLAY_SUFFIX if replay_count > 0 else ACTION_FIRST_STANDARD_SUFFIX
    elif args.short_prompt:
        prefix = SHORT_OS_PROMPT
        suffix = SHORT_REPLAY_SUFFIX if replay_count > 0 else SHORT_STANDARD_SUFFIX
    else:
        prefix = OS_PROMPT
        suffix = REPLAY_SUFFIX if replay_count > 0 else STANDARD_SUFFIX
    if getattr(args, "no_sudo_prompt", False):
        prefix += "\n\n" + NO_SUDO_PROMPT_RULE
    if replay_count > 0:
        return prefix + suffix.replace(
            "{previous_sample_utilization_target_position}",
            example_text(sessions, args.compact_replay, args.replay_char_limit),
        )
    return prefix + suffix


def parse_agent_response(text: str, tolerant_one_pass: bool = False) -> dict[str, Any]:
    # Some llama.cpp GBNF builds emit literal backslash-n inside the forced
    # string. Normalize it before parsing when the one-pass optimization is on.
    if tolerant_one_pass:
        text = text.replace("\\n", "\n")
    action_match = re.search(r"Act:\s*(.+)", text)
    if action_match is None:
        return {
            "action": "invalid",
            "content": None,
            "reason": 'Cannot extract action from agent response. Pattern: "Act:\\s*(.+)"',
        }
    action_str = action_match.group(1)
    if action_str.lower().startswith("bash"):
        content_list = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
        content = "\n\n".join(content_list)
        if not content.strip() and tolerant_one_pass:
            # A 64-token cap can truncate the closing fence. Take the script
            # tail anyway so it is evaluated rather than discarded as a format
            # violation; the evaluator still decides correctness.
            match = re.search(r"```bash\n(.*)", text, re.DOTALL)
            if match:
                content = match.group(1).strip()
        if not content.strip():
            return {"action": "invalid", "content": None, "reason": "Empty bash code block."}
        return {"action": "execute", "content": content, "reason": None}
    if action_str.lower().startswith("finish"):
        return {"action": "finish", "content": None, "reason": None}
    return {
        "action": "invalid",
        "content": None,
        "reason": 'Invalid action string matched by pattern "Act:\\s*(.+)"',
    }


def repair_agent_response(text: str) -> dict[str, Any]:
    """Recover common Qwen format slips without changing the model output.

    Handles literal backslash-n, a missing `Act: bash` wrapper when a code block
    is present, and a missing closing code fence. This is a disclosed protocol
    aid, not part of the original upstream benchmark.
    """
    text = text.replace("\\n", "\n")
    if not re.search(r"Act:\s*", text, re.I) and "```" in text:
        text = "Act: bash\n" + text
    parsed = parse_agent_response(text, tolerant_one_pass=True)
    if parsed["action"] == "invalid":
        return parsed
    return {**parsed, "repaired": True}


def call_model(messages: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    url = base_url + "/chat/completions"
    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": args.model_name,
        "messages": messages,
        "temperature": 0,
        "seed": 0,
        "max_tokens": args.max_tokens,
    }
    if args.grammar:
        body["grammar"] = args.grammar
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.model_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {}) or {}
        return {
            "ok": True,
            "text": text,
            "seconds": time.perf_counter() - started,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "error_type": "model_http_error",
            "error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:1000]}",
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001
        is_timeout = (
            isinstance(exc, (TimeoutError, socket.timeout))
            or "timed out" in str(exc).lower()
            or "timeout" in str(exc).lower()
        )
        error_type = "model_timeout" if is_timeout else "model_error"
        return {
            "ok": False,
            "error_type": error_type,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.perf_counter() - started,
        }


def oracle_response(task: dict[str, Any], round_index: int) -> str:
    if round_index == 0:
        script = task["evaluation_info"]["ground_truth_command_item"]["script"]
        return f"Think: Hello world. Act: bash\n```bash\n{script}\n```"
    return "Think: Hello world. Act: finish."


def run_session(
    task: dict[str, Any],
    args: argparse.Namespace,
    replay_sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "task_index": int(task["sample_index"]),
        "instruction": task["instruction"],
        "skills": task["skill_list"],
        "mode": args.mode,
        "replay_count": args.replay_count,
        "replay_task_indices": [s.get("task_index") for s in replay_sessions],
        "status": "unknown",
        "correct": False,
        "failure_type": None,
        "rounds": 0,
        "model_calls": 0,
        "model_seconds": 0.0,
        "command_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "shell_timeouts": 0,
        "format_repairs": 0,
        "transcript": [],
        "actions": [],
        "evaluation": None,
        "sandbox_error": None,
        "auto_finish": False,
    }
    sandbox: InteractiveSandbox | None = None
    try:
        sandbox = InteractiveSandbox(command_timeout=args.command_timeout)
        init_script = task["initialization_command_item"]["script"]
        init_result = sandbox.execute(init_script, timeout=args.command_timeout + 10)
        record["initialization"] = init_result
        record["command_seconds"] += float(init_result.get("seconds", 0.0))
        record["shell_timeouts"] += int(bool(init_result.get("timeout_flag")))
        if init_result.get("timeout_flag") or init_result.get("exit_code") != 0:
            record["status"] = "environment_error"
            record["failure_type"] = (
                "initialization_timeout" if init_result.get("timeout_flag") else "initialization_failed"
            )
            return record

        prompt = initial_prompt(replay_sessions, args)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": task["instruction"]},
        ]

        for round_index in range(args.max_rounds):
            record["rounds"] = round_index + 1
            if args.mode == "oracle":
                response = oracle_response(task, round_index)
                model_info = {"ok": True, "text": response, "seconds": 0.0}
            else:
                model_info = None
                for attempt in range(args.model_retries + 1):
                    candidate = call_model(messages, args)
                    record["model_calls"] += 1
                    record["model_seconds"] += float(candidate.get("seconds", 0.0))
                    if candidate.get("ok"):
                        model_info = candidate
                        break
                    if attempt == args.model_retries:
                        model_info = candidate
                assert model_info is not None
                if not model_info.get("ok"):
                    record["status"] = "infrastructure_error"
                    record["failure_type"] = model_info.get("error_type", "model_error")
                    record["model_error"] = model_info.get("error", "")
                    return record
                response = model_info["text"]
                record["prompt_tokens"] += int(model_info.get("prompt_tokens") or 0)
                record["completion_tokens"] += int(model_info.get("completion_tokens") or 0)

            messages.append({"role": "assistant", "content": response})
            record["transcript"].append(f"assistant: {response}")
            parsed = parse_agent_response(response, args.one_pass)
            if parsed["action"] == "invalid" and args.repair_format:
                repaired = repair_agent_response(response)
                if repaired["action"] != "invalid":
                    parsed = repaired
                    record["format_repairs"] += 1
            if parsed["action"] == "invalid":
                record["status"] = "agent_validation_failed"
                record["failure_type"] = "format_violation"
                record["finish_reason"] = parsed["reason"]
                break
            if parsed["action"] == "finish":
                record["status"] = "completed"
                break

            action_result = sandbox.execute(parsed["content"], timeout=args.command_timeout + 10)
            record["actions"].append(
                {
                    "script": parsed["content"],
                    "exit_code": action_result.get("exit_code"),
                    "timeout_flag": action_result.get("timeout_flag"),
                    "seconds": action_result.get("seconds"),
                    "output": action_result.get("output"),
                }
            )
            record["command_seconds"] += float(action_result.get("seconds", 0.0))
            record["shell_timeouts"] += int(bool(action_result.get("timeout_flag")))
            # A few tasks background a whole shell chain.  Give it a brief head
            # start before the evaluator begins its own sleeps, avoiding a
            # scheduler race that is unrelated to agent capability.
            if args.background_grace > 0 and parsed["content"].rstrip().endswith("&"):
                time.sleep(args.background_grace)
                record["background_grace_seconds"] = args.background_grace
            observation = (
                f"The command is marked as timeout since it did not finish within "
                f"{args.command_timeout} seconds."
                if action_result.get("timeout_flag")
                else (action_result.get("output") or "")
            )
            observation_message = (
                f"The output of the OS:\n\n{observation}"
                if observation
                else "The output of the OS is empty."
            )
            messages.append({"role": "user", "content": observation_message})
            record["transcript"].append(f"user: {observation_message}")
            if args.one_pass:
                # CPU-time optimisation: evaluate immediately after the first
                # executed action instead of spending another slow model call
                # on an "Act: finish" turn.
                record["status"] = "completed"
                record["auto_finish"] = True
                record["transcript"].append(
                    "assistant: Act: finish (automatic after one-pass action)"
                )
                break
        else:
            record["status"] = "task_limit_reached"
            record["failure_type"] = "failure_to_commit"

        evaluation_script = task["evaluation_info"]["evaluation_command_item"]["script"]
        eval_result = sandbox.execute(evaluation_script, timeout=args.command_timeout + 10)
        record["evaluation"] = eval_result
        record["command_seconds"] += float(eval_result.get("seconds", 0.0))
        record["shell_timeouts"] += int(bool(eval_result.get("timeout_flag")))
        if eval_result.get("timeout_flag"):
            record["status"] = "environment_error"
            record["failure_type"] = "evaluation_timeout"
        else:
            record["correct"] = eval_result.get("exit_code") == 0
            if not record["correct"] and record["status"] == "completed":
                record["failure_type"] = "incorrect_final_submission"
    except Exception as exc:  # noqa: BLE001
        record["status"] = "infrastructure_error"
        record["failure_type"] = "sandbox_exception"
        record["sandbox_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sandbox:
            meta = sandbox.close()
            record["sandbox_wall_seconds"] = meta.get("wall_seconds")
            if meta.get("outer_log"):
                record["sandbox_outer_log"] = meta["outer_log"]
        record["wall_seconds"] = round(time.perf_counter() - started, 4)
    return record


def summarize(records: list[dict[str, Any]], args: argparse.Namespace, dataset_path: str) -> dict[str, Any]:
    failures: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for record in records:
        failures[record.get("failure_type") or "none"] = failures.get(record.get("failure_type") or "none", 0) + 1
        statuses[record.get("status", "unknown")] = statuses.get(record.get("status", "unknown"), 0) + 1
    model_seconds = [r["model_seconds"] / r["model_calls"] for r in records if r.get("model_calls")]
    command_seconds = [r["command_seconds"] for r in records]
    return {
        "mode": args.mode,
        "dataset": dataset_path,
        "task_indices": [r["task_index"] for r in records],
        "task_count": len(records),
        "replay_count": args.replay_count,
        "correct": sum(bool(r.get("correct")) for r in records),
        "accuracy": round(sum(bool(r.get("correct")) for r in records) / len(records), 4) if records else 0,
        "status_counts": statuses,
        "failure_counts": failures,
        "shell_timeouts": sum(int(r.get("shell_timeouts", 0)) for r in records),
        "model_timeouts": sum(1 for r in records if r.get("failure_type") == "model_timeout"),
        "infrastructure_errors": sum(1 for r in records if r.get("status") == "infrastructure_error"),
        "model_calls": sum(int(r.get("model_calls", 0)) for r in records),
        "format_repairs": sum(int(r.get("format_repairs", 0)) for r in records),
        "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in records),
        "completion_tokens": sum(int(r.get("completion_tokens", 0)) for r in records),
        "mean_model_seconds_per_call": round(statistics.mean(model_seconds), 3) if model_seconds else None,
        "mean_command_seconds_per_task": round(statistics.mean(command_seconds), 3) if command_seconds else None,
        "total_wall_seconds": round(sum(float(r.get("wall_seconds", 0.0)) for r in records), 3),
        "settings": {
            "max_rounds": args.max_rounds,
            "command_timeout": args.command_timeout,
            "model_timeout": args.model_timeout,
            "model_retries": args.model_retries,
            "max_tokens": args.max_tokens,
            "grammar": bool(args.grammar),
            "grammar_file": args.grammar_file,
            "grammar_sha256": (
                hashlib.sha256(args.grammar.encode("utf-8")).hexdigest()
                if args.grammar
                else None
            ),
            "one_pass": args.one_pass,
            "action_first_prompt": args.action_first_prompt,
            "no_sudo_prompt": args.no_sudo_prompt,
            "repair_format": args.repair_format,
            "short_prompt": args.short_prompt,
            "compact_replay": args.compact_replay,
            "replay_char_limit": args.replay_char_limit,
            "background_grace": args.background_grace,
            "base_url": args.base_url if args.mode == "api" else None,
            "model_name": args.model_name if args.mode == "api" else None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


def exact_sign_test_p(baseline_only: int, replay_only: int) -> float:
    n = baseline_only + replay_only
    if n == 0:
        return 1.0
    k = min(baseline_only, replay_only)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/os_tasks.json")
    parser.add_argument("--mode", choices=["oracle", "api"], default="oracle")
    parser.add_argument("--tasks", default="all", help="all, start:stop, start-stop, or comma-separated indices")
    parser.add_argument("--replay-count", type=int, default=0)
    parser.add_argument("--seed-sessions", help="JSONL from an earlier no-replay run; used as recent successes")
    parser.add_argument("--output", required=True, help="JSONL session output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--command-timeout", type=float, default=20.0)
    parser.add_argument("--model-timeout", type=float, default=300.0)
    parser.add_argument("--model-retries", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--grammar",
        help="optional GBNF grammar text to constrain the model response (llama.cpp servers)",
    )
    parser.add_argument(
        "--grammar-file",
        help="path to a GBNF grammar file; read into --grammar for llama.cpp servers",
    )
    parser.add_argument(
        "--action-first-prompt",
        action="store_true",
        help="disclosed aid: no-prose prompt that demands exactly one Act: bash/finish action",
    )
    parser.add_argument(
        "--no-sudo-prompt",
        action="store_true",
        help="disclosed aid: tell the agent it already has root-like privileges and must not use sudo/systemd/interactive editors",
    )
    parser.add_argument(
        "--one-pass",
        action="store_true",
        help="evaluate after the first executed action; no second finish call",
    )
    parser.add_argument(
        "--repair-format",
        action="store_true",
        help="disclosed aid: recover missing Act: bash wrappers and truncated code fences",
    )
    parser.add_argument("--short-prompt", action="store_true", help="use compact format prompt")
    parser.add_argument(
        "--compact-replay",
        action="store_true",
        help="use question+first-script only for replay examples",
    )
    parser.add_argument(
        "--replay-char-limit",
        type=int,
        default=600,
        help="max characters per compact replay example (default: 600)",
    )
    parser.add_argument(
        "--background-grace",
        type=float,
        default=0.5,
        help="seconds to wait after an action whose script ends with '&' (default: 0.5)",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-name", default="llama3.1:8b")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    if args.grammar_file:
        args.grammar = Path(args.grammar_file).read_text(encoding="utf-8")

    tasks = load_tasks(args.data)
    indices = parse_indices(args.tasks, len(tasks))
    task_by_index = {int(task["sample_index"]): task for task in tasks}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
    completed = {int(record["task_index"]) for record in existing}
    all_records = list(existing)

    seed = load_seed_sessions(args.seed_sessions, args.replay_count)
    successes = list(seed)
    if args.resume:
        # Rebuild replay state from all previously completed successful sessions.
        successes = seed + [
            record
            for record in existing
            if record.get("correct") and record.get("status") == "completed"
        ]
    successes = successes[-args.replay_count :] if args.replay_count > 0 else []

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for index in indices:
            if index in completed:
                continue
            replay_sessions = successes[-args.replay_count :] if args.replay_count > 0 else []
            record = run_session(task_by_index[index], args, replay_sessions)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            all_records.append(record)
            if record.get("correct") and record.get("status") == "completed" and args.replay_count > 0:
                successes.append(record)
                successes = successes[-args.replay_count :]

    summary = summarize(all_records, args, str(Path(args.data).resolve()))
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
