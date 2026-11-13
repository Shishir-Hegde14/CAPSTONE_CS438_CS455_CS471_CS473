#!/usr/bin/env python3
"""Compare a baseline and replay run from run_os_cpu.py JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load(path: str) -> dict[int, dict[str, Any]]:
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["task_index"])] = record
    return records


def sign_test_p(baseline_only: int, replay_only: int) -> float:
    total = baseline_only + replay_only
    if total == 0:
        return 1.0
    k = min(baseline_only, replay_only)
    return min(1.0, 2.0 * sum(math.comb(total, i) for i in range(k + 1)) / (2**total))


def mean_per_call(records: list[dict[str, Any]]) -> float | None:
    values = [r["model_seconds"] / r["model_calls"] for r in records if r.get("model_calls")]
    return round(statistics.mean(values), 3) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("replay")
    parser.add_argument("--output")
    args = parser.parse_args()

    baseline = load(args.baseline)
    test = load(args.replay)
    common = sorted(set(baseline) & set(test))
    baseline_records = [baseline[i] for i in common]
    replay_records = [test[i] for i in common]

    both = sum(baseline[i]["correct"] and test[i]["correct"] for i in common)
    baseline_only = sum(baseline[i]["correct"] and not test[i]["correct"] for i in common)
    replay_only = sum(not baseline[i]["correct"] and test[i]["correct"] for i in common)
    neither = sum(not baseline[i]["correct"] and not test[i]["correct"] for i in common)

    report = {
        "paired_tasks": len(common),
        "baseline_correct": sum(r["correct"] for r in baseline_records),
        "replay_correct": sum(r["correct"] for r in replay_records),
        "baseline_accuracy": round(sum(r["correct"] for r in baseline_records) / len(common), 4) if common else 0,
        "replay_accuracy": round(sum(r["correct"] for r in replay_records) / len(common), 4) if common else 0,
        "both_correct": both,
        "baseline_only_correct": baseline_only,
        "replay_only_correct": replay_only,
        "neither_correct": neither,
        "exact_two_sided_sign_test_p": sign_test_p(baseline_only, replay_only),
        "baseline_model_timeouts": sum(r.get("failure_type") == "model_timeout" for r in baseline_records),
        "replay_model_timeouts": sum(r.get("failure_type") == "model_timeout" for r in replay_records),
        "baseline_shell_timeouts": sum(r.get("shell_timeouts", 0) for r in baseline_records),
        "replay_shell_timeouts": sum(r.get("shell_timeouts", 0) for r in replay_records),
        "baseline_seconds_per_model_call": mean_per_call(baseline_records),
        "replay_seconds_per_model_call": mean_per_call(replay_records),
        "caveat": "Infrastructure errors are included in the raw counts; inspect them separately before interpreting accuracy.",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
