"""Summarise the Option B runs from runs.json and usage.jsonl.

Usage:
    python analyze_option_b.py <outputs folder> optb_run1 optb_run2 optb_run3
"""

import json
import os
import statistics
import sys

OUTPUTS = sys.argv[1] if len(sys.argv) > 1 else "outputs"
CONTEXT_LIMIT = 8192


def load(run):
    with open(f"{OUTPUTS}/{run}/runs.json", encoding="utf-8") as f:
        sessions = json.load(f)
    usage = []
    path = f"{OUTPUTS}/{run}/callback_state/callback_usage/usage.jsonl"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            usage = [json.loads(line) for line in f if line.strip()]
    return sessions, usage


def outcome(session):
    return session["evaluation_record"]["outcome"]


def summarise(run):
    sessions, usage = load(run)
    correct = sum(outcome(s) == "correct" for s in sessions)
    print(f"\n{run}: {correct}/{len(sessions)} correct")
    statuses = {}
    for s in sessions:
        statuses[s["sample_status"]] = statuses.get(s["sample_status"], 0) + 1
    print("  status:", statuses)
    if usage:
        prompt = [u["prompt_tokens"] for u in usage]
        completion = [u["completion_tokens"] for u in usage]
        latency = [u["latency_seconds"] for u in usage]
        print(f"  calls: {len(usage)} ({len(usage) / len(sessions):.2f} per task)")
        print(f"  prompt tokens: mean {statistics.mean(prompt):.0f} max {max(prompt)}")
        print(f"  completion tokens: mean {statistics.mean(completion):.0f} max {max(completion)}")
        print(f"  latency s: mean {statistics.mean(latency):.1f} max {max(latency):.1f}")
        near = sum(p >= CONTEXT_LIMIT * 0.95 for p in prompt)
        print(f"  prompts at 95%+ of {CONTEXT_LIMIT}: {near}")
    return {str(s["sample_index"]): outcome(s) == "correct" for s in sessions}


def paired(a, b):
    both = sorted(set(a) & set(b), key=int)
    only_a = sum(a[i] and not b[i] for i in both)
    only_b = sum(b[i] and not a[i] for i in both)
    same = len(both) - only_a - only_b
    print(f"\npaired on {len(both)} tasks: only first right {only_a}, only second right {only_b}, same {same}")
    discordant = only_a + only_b
    if discordant:
        # Two-sided exact sign test on the tasks where the two runs disagree.
        from math import comb

        tail = sum(comb(discordant, k) for k in range(min(only_a, only_b) + 1))
        print(f"  sign test p = {min(1.0, 2 * tail / 2 ** discordant):.3f}")


if __name__ == "__main__":
    results = {run: summarise(run) for run in sys.argv[2:]}
    if "optb_run2" in results and "optb_run3" in results:
        paired(results["optb_run2"], results["optb_run3"])
